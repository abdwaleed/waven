"""Video downsampling and Gabor wavelet decomposition.

Functions here consume stimulus movies or downsampled movie arrays and write
wavelet coefficient arrays to disk. Filter-bank construction lives in
:mod:`waven.wavelets.filters`.
"""
import gc
import json
import math
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torch.nn.functional as F
from skimage.filters import gabor_kernel
from tqdm import tqdm

from ..runtime.performance import (
    ConvolutionWorkload,
    OperationTelemetry,
    amp_enabled,
    available_ram_bytes,
    configure_zarr_codec_threads,
    configure_torch_cpu_threads,
    convolution_precision_scope,
    autotuned_frame_chunk_size,
    cpu_worker_count,
    enabled_feature,
    gpu_available_vram_bytes,
    gpu_vram_bytes,
    plan_convolution_execution,
    resolve_compute_device,
    torch_compile_enabled,
    video_downsample_chunk_size,
    wavelet_filter_chunk_size,
)
from ..runtime.task_control import check_cancelled, progress_message
from ..stimulus.metadata import coverage_crop_bounds
from .filters import has_enough_ram


def _array_bytes(shape, dtype=np.float32):
    """Return exact bytes for an array shape/dtype pair."""
    return int(math.prod(tuple(int(v) for v in shape)) * np.dtype(dtype).itemsize)


def _host_convolution_output(value):
    """Detach one tensor or a tuple of tensors into NumPy output payloads."""
    if isinstance(value, tuple):
        return tuple(_host_convolution_output(item) for item in value)
    return value.cpu().numpy()


def _convolution_output_bytes(value):
    """Return the resident host size of a convolution output payload."""
    if isinstance(value, tuple):
        return sum(_convolution_output_bytes(item) for item in value)
    return int(value.nbytes)


def _concatenate_convolution_outputs(outputs):
    """Concatenate frame shards without losing multi-product payload structure."""
    first = outputs[0]
    if isinstance(first, tuple):
        return tuple(
            np.concatenate([output[index] for output in outputs], axis=0)
            for index in range(len(first))
        )
    return np.concatenate(outputs, axis=0)


def _is_cuda_oom(exc):
    """Return whether an exception is a recoverable CUDA allocation failure."""
    return str(exc).lower().find("out of memory") >= 0


def _release_cuda_working_set(devices=None):
    """Release failed-workspace allocations before a smaller retry.

    Workload-aware frame sharding can have tensors on more than one CUDA
    device.  Empty every participating cache rather than only the default GPU,
    otherwise a retry can fail on a peer that still retains its failed batch.
    """
    if not torch.cuda.is_available():
        return
    gc.collect()
    if devices is None:
        devices = tuple(range(torch.cuda.device_count()))
    elif isinstance(devices, (str, int)):
        devices = (devices,)
    for device in devices:
        try:
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            # A device can disappear during teardown on some desktop CUDA
            # stacks; cache cleanup should never mask the original failure.
            pass


def _resource_aware_frame_chunk_size(
    num_frames,
    nx,
    ny,
    filter_count,
    output_channels,
    device,
    kernel_shape=None,
    output_buffer_count=1,
):
    """Choose a frame batch from the complete convolution working set.

    The former generic autotuner considered only the input movie.  For a Gabor
    bank, the convolution response and post-processing buffers scale with every
    active parameter: orientation, sigma/frequency group, phase count, and the
    output grid.  Estimating all live tensors keeps high-resolution banks from
    selecting an input-safe but output-impossible batch.
    """
    frames = max(1, int(num_frames))
    pixels = max(1, int(nx) * int(ny))
    filter_count = max(1, int(filter_count))
    output_channels = max(1, int(output_channels))
    output_buffer_count = max(1, int(output_buffer_count))
    output_dtype_bytes = 2 if str(device).startswith("cuda") and amp_enabled() else 4
    # Input remains float32.  Response, squared/summed output, permutation,
    # and transfer staging use the active convolution precision.
    bytes_per_frame = pixels * (
        np.dtype(np.float32).itemsize
        + output_dtype_bytes * (filter_count + output_channels * output_buffer_count)
    )
    if kernel_shape is None:
        kernel_bytes = 0
    else:
        kernel_height, kernel_width = (max(1, int(value)) for value in kernel_shape)
        # Include the module copy plus a modest cuDNN/kernel-workspace reserve.
        kernel_bytes = filter_count * kernel_height * kernel_width * np.dtype(np.float32).itemsize * 2
    if str(device).startswith("cuda"):
        # A 40% free-VRAM budget leaves room for cuDNN workspace, the display
        # driver, asynchronous copies, and an interactive GUI.
        budget = int(gpu_available_vram_bytes() * 0.40) - kernel_bytes
    else:
        # CPU convolution may still hold NumPy input, Torch output, and writer
        # payloads at once; retain substantial OS/file-cache headroom.
        budget = int(available_ram_bytes() * 0.20) - kernel_bytes
    return max(1, min(frames, max(1, budget // max(1, bytes_per_frame))))


def _coarse_power_zarr_layout(
    final_shape,
    frame_chunk_size,
    filter_group_size,
    target_bytes=96 * 1024**2,
    rf_neuron_count=None,
):
    """Choose write-oriented, RF-read-aligned chunks for coarse power.

    Each direct-convolution result covers one time range and one sigma group.
    Matching those dimensions prevents read/modify/recompress writes.  Spatial
    chunks target roughly 96 MiB of raw data (with a small alignment allowance)
    so Zarr performs substantially fewer codec/store operations than the former
    fixed 16 x 16 layout, while remaining comfortably bounded for a writer.
    Coarse-RF correlation consumes the same spatial chunks as tiles.  The
    spatial target also accounts for current free VRAM, the complete Gabor
    feature bank, and the neural population.  It only shrinks a storage tile
    when it would otherwise require more than three GPU subtiles: smaller
    chunks can reduce GPU transfer cost, but too many extra Zarr chunks would
    make the I/O-bound correlation stage slower.
    """
    if len(final_shape) == 5:
        num_frames, nx, ny, n_orientations, n_sigmas = (int(value) for value in final_shape)
        n_frequencies = 1
    elif len(final_shape) == 6:
        num_frames, nx, ny, n_orientations, n_sigmas, n_frequencies = (
            int(value) for value in final_shape
        )
    else:
        raise ValueError(
            "Coarse RF power layout expects (time, x, y, orientation, sigma) "
            "or (time, x, y, orientation, sigma, frequency)."
        )
    time_chunk = min(num_frames, max(1, int(frame_chunk_size)))
    sigma_chunk = min(n_sigmas, max(1, int(filter_group_size)))
    bytes_per_pixel = max(
        1,
        time_chunk * n_orientations * sigma_chunk * n_frequencies * np.dtype(np.float32).itemsize,
    )
    storage_pixel_budget = max(1, int(target_bytes) // bytes_per_pixel)
    pixel_budget = storage_pixel_budget
    gpu_pixel_budget = None
    allowed_gpu_subtiles = 1
    free_vram_bytes = gpu_available_vram_bytes() if torch.cuda.is_available() else 0
    # A nearly exhausted GPU cannot provide a useful stable cache-layout
    # signal.  Preserve the I/O-efficient storage layout and let the runtime
    # RF path safely downshift or use CPU for that exceptional session.
    if free_vram_bytes >= 2 * 1024**3:
        # RF correlation holds a persistent feature-by-neuron accumulator, the
        # current GEMM result, and one time-by-feature block in float64.  Use
        # the same 60% free-VRAM ceiling as the runtime RF planner so cache
        # preparation and analysis agree about what a useful tile looks like.
        n_neurons = max(1, int(rf_neuron_count or 256))
        full_feature_bytes_per_pixel = np.dtype(np.float64).itemsize * int(n_orientations) * int(n_sigmas) * int(n_frequencies) * (
            n_neurons * 2 + time_chunk
        )
        gpu_pixel_budget = max(
            1,
            int(free_vram_bytes * 0.60) // max(1, full_feature_bytes_per_pixel),
        )
        # A one-tile GPU path is worthwhile when it does not nearly double the
        # number of spatial Zarr chunks.  Otherwise retain up to three bounded
        # GPU subtiles, preserving sequential compressed-cache throughput.
        if gpu_pixel_budget < storage_pixel_budget * 0.60:
            allowed_gpu_subtiles = 3
        pixel_budget = min(storage_pixel_budget, gpu_pixel_budget * allowed_gpu_subtiles)

    def spatial_candidates(limit):
        values = {1, int(limit)}
        values.update(range(8, int(limit) + 1, 8))
        return tuple(sorted(values))

    allowance = max(1, int(pixel_budget * 1.25))
    candidates = []
    for x_chunk in spatial_candidates(nx):
        for y_chunk in spatial_candidates(ny):
            pixels = x_chunk * y_chunk
            if pixels <= allowance:
                tile_count = math.ceil(nx / x_chunk) * math.ceil(ny / y_chunk)
                candidates.append((tile_count, -pixels, x_chunk, y_chunk))
    if not candidates:
        x_chunk, y_chunk = 1, 1
    else:
        _tile_count, _negative_pixels, x_chunk, y_chunk = min(candidates)
    if gpu_pixel_budget is not None:
        planned_features = x_chunk * y_chunk * n_orientations * n_sigmas * n_frequencies
        estimated_subtiles = math.ceil((x_chunk * y_chunk) / max(1, gpu_pixel_budget))
        print(
            "Coarse RF Zarr tile planner | "
            f"free VRAM={free_vram_bytes / 1024**3:.2f} GiB, "
            f"neurons={rf_neuron_count or 256}, filters={n_orientations}x{n_sigmas}x{n_frequencies}, "
            f"tile={x_chunk}x{y_chunk} ({planned_features:,} features), "
            f"estimated GPU subtiles={estimated_subtiles} (limit={allowed_gpu_subtiles})."
        )
    layout = (time_chunk, x_chunk, y_chunk, n_orientations, sigma_chunk)
    return layout if len(final_shape) == 5 else layout + (n_frequencies,)


def coarse_rf_zarr_layout(final_shape, frame_chunk_size, filter_group_size, rf_neuron_count=None):
    """Public coarse-RF cache layout planner shared by both backends."""
    return _coarse_power_zarr_layout(
        final_shape,
        frame_chunk_size,
        filter_group_size,
        rf_neuron_count=rf_neuron_count,
    )


def _full_convolution_zarr_layout(
    final_shape,
    frame_chunk_size,
    filter_group_size,
    target_bytes=96 * 1024**2,
):
    """Choose full-model Zarr chunks that match its grouped write pattern.

    Full-model convolution writes one sigma/frequency result at a time.  Sigma
    and frequency chunks must therefore remain one, avoiding a read,
    decompression, and recompression of every other filter result on each
    assignment.  Spatial chunks use the remaining payload budget.
    """
    num_frames, nx, ny, n_orientations, _n_sigmas, _n_frequencies = (
        int(value) for value in final_shape
    )
    time_chunk = min(num_frames, max(1, int(frame_chunk_size)))
    # A group can contain several independent writes, but no Zarr chunk contains
    # more than one sigma/frequency result.  Size each physical chunk from one
    # such result rather than making all chunks needlessly smaller for a larger
    # GPU filter group.
    group_size = 1
    bytes_per_pixel = max(
        1,
        time_chunk * n_orientations * group_size * np.dtype(np.float32).itemsize,
    )
    pixel_budget = max(1, int(target_bytes) // bytes_per_pixel)

    def spatial_candidates(limit):
        values = {1, int(limit)}
        values.update(range(8, int(limit) + 1, 8))
        return tuple(sorted(values))

    allowance = max(1, int(pixel_budget * 1.25))
    candidates = []
    for x_chunk in spatial_candidates(nx):
        for y_chunk in spatial_candidates(ny):
            pixels = x_chunk * y_chunk
            if pixels <= allowance:
                tile_count = math.ceil(nx / x_chunk) * math.ceil(ny / y_chunk)
                candidates.append((tile_count, -pixels, x_chunk, y_chunk))
    if candidates:
        _tile_count, _negative_pixels, x_chunk, y_chunk = min(candidates)
    else:
        x_chunk, y_chunk = 1, 1
    return time_chunk, x_chunk, y_chunk, n_orientations, 1, 1


def _coarse_power_compressor(blosc_type):
    """Return a speed-first codec for the intermediate coarse-RF cache."""
    codec = os.environ.get("WAVEN_COARSE_ZARR_CODEC", "lz4").strip().lower()
    if codec not in {"lz4", "zstd"}:
        print(f"Unknown WAVEN_COARSE_ZARR_CODEC={codec!r}; using lz4.")
        codec = "lz4"
    if codec == "zstd":
        return blosc_type(cname="zstd", clevel=3, shuffle=blosc_type.BITSHUFFLE), "zstd level 3"
    # Coarse power is an intermediate cache read once by RF correlation.  LZ4
    # greatly reduces encoder backpressure on typical Windows/NVMe systems;
    # users who prioritize smaller caches can explicitly select zstd.
    return blosc_type(cname="lz4", clevel=1, shuffle=blosc_type.BITSHUFFLE), "lz4 level 1"


class _AsyncSliceWriter:
    """One bounded writer thread for disk-backed array slices.

    A single writer preserves Zarr/memmap ordering guarantees while allowing
    the CPU/GPU producer to start the next convolution chunk.  Queue capacity
    bounds retained response buffers and therefore RAM use.
    """

    def __init__(self, telemetry=None, max_queue=2):
        self.telemetry = telemetry
        self.queue = queue.Queue(maxsize=max(1, int(max_queue)))
        self.error = None
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="waven-array-writer", daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                return
            callback, payload, byte_count, on_complete = task
            if self.error is not None:
                # After a capacity/I/O failure, discard already queued payloads
                # instead of continuing writes that could consume the remaining
                # safety reserve before the producer receives the exception.
                self.queue.task_done()
                continue
            try:
                started = time.perf_counter()
                callback(payload)
                if self.telemetry is not None:
                    self.telemetry.add("output", time.perf_counter() - started, byte_count)
                if on_complete is not None:
                    on_complete()
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def submit(self, callback, payload, byte_count, on_complete=None):
        if self._closed:
            raise RuntimeError("Cannot submit to a closed array writer.")
        if self.error is not None:
            raise self.error
        queued_at = time.perf_counter()
        self.queue.put((callback, payload, int(byte_count), on_complete))
        if self.telemetry is not None:
            self.telemetry.add("output_backpressure", time.perf_counter() - queued_at)

    def close(self):
        if self._closed:
            if self.error is not None:
                raise self.error
            return
        self._closed = True
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error


def _safe_work_array(shape, folder_path, name, dtype=np.float32):
    """Allocate a temporary array in RAM only when OS headroom remains."""
    required = _array_bytes(shape, dtype)
    if has_enough_ram(required, safety_margin=1.50):
        return np.empty(shape, dtype=dtype), None
    os.makedirs(folder_path, exist_ok=True)
    free = shutil.disk_usage(folder_path).free
    if free < int(required * 1.10):
        raise OSError(
            f"Insufficient disk space for temporary {name}: need {required * 1.10 / 1024**3:.2f} GiB, "
            f"available {free / 1024**3:.2f} GiB."
        )
    path = os.path.join(folder_path, f".{name}.waven-work.mmap")
    print(f"Using disk-backed temporary {name} ({required / 1024**3:.2f} GiB) to preserve RAM.")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape), path


def _require_disk_space(folder_path, required_bytes, label):
    """Preflight disk capacity before creating a large memory-mapped output."""
    os.makedirs(folder_path, exist_ok=True)
    free = shutil.disk_usage(folder_path).free
    required = int(required_bytes * 1.10)
    if free < required:
        raise OSError(
            f"Insufficient disk space for {label}: need {required / 1024**3:.2f} GiB including headroom, "
            f"available {free / 1024**3:.2f} GiB."
        )


class _CompressedCacheCapacityGuard:
    """Stop a compressed Zarr cache before it can exhaust its volume.

    A Zarr cache is written incrementally and its physical size depends on the
    actual movie/filter values.  Reserving the whole uncompressed shape rejects
    scientifically valid high-parameter runs, while assuming a compression
    ratio can fill a disk.  This guard measures the first completed chunks,
    projects their physical footprint, and preserves the partial cache for a
    safe resume or parameter change.
    """

    def __init__(self, folder_path, total_logical_bytes, label, reserve_bytes=2 * 1024**3):
        self.folder_path = str(folder_path)
        self.total_logical_bytes = max(1, int(total_logical_bytes))
        self.label = str(label)
        self.initial_free = int(shutil.disk_usage(self.folder_path).free)
        self.reserve_bytes = min(
            max(256 * 1024**2, int(reserve_bytes)),
            max(256 * 1024**2, self.initial_free // 5),
        )
        self.logical_written = 0
        self._minimum_sample_bytes = min(self.total_logical_bytes, 64 * 1024**2)

    def check_before_start(self):
        if self.initial_free <= self.reserve_bytes:
            raise OSError(
                f"Insufficient free disk space to safely begin {self.label}: "
                f"available {self.initial_free / 1024**3:.2f} GiB, "
                f"reserve {self.reserve_bytes / 1024**3:.2f} GiB."
            )

    def record_completed_write(self, logical_bytes):
        self.logical_written += max(0, int(logical_bytes))
        free_now = int(shutil.disk_usage(self.folder_path).free)
        physical_written = max(0, self.initial_free - free_now)
        if free_now <= self.reserve_bytes:
            raise OSError(
                f"Stopping {self.label} before the volume is exhausted; only "
                f"{free_now / 1024**3:.2f} GiB remains. The partial Zarr cache is resumable."
            )
        if self.logical_written < self._minimum_sample_bytes or physical_written <= 0:
            return
        projected_bytes = physical_written * self.total_logical_bytes / self.logical_written
        usable_bytes = self.initial_free - self.reserve_bytes
        if projected_bytes > usable_bytes:
            raise OSError(
                f"Compressed {self.label} is projected to require {projected_bytes / 1024**3:.2f} GiB "
                f"on this volume, but only {usable_bytes / 1024**3:.2f} GiB is safely usable. "
                "The partial Zarr cache is resumable; reduce any feature-grid parameter or choose a larger volume."
            )


def _prepare_compressed_cache_capacity(folder_path, logical_bytes, label):
    """Permit compression-backed Zarr output while retaining a measured guard."""
    os.makedirs(folder_path, exist_ok=True)
    guard = _CompressedCacheCapacityGuard(folder_path, logical_bytes, label)
    guard.check_before_start()
    raw_gib = int(logical_bytes) / 1024**3
    free_gib = guard.initial_free / 1024**3
    if logical_bytes * 1.10 > guard.initial_free:
        print(
            f"{label} raw shape is {raw_gib:.2f} GiB but Zarr is compressed; "
            f"starting with {free_gib:.2f} GiB free and monitoring physical usage."
        )
    return guard


def _release_work_array(array, path):
    """Flush and remove a disk-backed temporary workspace when complete."""
    if hasattr(array, "flush"):
        array.flush()
    del array
    if path and os.path.exists(path):
        os.remove(path)


def _maybe_load_into_ram(array, label):
    """Copy disk-backed arrays into RAM when repeated access will be faster."""
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", np.float32)
    if shape is None:
        return array
    required = _array_bytes(shape, dtype)
    if has_enough_ram(required, safety_margin=1.35):
        print(f"Loading {label} into RAM for faster repeated access ({required / (1024**3):.2f} GB).")
        return np.array(array)
    print(f"Keeping {label} disk-backed to avoid memory pressure ({required / (1024**3):.2f} GB).")
    return array


def _open_zarr_array(zarr_module, path, **kwargs):
    """Open a Zarr array across Zarr 2/3 compressor API differences."""
    try:
        return zarr_module.open(path, **kwargs)
    except TypeError:
        compressor = kwargs.pop("compressor", None)
        if compressor is not None:
            kwargs["compressors"] = [compressor]
            try:
                return zarr_module.open(path, **kwargs)
            except Exception:
                kwargs.pop("compressors", None)
        return zarr_module.open(path, **kwargs)


def _convolution_progress_path(save_path):
    """Return the sidecar used to resume an interrupted convolution cache."""
    return f"{save_path}.waven-progress.json"


class _ConvolutionProgress:
    """Persist completed output tiles without treating a partial cache as final.

    The progress sidecar is deliberately separate from the normal artifact
    metadata.  A cache remains unavailable to downstream analysis until the
    caller writes its normal completion metadata, while a cancelled run can
    still reuse every tile recorded here on its next attempt.
    """

    def __init__(self, save_path, shape, kind, flush_every=8):
        self.path = _convolution_progress_path(save_path)
        self.shape = tuple(int(v) for v in shape)
        self.kind = str(kind)
        self.flush_every = max(1, int(flush_every))
        self.completed = set()
        self._pending = 0
        self.reusable = False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if tuple(payload.get("shape", ())) == self.shape and payload.get("kind") == self.kind:
                self.completed = {str(key) for key in payload.get("completed", ())}
                self.reusable = True
        except (OSError, ValueError, TypeError):
            pass

    def mark(self, key):
        key = str(key)
        if key in self.completed:
            return
        self.completed.add(key)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._pending and os.path.exists(self.path):
            return
        payload = {
            "kind": self.kind,
            "shape": self.shape,
            "completed": sorted(self.completed),
            "updated_at": time.time(),
        }
        temporary_path = f"{self.path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary_path, self.path)
        self._pending = 0

    def discard(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def _open_resumable_zarr(
    zarr_module,
    save_path,
    shape,
    chunks,
    compressor,
    progress,
    required_chunk_axes=(),
):
    """Open a matching interrupted Zarr cache, or start a fresh one safely.

    ``required_chunk_axes`` names axes whose chunking is part of the write
    contract.  Time tiles may safely adapt to current VRAM on resume, while the
    Full Model's orientation/sigma/frequency axes must remain aligned with its
    per-combination writer.
    """
    requested_chunks = tuple(int(value) for value in chunks)
    required_chunk_axes = tuple(int(axis) for axis in required_chunk_axes)
    if progress.reusable and os.path.exists(save_path):
        try:
            existing = _open_zarr_array(zarr_module, save_path, mode="a")
            existing_chunks = tuple(int(value) for value in (getattr(existing, "chunks", ()) or ()))
            layout_matches = (
                len(existing_chunks) == len(requested_chunks)
                and all(existing_chunks[axis] == requested_chunks[axis] for axis in required_chunk_axes)
            )
            if tuple(existing.shape) == tuple(shape) and layout_matches:
                print(f"Resuming interrupted convolution cache: {save_path} ({len(progress.completed)} saved tiles)")
                return existing
            print(
                "Restarting incomplete convolution cache because its required Zarr layout changed: "
                f"expected shape/chunks {tuple(shape)}/{requested_chunks}, got "
                f"{tuple(existing.shape)}/{existing_chunks}; required axes={required_chunk_axes}."
            )
        except Exception:
            pass
    elif os.path.exists(save_path):
        print(
            "Restarting incomplete convolution cache because its saved progress does not "
            "match the current cache layout."
        )
    progress.completed.clear()
    progress.reusable = False
    progress._pending = 1
    progress.flush()
    return _open_zarr_array(
        zarr_module, save_path, mode="w", shape=shape, chunks=chunks,
        dtype=np.float32, compressor=compressor,
    )


def _completed_output_interval(progress, group_start, start, end):
    """Return whether persisted tiles cover one requested frame interval."""
    if progress is None:
        return False
    prefix = f"g{int(group_start)}:t"
    intervals = []
    for key in progress.completed:
        if not key.startswith(prefix):
            continue
        try:
            tile_start, tile_end = key[len(prefix):].split("-", 1)
            intervals.append((int(tile_start), int(tile_end)))
        except ValueError:
            # Ignore sidecars written by an older build; their progress is not
            # sufficiently specific to prove coverage of an adaptive chunk.
            continue
    cursor = int(start)
    for tile_start, tile_end in sorted(intervals):
        if tile_end <= cursor:
            continue
        if tile_start > cursor:
            break
        cursor = max(cursor, tile_end)
        if cursor >= int(end):
            return True
    return cursor >= int(end)


def _prepare_video_flat(videodata, device):
    """Move the movie to the compute device once and return pixel-by-time data."""
    if not isinstance(videodata, torch.Tensor):
        video_tensor = torch.as_tensor(videodata, dtype=torch.float32, device=device)
    else:
        video_tensor = videodata.to(device=device, dtype=torch.float32)

    num_frames = video_tensor.shape[0]
    video_flat = video_tensor.reshape(num_frames, -1).t()
    spatial_pixels = video_flat.shape[0]
    return video_tensor, video_flat, num_frames, spatial_pixels


def _select_library_phase(waveletLibrary, phase, frequency_index, spatial_pixels):
    """Return one phase plane from a Gabor library and validate pixel count."""
    if waveletLibrary.ndim == 6:
        if frequency_index is None:
            raise ValueError(
                "This Gabor slice still has an independent frequency axis. "
                "Pass frequency_index explicitly, or use the coupled coarse "
                "library for coarse RF decomposition. Refusing to silently "
                "use frequency index 0."
            )
        lib_phase = waveletLibrary[:, :, :, int(frequency_index), phase, :]
    else:
        lib_phase = waveletLibrary[:, :, :, phase, :]

    if lib_phase.shape[-1] != spatial_pixels:
        raise ValueError(
            "Gabor library pixel axis does not match video frames: "
            f"library has {lib_phase.shape[-1]} pixels, video has {spatial_pixels}. "
            "Check NX/NY and the downsampled movie shape in config.json."
        )
    return lib_phase


@torch.no_grad()
def _project_library_phase(
    video_flat,
    waveletLibrary,
    phase,
    WT_flat,
    s_idx,
    device,
    spatial_pixels,
    filter_chunk_size=None,
    frequency_index=None,
    cancel_event=None,
):
    """Project one phase of a Gabor slice onto a prepared flattened movie."""
    if filter_chunk_size is None:
        filter_chunk_size = wavelet_filter_chunk_size()

    lib_phase = _select_library_phase(
        waveletLibrary,
        phase,
        frequency_index,
        spatial_pixels,
    )
    lib_flat = lib_phase.reshape(-1, spatial_pixels)
    num_filters = lib_flat.shape[0]

    for start in range(0, num_filters, filter_chunk_size):
        check_cancelled(cancel_event)
        end = min(start + filter_chunk_size, num_filters)
        lib_chunk = torch.tensor(
            lib_flat[start:end, :],
            dtype=torch.float32,
            device=device,
        )
        product = torch.matmul(lib_chunk, video_flat)
        WT_flat[start:end, :, s_idx] = product.cpu().numpy()
        del lib_chunk, product


def _legacy_frequency_for_sigma(sigma):
    """Return the coupled coarse-library frequency used by legacy Gabor filters."""
    return (-0.016 * float(sigma)) + 0.148


def _phase_offset_from_index(phase, phase_offsets=None):
    """Return the configured phase offset for a legacy phase index."""
    phase_index = int(phase)
    if phase_offsets is None:
        phase_offsets = (0.0, np.pi / 2)
    phase_offsets = tuple(float(value) for value in phase_offsets)
    if phase_index < 0 or phase_index >= len(phase_offsets):
        raise ValueError(
            f"Phase index {phase_index} is outside the configured phase offsets "
            f"({len(phase_offsets)} phases)."
        )
    return phase_offsets[phase_index]


def _gabor_kernel_for_conv(angle, sigma, phase, frequency=None, coupled_frequency=False):
    """Return a compact 2D kernel matching ``makeGaborFilter`` orientation/layout.

    ``makeGaborFilter`` places ``gabor_kernel(...).real`` on an ``x, y`` canvas,
    then transposes the cropped frame so it aligns with video frames stored as
    ``y, x``.  PyTorch ``conv2d`` receives kernels as ``height, width``;
    therefore the compact convolution kernel is the transposed Gabor real part.
    """
    if coupled_frequency:
        frequency = _legacy_frequency_for_sigma(sigma)
    if frequency is None:
        raise ValueError("frequency is required unless coupled_frequency=True")
    kernel = gabor_kernel(
        frequency=float(frequency),
        theta=float(angle),
        sigma_x=float(sigma),
        sigma_y=float(sigma),
        offset=float(phase),
    ).real
    return np.asarray(kernel.T, dtype=np.float32)


def _conv_frame_chunk_size(
    num_frames,
    nx=None,
    ny=None,
    n_channels=1,
    filter_count=None,
    output_channels=None,
    device=None,
    kernel_shape=None,
    output_buffer_count=1,
):
    """Return a conservative, parameter-aware frame chunk size for convolution."""
    device = device or resolve_compute_device(prefer_gpu=True)
    if nx is not None and ny is not None and filter_count is not None:
        return _resource_aware_frame_chunk_size(
            num_frames,
            nx,
            ny,
            filter_count,
            output_channels if output_channels is not None else filter_count,
            device,
            kernel_shape=kernel_shape,
            output_buffer_count=output_buffer_count,
        )
    if nx is not None and ny is not None:
        return autotuned_frame_chunk_size(num_frames, nx, ny, n_channels=n_channels)
    if device == "cuda":
        return max(1, min(int(num_frames), 256))
    return max(1, min(int(num_frames), 128))


def _conv_filter_group_size(
    frame_chunk_size,
    nx,
    ny,
    n_orientations,
    device,
    dtype_bytes=None,
    filter_channels=None,
    output_channels=None,
    output_buffer_count=2,
):
    """Return a safe number of sigma/frequency combinations per convolution.

    A group expands every live response tensor.  Its limit therefore depends on
    the actual phase/filter bank, output channels, precision, and spatial grid,
    not simply on the number of orientation bins.  ``filter_channels`` and
    ``output_channels`` describe one sigma/frequency combination; callers use
    two filter channels for the fused real/imaginary coarse-power path.
    """
    spatial_pixels = max(1, int(nx) * int(ny))
    filter_channels = max(1, int(filter_channels or n_orientations))
    output_channels = max(1, int(output_channels or n_orientations))
    output_buffer_count = max(1, int(output_buffer_count))
    if dtype_bytes is None:
        dtype_bytes = 2 if str(device).startswith("cuda") and amp_enabled() else 4
    # The input movie is shared by all groups; the response, post-processing,
    # host-transfer staging, and writer payload scale with the group count.
    per_group_bytes = int(frame_chunk_size) * spatial_pixels * int(dtype_bytes) * (
        filter_channels + output_channels * output_buffer_count
    )
    if per_group_bytes <= 0:
        return 1
    if str(device).startswith("cuda"):
        # Use currently free VRAM and reserve room for cuDNN workspace, display
        # memory, and an interactive GUI.
        budget = int(gpu_available_vram_bytes() * 0.35)
        if budget <= 0:
            budget = int(available_ram_bytes() * 0.20)
        return max(1, min(32, budget // per_group_bytes))
    budget = int(available_ram_bytes() * 0.20)
    return max(1, min(8, budget // per_group_bytes))


def _center_pad_kernels(kernels):
    """Pad variable-sized Gabor kernels into one center-aligned bank."""
    max_h = max(kernel.shape[0] for kernel in kernels)
    max_w = max(kernel.shape[1] for kernel in kernels)
    if max_h % 2 == 0:
        max_h += 1
    if max_w % 2 == 0:
        max_w += 1

    bank = np.zeros((len(kernels), max_h, max_w), dtype=np.float32)
    for idx, kernel in enumerate(kernels):
        h, w = kernel.shape
        top = (max_h - h) // 2
        left = (max_w - w) // 2
        bank[idx, top:top + h, left:left + w] = kernel
    return bank


class _ConvolutionRunner(torch.nn.Module):
    """Fixed-kernel convolution module eligible for optional ``torch.compile``."""

    def __init__(self, weights, padding):
        super().__init__()
        self.register_buffer("weights", weights)
        self.padding = padding

    def forward(self, values):
        return F.conv2d(values, self.weights, padding=self.padding)


class _CompileFallbackRunner(torch.nn.Module):
    """Use a compiled runner when possible, then permanently fall back safely.

    ``torch.compile`` is lazy on several PyTorch releases: construction can
    succeed even though the first real CUDA call fails due to an unavailable
    compiler, driver, or unsupported graph. This wrapper retries that first
    failing chunk eagerly and disables compilation for the rest of the job.
    """

    def __init__(self, eager_runner, compiled_runner):
        super().__init__()
        self.eager_runner = eager_runner
        self.compiled_runner = compiled_runner
        self._compiled_active = True

    def forward(self, values):
        if self._compiled_active:
            try:
                return self.compiled_runner(values)
            except Exception as exc:
                self._compiled_active = False
                print(f"torch.compile execution failed; using eager convolution: {exc}")
        return self.eager_runner(values)


def _maybe_compile_convolution(runner):
    """Compile a stable convolution runner only when explicitly requested."""
    if not torch_compile_enabled():
        return runner
    try:
        return _CompileFallbackRunner(runner, torch.compile(runner, mode="reduce-overhead"))
    except Exception as exc:
        print(f"torch.compile setup failed; using eager convolution: {exc}")
        return runner


@torch.no_grad()
def _conv2d_wavelet_bank(
    videodata,
    kernels,
    device,
    frame_chunk_size,
    cancel_event=None,
    telemetry=None,
    postprocess=None,
    output_channels=None,
    execution_plan=None,
):
    """Yield ``conv2d`` responses as ``(start, end, chunk, x, y, theta)`` chunks."""
    kernel_array = _center_pad_kernels(kernels)
    pad_y = int(kernel_array.shape[1] // 2)
    pad_x = int(kernel_array.shape[2] // 2)
    num_frames = int(videodata.shape[0])
    output_channels = int(output_channels or len(kernels))
    if execution_plan is None:
        workload = ConvolutionWorkload(
            frame_count=min(num_frames, max(1, int(frame_chunk_size))),
            spatial_pixels=int(videodata.shape[1]) * int(videodata.shape[2]),
            filter_channels=len(kernels),
            output_channels=output_channels,
            kernel_height=kernel_array.shape[1],
            kernel_width=kernel_array.shape[2],
            activation_dtype_bytes=2 if str(device).startswith("cuda") and amp_enabled() else 4,
            output_buffer_count=2,
        )
        execution_plan = plan_convolution_execution(
            workload,
            allow_multi_gpu=(
                str(device).startswith("cuda") and enabled_feature("MULTI_GPU", default=False)
            ),
        )
    use_frame_parallel = (
        str(device).startswith("cuda")
        and execution_plan.strategy == "frame_parallel"
        and len(execution_plan.devices) > 1
    )
    active_device = (
        execution_plan.devices[0]
        if (
            str(device).startswith("cuda")
            and execution_plan.devices
            and str(execution_plan.devices[0]).startswith("cuda")
        )
        else device
    )

    def build_runner(target_device):
        kernel_tensor = torch.as_tensor(
            kernel_array[:, None, :, :], dtype=torch.float32, device=target_device,
        )
        runner = _maybe_compile_convolution(
            _ConvolutionRunner(kernel_tensor, (pad_y, pad_x))
        )
        return kernel_tensor, runner

    if use_frame_parallel:
        frame_parallel_runners = [
            (target_device, *build_runner(target_device))
            for target_device in execution_plan.devices
        ]
        print(
            "Convolution using workload-aware gather-free frame sharding across "
            f"{len(frame_parallel_runners)} GPUs: {execution_plan.devices}; "
            f"about {execution_plan.frames_per_device} frames/GPU."
        )
    else:
        kernel_tensor, runner = build_runner(active_device)
        frame_parallel_runners = ()

    def read_frames(start, end):
        read_start = time.perf_counter()
        frames = np.asarray(videodata[start:end], dtype=np.float32)
        return frames, time.perf_counter() - read_start

    # The resource planner already accounts for the complete live filter/output
    # working set.  Begin at that planned size so each output matches its Zarr
    # time chunk: smaller automatic batches force read/modify/recompress writes
    # and were a major source of disk backpressure.  CUDA OOM handling below
    # remains free to shrink repeatedly when the desktop is busier than the
    # snapshot used by the planner.
    max_frame_chunk = max(1, int(frame_chunk_size))
    active_frame_chunk = max_frame_chunk
    prefetch = num_frames > active_frame_chunk
    reader_executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
    gpu_executor = (
        ThreadPoolExecutor(max_workers=len(frame_parallel_runners))
        if frame_parallel_runners
        else None
    )
    future = None
    future_end = None

    def convolve_frames_on_device(frame_values, target_device, target_runner):
        """Run one resident frame block and return host output plus elapsed time."""
        frame_tensor = response = output_tensor = None
        compute_start = time.perf_counter()
        try:
            # ``torch.no_grad`` is thread-local. Frame-parallel workers need
            # their own scope rather than relying on the decorator on the
            # caller thread.
            with torch.no_grad():
                frame_tensor = torch.as_tensor(
                    frame_values[:, None, :, :], dtype=torch.float32, device=target_device,
                )
                with convolution_precision_scope(target_device):
                    response = target_runner(frame_tensor)
                    output_tensor = response.permute(0, 3, 2, 1) if postprocess is None else postprocess(response)
            output = _host_convolution_output(output_tensor)
            return output, time.perf_counter() - compute_start
        finally:
            del frame_tensor, response, output_tensor

    def convolve_frames(frame_values):
        """Run one planned batch, optionally sharded without GPU result gather."""
        if not frame_parallel_runners:
            return convolve_frames_on_device(frame_values, active_device, runner)
        frame_slices = [
            block for block in np.array_split(frame_values, len(frame_parallel_runners), axis=0)
            if len(block)
        ]
        futures = [
            gpu_executor.submit(convolve_frames_on_device, block, target_device, target_runner)
            for block, (target_device, _kernel_tensor, target_runner) in zip(
                frame_slices, frame_parallel_runners,
            )
        ]
        outputs_and_seconds = [future.result() for future in futures]
        return (
            _concatenate_convolution_outputs(
                [output for output, _seconds in outputs_and_seconds]
            ),
            max(seconds for _output, seconds in outputs_and_seconds),
        )

    def convolve_with_backoff(frame_values, absolute_start):
        """Yield successful subchunks, repeatedly shrinking after CUDA OOM.

        Keep completed subchunks flowing to the writer instead of retaining a
        list of multi-gigabyte host outputs while a failed batch is retried.
        """
        nonlocal active_frame_chunk
        local_start = 0
        safe_size = len(frame_values)
        while local_start < len(frame_values):
            remaining = len(frame_values) - local_start
            attempt_size = min(safe_size, remaining)
            try:
                output, compute_seconds = convolve_frames(
                    frame_values[local_start:local_start + attempt_size]
                )
            except RuntimeError as exc:
                if (
                    not str(device).startswith("cuda")
                    or not _is_cuda_oom(exc)
                    or attempt_size <= 1
                ):
                    raise
                _release_cuda_working_set()
                safe_size = max(1, attempt_size // 2)
                active_frame_chunk = min(active_frame_chunk, safe_size)
                print(
                    f"CUDA convolution OOM; retrying {attempt_size} frames as "
                    f"{safe_size}-frame blocks."
                )
                continue
            yield (
                absolute_start + local_start,
                absolute_start + local_start + attempt_size,
                output,
                compute_seconds,
            )
            local_start += attempt_size
    try:
        if not str(device).startswith("cuda"):
            configure_torch_cpu_threads()
        start = 0
        while start < num_frames:
            check_cancelled(cancel_event)
            if reader_executor is not None:
                if future is None:
                    future_end = min(start + active_frame_chunk, num_frames)
                    future = reader_executor.submit(read_frames, start, future_end)
                frames, read_seconds = future.result()
                end = int(future_end)
                if end < num_frames:
                    # The current measured size is used for the next read. Any
                    # timing-based adjustment below applies to the following
                    # chunk, so decoding still overlaps current GPU work.
                    future_end = min(end + active_frame_chunk, num_frames)
                    future = reader_executor.submit(read_frames, end, future_end)
                else:
                    future = None
                    future_end = None
            else:
                end = min(start + active_frame_chunk, num_frames)
                frames, read_seconds = read_frames(start, end)
            if telemetry is not None:
                telemetry.add("input", read_seconds, frames.nbytes)
            for output_start, output_end, output, output_seconds in convolve_with_backoff(frames, start):
                if telemetry is not None:
                    telemetry.add(
                        "gpu_compute_and_transfer",
                        output_seconds,
                        _convolution_output_bytes(output),
                    )
                yield output_start, output_end, output
            start = end
    finally:
        if reader_executor is not None:
            reader_executor.shutdown(wait=True)
        if gpu_executor is not None:
            gpu_executor.shutdown(wait=True)
        if frame_parallel_runners:
            del frame_parallel_runners
        else:
            del runner, kernel_tensor
        if str(device).startswith("cuda"):
            _release_cuda_working_set(execution_plan.devices)


@torch.no_grad()
def _conv2d_wavelet_group(videodata, kernels, device, frame_chunk_size, n_orientations, cancel_event=None, telemetry=None):
    """Yield grouped convolution responses as ``(start, end, chunk, n_groups)``."""
    for start, end, response in _conv2d_wavelet_bank(
        videodata,
        kernels,
        device,
        frame_chunk_size,
        cancel_event=cancel_event,
        telemetry=telemetry,
    ):
        group_count = int(response.shape[-1] // int(n_orientations))
        reshaped = response.reshape(
            response.shape[0],
            response.shape[1],
            response.shape[2],
            group_count,
            int(n_orientations),
        )
        yield start, end, np.moveaxis(reshaped, 3, 4)


@torch.no_grad()
def _time_major_convolution_groups(
    videodata, groups, device, frame_chunk_size, cancel_event=None, telemetry=None,
    skip_frame_starts=(),
):
    """Convolve every filter group while one movie chunk is resident.

    Group-major execution rereads the complete movie for every sigma/frequency
    group.  This time-major schedule uploads each frame chunk once, evaluates
    its bounded groups, and yields results tagged with their destination slice.
    It is enabled only when all kernel banks fit comfortably in current RAM/
    VRAM; callers retain the established group-major fallback otherwise.
    """
    prepared = []
    kernel_bytes = 0
    for metadata, kernels, postprocess in groups:
        kernel_array = _center_pad_kernels(kernels)
        kernel_bytes += kernel_array.nbytes
        prepared.append((metadata, kernel_array, postprocess))
    if str(device).startswith("cuda") and kernel_bytes > gpu_available_vram_bytes() * 0.12:
        raise MemoryError("All time-major convolution kernel groups do not fit within the safe VRAM budget.")
    if not str(device).startswith("cuda") and kernel_bytes > available_ram_bytes() * 0.12:
        raise MemoryError("All time-major convolution kernel groups do not fit within the safe RAM budget.")

    runners = []
    for metadata, kernel_array, postprocess in prepared:
        pad_y = int(kernel_array.shape[1] // 2)
        pad_x = int(kernel_array.shape[2] // 2)
        weights = torch.as_tensor(kernel_array[:, None, :, :], dtype=torch.float32, device=device)
        runners.append((metadata, _maybe_compile_convolution(_ConvolutionRunner(weights, (pad_y, pad_x))), postprocess))

    num_frames = int(videodata.shape[0])
    active_frame_chunk = max(1, int(frame_chunk_size))
    skip_frame_starts = {int(start) for start in skip_frame_starts}
    if not str(device).startswith("cuda"):
        configure_torch_cpu_threads()
    try:
        for start in range(0, num_frames, active_frame_chunk):
            check_cancelled(cancel_event)
            if start in skip_frame_starts:
                continue
            end = min(start + active_frame_chunk, num_frames)
            read_start = time.perf_counter()
            frames = np.asarray(videodata[start:end], dtype=np.float32)
            if telemetry is not None:
                telemetry.add("input", time.perf_counter() - read_start, frames.nbytes)
            frame_tensor = None
            try:
                frame_tensor = torch.as_tensor(frames[:, None, :, :], dtype=torch.float32, device=device)
                for metadata, runner, postprocess in runners:
                    # A frame chunk can contain many filter groups.  Keep
                    # cancellation responsive without abandoning the bounded
                    # resident frame buffer or partially executing a group.
                    check_cancelled(cancel_event)
                    response = output_tensor = None
                    try:
                        compute_start = time.perf_counter()
                        with convolution_precision_scope(device):
                            response = runner(frame_tensor)
                            output_tensor = response.permute(0, 3, 2, 1) if postprocess is None else postprocess(response)
                        output = _host_convolution_output(output_tensor)
                        if telemetry is not None:
                            telemetry.add(
                                "gpu_compute_and_transfer",
                                time.perf_counter() - compute_start,
                                _convolution_output_bytes(output),
                            )
                        yield metadata, start, end, output
                    finally:
                        del response, output_tensor
            except RuntimeError as exc:
                if str(device).startswith("cuda") and _is_cuda_oom(exc):
                    _release_cuda_working_set()
                    raise MemoryError(
                        "Time-major convolution lost its VRAM headroom; retrying with group-major scheduling."
                    ) from exc
                raise
            finally:
                del frame_tensor
    finally:
        del runners
        if str(device).startswith("cuda"):
            _release_cuda_working_set()


def convolution_kernel_cache_path(folder_path, kind):
    """Return the compact convolution-kernel cache path for an analysis scale."""
    kind = str(kind).lower()
    if kind not in {"coarse", "fine"}:
        raise ValueError(f"Unknown convolution kernel cache kind: {kind}")
    return os.path.join(folder_path, f"gabor_kernels_{kind}_conv.npz")


def _kernel_cache_matches(
    cache_path, kind, sigmas, frequencies, n_orientations, phase_offsets, coupled_frequencies=None,
):
    """Return whether an existing compact-kernel cache matches requested metadata."""
    if not cache_path or not os.path.exists(cache_path):
        return False
    try:
        with np.load(cache_path) as cache:
            cache_kind = str(cache["kind"].item())
            cache_sigmas = np.asarray(cache["sigmas"], dtype=float)
            cache_frequencies = np.asarray(cache["frequencies"], dtype=float)
            cache_coupled_frequencies = np.asarray(
                cache["coupled_frequencies"] if "coupled_frequencies" in cache.files else [], dtype=float
            )
            cache_phases = np.asarray(cache["phase_offsets"], dtype=float)
            cache_orientations = int(cache["n_orientations"].item())
            cache_kernels = cache["kernels"]
    except Exception as exc:
        print(f"Could not read convolution kernel cache {cache_path}: {exc}")
        return False

    frequencies = np.asarray(frequencies if frequencies is not None else [], dtype=float)
    coupled_frequencies = np.asarray(
        coupled_frequencies if coupled_frequencies is not None else [], dtype=float
    )
    phase_offsets = np.asarray(phase_offsets if phase_offsets is not None else (0.0, np.pi / 2), dtype=float)
    expected_prefix = (len(phase_offsets), len(sigmas), max(1, len(frequencies)))
    if cache_kind != str(kind).lower() or cache_orientations != int(n_orientations):
        return False
    if tuple(cache_kernels.shape[:3]) != expected_prefix:
        return False
    if cache_sigmas.shape != np.asarray(sigmas, dtype=float).shape:
        return False
    if cache_frequencies.shape != frequencies.shape:
        return False
    if cache_coupled_frequencies.shape != coupled_frequencies.shape:
        return False
    if cache_phases.shape != phase_offsets.shape:
        return False
    return (
        np.allclose(cache_sigmas, np.asarray(sigmas, dtype=float))
        and np.allclose(cache_frequencies, frequencies)
        and np.allclose(cache_coupled_frequencies, coupled_frequencies)
        and np.allclose(cache_phases, phase_offsets)
    )


def build_convolution_kernel_cache(
    folder_path,
    kind,
    sigmas,
    n_orientations,
    phase_offsets=None,
    frequencies=None,
    coupled_frequencies=None,
    force=False,
    cancel_event=None,
):
    """Build or reuse the compact Gabor-kernel cache used by convolution wavelets.

    The cache stores center-padded spatial kernels, not flattened image-sized
    filter libraries.  A coarse cache with no frequencies uses the legacy
    sigma-coupled frequency.  ``coupled_frequencies`` supplies one explicit
    frequency per sigma without adding an output axis; ``frequencies`` creates
    an independent coarse frequency axis, just as the full-model cache does.
    """
    kind = str(kind).lower()
    if kind not in {"coarse", "fine"}:
        raise ValueError(f"Unknown convolution kernel cache kind: {kind}")

    os.makedirs(folder_path, exist_ok=True)
    sigmas = np.asarray(sigmas, dtype=float)
    phase_offsets = np.asarray(phase_offsets if phase_offsets is not None else (0.0, np.pi / 2), dtype=float)
    frequencies = np.asarray(frequencies if frequencies is not None else [], dtype=float)
    coupled_frequencies = np.asarray(
        coupled_frequencies if coupled_frequencies is not None else [], dtype=float
    )
    if coupled_frequencies.size:
        if kind != "coarse":
            raise ValueError("Explicit matched frequencies are supported only for the coarse kernel cache.")
        if frequencies.size:
            raise ValueError("Choose either matched sigma/frequency pairs or an independent frequency list.")
        if coupled_frequencies.shape != sigmas.shape:
            raise ValueError("Matched coarse frequencies must contain exactly one value per sigma.")
        if np.any(coupled_frequencies <= 0):
            raise ValueError("Matched coarse frequencies must be greater than zero.")
    frequencies_for_cache = frequencies if frequencies.size else np.asarray([], dtype=float)
    kernel_frequencies = frequencies_for_cache if frequencies_for_cache.size else np.asarray([0.0], dtype=float)

    cache_path = convolution_kernel_cache_path(folder_path, kind)
    if not force and _kernel_cache_matches(
        cache_path,
        kind,
        sigmas,
        frequencies_for_cache,
        n_orientations,
        phase_offsets,
        coupled_frequencies,
    ):
        print(f"Resume: found completed convolution kernel cache, reusing {cache_path}")
        return cache_path

    thetas = np.array([(idx * np.pi) / int(n_orientations) for idx in range(int(n_orientations))])
    kernels = []
    for phase_offset in phase_offsets:
        for sigma_index, sigma in enumerate(sigmas):
            freq_iter = kernel_frequencies
            for frequency in freq_iter:
                check_cancelled(cancel_event)
                if kind == "coarse" and coupled_frequencies.size:
                    orientation_kernels = [
                        _gabor_kernel_for_conv(
                            theta, sigma, phase_offset, frequency=coupled_frequencies[sigma_index]
                        )
                        for theta in thetas
                    ]
                elif kind == "coarse" and not frequencies_for_cache.size:
                    orientation_kernels = [
                        _gabor_kernel_for_conv(theta, sigma, phase_offset, coupled_frequency=True)
                        for theta in thetas
                    ]
                else:
                    orientation_kernels = [
                        _gabor_kernel_for_conv(theta, sigma, phase_offset, frequency=frequency)
                        for theta in thetas
                    ]
                kernels.extend(orientation_kernels)

    padded = _center_pad_kernels(kernels)
    kernel_shape = (
        len(phase_offsets),
        len(sigmas),
        len(kernel_frequencies),
        int(n_orientations),
        padded.shape[-2],
        padded.shape[-1],
    )
    kernel_bank = padded.reshape(kernel_shape)
    np.savez(
        cache_path,
        kind=np.asarray(kind),
        sigmas=sigmas,
        frequencies=frequencies_for_cache,
        coupled_frequencies=coupled_frequencies,
        phase_offsets=phase_offsets,
        n_orientations=np.asarray(int(n_orientations)),
        kernels=kernel_bank,
    )
    print(f"Convolution kernel cache saved to: {cache_path}")
    return cache_path


def _load_convolution_kernel_cache(
    cache_path, kind, sigmas, frequencies, n_orientations, phase_offsets, coupled_frequencies=None,
):
    """Load and validate a compact convolution-kernel cache."""
    if not cache_path:
        return None
    frequencies = np.asarray(frequencies if frequencies is not None else [], dtype=float)
    if not _kernel_cache_matches(
        cache_path, kind, sigmas, frequencies, n_orientations, phase_offsets, coupled_frequencies,
    ):
        raise ValueError(
            "Convolution kernel cache does not match the current configuration. "
            f"Rebuild the Gabor library for the selected backend, or remove this cache: {cache_path}"
        )
    with np.load(cache_path) as cache:
        return np.asarray(cache["kernels"], dtype=np.float32)


def _select_cached_sigma_axis(kernel_cache, cached_sigmas, requested_sigmas):
    """Select requested sigma values from a validated superset kernel cache.

    Fine Gabor assets deliberately store the union of the coarse and full-model
    sigma lists.  The full-model wavelet product, however, has only the
    ``Sigmas Full Model`` axis.  Keep that output contract while allowing the
    compact convolution cache to mirror the legacy fine-library contract.
    """
    cached_sigmas = np.asarray(cached_sigmas, dtype=float)
    requested_sigmas = np.asarray(requested_sigmas, dtype=float)
    indices = []
    for sigma in requested_sigmas:
        matches = np.flatnonzero(np.isclose(cached_sigmas, sigma, rtol=1e-6, atol=1e-9))
        if len(matches) != 1:
            raise ValueError(
                "The fine convolution kernel cache does not contain a unique kernel "
                f"for full-model sigma {float(sigma):g}. Rebuild the Gabor assets."
            )
        indices.append(int(matches[0]))
    return np.asarray(kernel_cache)[:, indices, ...]


def _process_binary_chunk(frames_buffer, xi, xe, yi, ye, shape, actual_len):
    """Background worker for binary downsampling."""
    chunk_arr = np.stack(frames_buffer, axis=0) > 100
    cropped = chunk_arr[:, xi:xe, yi:ye]
    resized = skimage.transform.resize(cropped, (actual_len, shape[0], shape[1]), anti_aliasing=True)
    return (resized >= 0.5).astype(bool)


def downsample_video_binary(
    path,
    visual_coverage,
    analysis_coverage,
    shape=(54, 135),
    chunk_size: Optional[int] = None,
    ratios=(1, 1),
    save_path=None,
    output_format="npy",
    zarr_chunks=None,
    cancel_event=None,
):
    """Downsample a binary stimulus movie to the analysis grid via disk streaming.

    Frames are read in chunks, cropped to the analysis field of view, resized
    with anti-aliasing, and written to a memory-mapped ``_downsampled.npy`` file
    so peak RAM stays bounded regardless of movie length.
    """
    if chunk_size is None:
        chunk_size = video_downsample_chunk_size()
    chunk_size = max(16, int(chunk_size))
    import cv2
    import numpy as np
    import skimage.transform
    import gc

    # OpenCV otherwise creates its own native pool for every outer resize
    # worker.  That nested parallelism can occupy every core and starve Tk's
    # event loop on small Windows machines.  The bounded outer pool below is
    # the only parallel layer used by this GUI-facing operation.
    previous_cv2_threads = None
    try:
        previous_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
    except Exception:
        pass

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        if previous_cv2_threads is not None:
            cv2.setNumThreads(previous_cv2_threads)
        raise IOError(f"Cannot open video file for downsampling: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Grab one frame to dynamically determine cropping bounds
    ret, first_img = cap.read()
    if not ret:
        cap.release()
        if previous_cv2_threads is not None:
            cv2.setNumThreads(previous_cv2_threads)
        raise IOError(f"Cannot read the first frame from video: {path}")
    xi, xe, yi, ye = coverage_crop_bounds(
        first_img.shape[:2], visual_coverage, analysis_coverage,
    )
    print(
        "Stimulus crop (rows, cols): "
        f"{xi}:{xe}, {yi}:{ye}; source={first_img.shape[0]}x{first_img.shape[1]}"
    )
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video

    # ``skimage.transform.resize`` over a large 3-D frame stack was the
    # dominant stage in real recordings: it allocated a float volume for an
    # entire decode chunk and used only one CPU core.  Binary stimuli are much
    # more efficiently downsampled frame-wise with area interpolation.  CUDA
    # uses a bounded batch of the same area-averaging operation; CPU fallback
    # uses OpenCV, which releases the GIL so independent chunks can overlap.
    cropped_height, cropped_width = xe - xi, ye - yi
    bytes_per_gpu_frame = max(1, cropped_height * cropped_width * np.dtype(np.float32).itemsize)
    free_resize_vram = gpu_available_vram_bytes()
    use_gpu_resize = (
        enabled_feature("DOWNSAMPLE_GPU", default=True)
        and torch.cuda.is_available()
        # Reserve room for at least eight frames plus interpolation and transfer
        # buffers.  Falling back before worker startup is safer than a CUDA OOM.
        and free_resize_vram >= bytes_per_gpu_frame * 6 * 8
    )
    resize_prefetch = True
    resize_workers = 1 if use_gpu_resize else cpu_worker_count(cap=4)
    max_pending_chunks = (
        (2 if use_gpu_resize else resize_workers + 1)
        if resize_prefetch
        else 1
    )
    source_frame_bytes = max(1, int(first_img.shape[0]) * int(first_img.shape[1]))
    pipeline_budget = max(64 * 1024**2, int(available_ram_bytes() * 0.12))
    max_chunk_from_pipeline = max(16, pipeline_budget // (source_frame_bytes * max_pending_chunks))
    chunk_size = min(chunk_size, max_chunk_from_pipeline)
    resize_backend = "CUDA area interpolation" if use_gpu_resize else "OpenCV area interpolation"
    gpu_batch_size = 0
    if use_gpu_resize:
        # Input, interpolation workspace, and transfer buffers coexist on the
        # GPU.  Use only a small fraction of free VRAM so the downsampler can
        # run safely alongside the rest of the application.
        gpu_batch_size = max(
            8,
            min(256, free_resize_vram // max(1, bytes_per_gpu_frame * 6)),
        )
    
    output_format = str(output_format or "npy").lower()
    if output_format not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported downsampled video output format: {output_format}")

    # Pre-allocate output directly on disk to save RAM
    if save_path is None:
        save_path = path[:-4] + ('.zarr' if output_format == "zarr" else '_downsampled.npy')
    output_shape = (total_frames, shape[0], shape[1])
    if output_format == "zarr":
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            cap.release()
            if previous_cv2_threads is not None:
                cv2.setNumThreads(previous_cv2_threads)
            raise ImportError(
                "Zarr downsampled video output requires the 'zarr' and 'numcodecs' packages. "
                "Install project requirements or select 'npy'."
            ) from exc
        if zarr_chunks is None:
            zarr_chunks = (min(total_frames, max(1, chunk_size)), shape[0], shape[1])
        zarr_chunks = tuple(
            min(int(dim), int(max(1, chunk)))
            for dim, chunk in zip(output_shape, zarr_chunks)
        )
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        output_mmap = _open_zarr_array(
            zarr,
            save_path,
            mode="w",
            shape=output_shape,
            chunks=zarr_chunks,
            dtype=bool,
            compressor=compressor,
        )
        print(f"Writing downsampled binary video directly to Zarr: {save_path}")
        print(f"Zarr chunks: {zarr_chunks}")
    else:
        output_mmap = np.lib.format.open_memmap(save_path, mode='w+', dtype=bool, shape=output_shape)
    
    frames_buffer = []
    frame_idx = 0
    progress_start = time.time()
    telemetry = OperationTelemetry("Video downsample")

    def resize_binary_chunk(frames):
        resize_start = time.perf_counter()
        actual_len = len(frames)
        resized = np.empty((actual_len, shape[0], shape[1]), dtype=bool)
        source_bytes = 0
        if use_gpu_resize:
            for batch_start in range(0, actual_len, gpu_batch_size):
                check_cancelled(cancel_event)
                batch_end = min(actual_len, batch_start + gpu_batch_size)
                batch = np.stack(frames[batch_start:batch_end], axis=0)
                cropped = np.ascontiguousarray(batch[:, xi:xe, yi:ye] > 100, dtype=np.float32)
                source_bytes += cropped.nbytes
                with torch.inference_mode():
                    tensor = torch.from_numpy(cropped).unsqueeze(1).to("cuda:0", non_blocking=False)
                    reduced = F.interpolate(tensor, size=shape, mode="area")
                    resized[batch_start:batch_end] = reduced.squeeze(1).cpu().numpy() >= 0.5
                del tensor, reduced
        else:
            for frame_index, frame in enumerate(frames):
                check_cancelled(cancel_event)
                cropped = frame[xi:xe, yi:ye]
                source_bytes += cropped.nbytes
                binary = np.ascontiguousarray(cropped > 100, dtype=np.uint8)
                binary *= np.uint8(255)
                interpolation = (
                    cv2.INTER_AREA
                    if cropped.shape[0] >= shape[0] and cropped.shape[1] >= shape[1]
                    else cv2.INTER_LINEAR
                )
                reduced = cv2.resize(binary, (shape[1], shape[0]), interpolation=interpolation)
                resized[frame_index] = reduced >= 128
        return resized, time.perf_counter() - resize_start, source_bytes

    # Bounded in-flight resize chunks overlap serial video decoding with either
    # GPU interpolation or several GIL-free OpenCV workers.  The queue budget
    # above prevents this from recreating the former multi-gigabyte float-stack
    # peak in RAM.
    pending = []
    executor = (
        ThreadPoolExecutor(max_workers=resize_workers)
        if resize_prefetch
        else None
    )

    def write_next():
        nonlocal frame_idx
        pending_item, future = pending.pop(0)
        if future is None:
            frames = pending_item
            actual_len = len(frames)
            resized, resize_seconds, source_bytes = resize_binary_chunk(frames)
        else:
            actual_len = int(pending_item)
            resized, resize_seconds, source_bytes = future.result()
        write_start = time.perf_counter()
        output_mmap[frame_idx:frame_idx + actual_len] = resized
        telemetry.add("resize", resize_seconds, source_bytes)
        telemetry.add("output", time.perf_counter() - write_start, resized.nbytes)
        telemetry.maybe_report()
        frame_idx += actual_len
        print(progress_message("Video downsample", frame_idx, total_frames, progress_start, unit="frames"))

    backend_detail = f"; GPU batch={gpu_batch_size}" if use_gpu_resize else ""
    print(
        f"Downsampling {total_frames} frames directly to disk | backend={resize_backend}, "
        f"workers={resize_workers}, frame chunk={chunk_size}, queue={max_pending_chunks}{backend_detail}",
        end="\n\n",
    )
    try:
        while True:
            check_cancelled(cancel_event)
            decode_start = time.perf_counter()
            ret, img = cap.read()
            telemetry.add("decode", time.perf_counter() - decode_start, img.nbytes if ret else 0)
            if not ret:
                break
            frames_buffer.append(img[:, :, 0])
            if len(frames_buffer) == chunk_size:
                frames = frames_buffer
                frames_buffer = []
                future = executor.submit(resize_binary_chunk, frames) if executor else None
                pending.append((len(frames) if future is not None else frames, future))
                if len(pending) >= max_pending_chunks:
                    write_next()
        check_cancelled(cancel_event)
        if frames_buffer:
            future = executor.submit(resize_binary_chunk, frames_buffer) if executor else None
            pending.append((len(frames_buffer) if future is not None else frames_buffer, future))
        while pending:
            write_next()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if previous_cv2_threads is not None:
            try:
                cv2.setNumThreads(previous_cv2_threads)
            except Exception:
                pass
        
    cap.release()
    if hasattr(output_mmap, "flush"):
        output_mmap.flush()
    del output_mmap
    gc.collect()
    print(f"Success! Saved optimized binary array to: {save_path}")
    telemetry.report()


def _process_uint_chunk(frames_buffer, shape, actual_len):
    """Background worker for uint downsampling."""
    chunk_arr = np.stack(frames_buffer, axis=0)
    resized = skimage.transform.resize(chunk_arr, (actual_len, shape[0], shape[1]), anti_aliasing=True, preserve_range=True)
    return resized.astype(np.uint8)


def downsample_video_uint(path, shape=(54, 135)):
    """
    Auto-scaling UInt Downsampler.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Estimate bytes: total_frames * width * height * 1 byte (uint8) 
    # Multiply by 2 as a buffer for the background threads
    estimated_bytes = total_frames * shape[0] * shape[1] * 2 
    
    # Use our new smart checker!
    high_ram = has_enough_ram(estimated_bytes)
    
    chunk_size = 1000 if high_ram else 300
    max_workers = 4 if high_ram else 2

    futures = []
    frames_buffer = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=total_frames, desc="Downsampling UInt") as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break
                
            frames_buffer.append(img[:, :, 0])
            pbar.update(1)
            
            if len(frames_buffer) == chunk_size:
                futures.append(executor.submit(_process_uint_chunk, list(frames_buffer), shape, chunk_size))
                frames_buffer.clear()

        if len(frames_buffer) > 0:
            futures.append(executor.submit(_process_uint_chunk, list(frames_buffer), shape, len(frames_buffer)))

    cap.release()
    processed_chunks = [future.result() for future in futures]
    
    if processed_chunks:
        final_video = np.concatenate(processed_chunks, axis=0)
        save_path = path[:-4] + '_downsampled.npy'
        np.save(save_path, final_video)
        print(f"Success! Saved optimized uint array to: {save_path}")


@torch.no_grad()
def getWTfromNPY(
    videodata,
    waveletLibrary,
    phase,
    WT_flat,
    s_idx,
    filter_chunk_size: Optional[int] = None,
    frequency_index: Optional[int] = None,
    cancel_event=None,
):
    """Project video frames onto one Gabor scale, writing into ``WT_flat``.

    The filter bank is applied in batches on the best available device (CUDA when
    present, otherwise CPU).  Results stream directly into ``WT_flat`` so the
    full wavelet tensor never has to exist on the GPU at once.
    """
    device = resolve_compute_device(prefer_gpu=True)
    if filter_chunk_size is None:
        filter_chunk_size = wavelet_filter_chunk_size()

    _, video_flat, _, spatial_pixels = _prepare_video_flat(videodata, device)
    _project_library_phase(
        video_flat,
        waveletLibrary,
        phase,
        WT_flat,
        s_idx,
        device,
        spatial_pixels,
        filter_chunk_size=filter_chunk_size,
        frequency_index=frequency_index,
        cancel_event=cancel_event,
    )

    if device == "cuda":
        torch.cuda.empty_cache()


def waveletTransform(frame,phase, L):
    """Function for waveletTransform.

    Args:
        frame: Input value for this operation.
        phase: Input value for this operation.
        L: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    device = resolve_compute_device(prefer_gpu=True)
    output = L[:, :, :, phase].to(device) @ torch.as_tensor(frame.flatten(), device=device, dtype=torch.float32)
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletTransform3D(frame, L):
    """Function for waveletTransform3D.

    Args:
        frame: Input value for this operation.
        L: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    device = resolve_compute_device(prefer_gpu=True)
    output = L.to(device) @ torch.as_tensor(frame.flatten(), device=device, dtype=torch.float32)
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletDecomposition(
    videodata,
    phase,
    sigmas,
    folder_path,
    library_path,
    output_format="npy",
    output_stem=None,
    zarr_chunks=None,
    cancel_event=None,
):
    """Decompose a downsampled movie into Gabor wavelet coefficients.

    Chooses in-memory or memory-mapped output based on ``has_enough_ram``, then
    iterates over sigma scales and fills ``dwt_videodata_{phase}.npy``.
    """
    from ..storage.array_store import load_array

    print(f"Loading Gabor library from {library_path} (mmap_mode='r')...", end="\n\n")
    L = _maybe_load_into_ram(load_array(library_path, mmap_mode='r'), "coarse Gabor library")
    if L.ndim >= 7:
        raise ValueError(
            "waveletDecomposition is the coarse RF path and expects a coupled "
            "coarse library without an independent frequency axis. Build/use "
            "the Coarse Library for coarse RF analysis, and reserve the Fine "
            "Library for waveletDecompositionFull."
        )
    
    prefix_shape = L.shape[:3]  # (lx, ly, thetas)
    num_filters = int(np.prod(prefix_shape))
    T = videodata.shape[0]
    if len(sigmas) > L.shape[3]:
        raise ValueError(
            f"Requested {len(sigmas)} sigma values, but the Gabor library "
            f"contains {L.shape[3]} sigma planes."
        )
    
    # Legacy analysis expects time-first wavelet arrays for coarse decomposition.
    final_shape = (T,) + prefix_shape + (len(sigmas),)
    
    required_bytes = _array_bytes(final_shape, np.float32)
    
    output_format = str(output_format or "npy").lower()
    if output_format not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported coarse wavelet output format: {output_format}")
    output_stem = output_stem or f"dwt_videodata_{phase}"
    save_path = os.path.join(folder_path, f"{output_stem}.{output_format}")
    
    temp_bytes = _array_bytes((num_filters, T, 1), np.float32)
    working_bytes = required_bytes + temp_bytes

    # Zarr is always disk-backed. Its bounded chunks preserve RAM headroom
    # while allowing independent RF/model products to be streamed safely.
    if output_format == "zarr":
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise ImportError(
                "Zarr coarse-wavelet output requires the 'zarr' and 'numcodecs' packages."
            ) from exc
        _require_disk_space(folder_path, required_bytes, f"coarse wavelet phase {phase}")
        if zarr_chunks is None:
            zarr_chunks = (min(T, 128), min(prefix_shape[0], 16), min(prefix_shape[1], 16), prefix_shape[2], len(sigmas))
        zarr_chunks = tuple(
            min(int(dim), max(1, int(chunk)))
            for dim, chunk in zip(final_shape, zarr_chunks)
        )
        wt_final = _open_zarr_array(
            zarr, save_path, mode="w", shape=final_shape, chunks=zarr_chunks,
            dtype=np.float32,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        use_mmap = False
        print(f"Writing coarse phase {phase} directly to Zarr: {save_path}")
    # Prefer RAM for compute and use disk-backed output only when memory is tight.
    elif has_enough_ram(working_bytes, safety_margin=1.20):
        WT_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
        print(f"Using RAM output for coarse phase {phase} ({working_bytes / (1024**3):.2f} GB working set).")
    else:
        _require_disk_space(folder_path, required_bytes, f"coarse wavelet phase {phase}")
        WT_final = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=final_shape)
        use_mmap = True
        print(f"Using disk-backed output for coarse phase {phase}; RAM is below safe working set.")

    temp_flat, temp_flat_path = _safe_work_array(
        (num_filters, T, 1), folder_path, f"coarse_phase_{phase}_projection"
    )
    device = resolve_compute_device(prefer_gpu=True)
    video_tensor, video_flat, _, spatial_pixels = _prepare_video_flat(videodata, device)

    start_time = time.time()
    for s, ss in enumerate(sigmas):
        check_cancelled(cancel_event)
        print(f"Processing sigma {s + 1}/{len(sigmas)}...", end="\n\n")
        temp_flat.fill(0)
        _project_library_phase(
            video_flat,
            L[:, :, :, s],
            phase,
            temp_flat,
            0,
            device,
            spatial_pixels,
            cancel_event=cancel_event,
        )
        spatial = temp_flat[:, :, 0].reshape(prefix_shape + (T,))
        WT_final[..., s] = np.transpose(spatial, (3, 0, 1, 2))
        print(progress_message(f"Coarse phase {phase}", s + 1, len(sigmas), start_time))
        
        gc.collect() 

    del video_flat, video_tensor, spatial
    _release_work_array(temp_flat, temp_flat_path)
    if device == "cuda":
        torch.cuda.empty_cache()
        
    if output_format == "zarr":
        print(f"Success! Saved coarse Zarr array to {save_path}")
    elif use_mmap:
        WT_final.flush()
        del WT_final
        print(f"Success! Saved streamed disk array to {save_path}")
    else:
        print("Saving array to disk...", end="\n\n")
        np.save(save_path, WT_final)
        print(f"Success! Saved RAM array to {save_path}", end="\n\n")


def waveletDecompositionFull(
    videodata,
    phase,
    sigmas,
    frequencies,
    folder_path,
    library_path,
    library_sigmas=None,
    sigma_indices=None,
    output_format="npy",
    zarr_chunks=None,
    cancel_event=None,
):
    """Decompose a movie into full-resolution wavelets for ``run_Full_Model``.

    Writes ``dwt_videodata2_r.npy`` (phase 0) or ``dwt_videodata2_i.npy`` (phase 1)
    with shape ``(T, nx, ny, n_orientations, n_sigmas, n_frequencies)``.
    """
    from ..config import resolve_sigma_indices

    print(
        f"Loading Gabor library from {library_path} (mmap_mode='r') "
        "for full-model decomposition...",
        end="\n\n",
    )
    from ..storage.array_store import load_array

    library = _maybe_load_into_ram(load_array(library_path, mmap_mode="r"), "fine Gabor library")
    lx, ly, num_t = int(library.shape[0]), int(library.shape[1]), int(library.shape[2])
    num_frames = videodata.shape[0]
    sigmas = np.asarray(sigmas, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    ns = len(sigmas)
    if frequencies.size == 0:
        frequencies = np.asarray([0.0], dtype=float)
    nf = len(frequencies)
    if library.ndim < 7 and nf > 1:
        raise ValueError(
            "Full-model wavelet generation was asked for multiple frequencies, "
            "but the Gabor library has no frequency axis. Regenerate the library "
            "from config.json with the configured Frequencies list."
        )

    if sigma_indices is None:
        if library_sigmas is None:
            if library.ndim >= 7:
                raise ValueError(
                    "library_sigmas is required when the Gabor library includes "
                    "a dedicated sigma axis"
                )
            library_sigmas = tuple(range(library.shape[3]))
        sigma_indices = resolve_sigma_indices(library_sigmas, sigmas)
    else:
        sigma_indices = tuple(int(i) for i in sigma_indices)

    phase_suffix = "_r" if phase == 0 else "_i"
    output_format = str(output_format or "npy").lower()
    if output_format not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported full-model wavelet output format: {output_format}")
    save_ext = ".zarr" if output_format == "zarr" else ".npy"
    save_path = os.path.join(folder_path, f"dwt_videodata2{phase_suffix}{save_ext}")
    final_shape = (num_frames, lx, ly, num_t, ns, nf)
    required_bytes = _array_bytes(final_shape, np.float32)
    num_filters = lx * ly * num_t
    temp_bytes = _array_bytes((num_filters, num_frames, 1), np.float32)
    working_bytes = required_bytes + temp_bytes

    if output_format == "zarr":
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise ImportError(
                "Zarr wavelet output requires the 'zarr' and 'numcodecs' packages. "
                "Install project requirements or select 'npy' as the full-model format."
            ) from exc
        os.makedirs(folder_path, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        if zarr_chunks is None:
            zarr_chunks = (min(num_frames, 1800), 1, 1, num_t, ns, nf)
        zarr_chunks = tuple(
            min(int(dim), int(max(1, chunk)))
            for dim, chunk in zip(final_shape, zarr_chunks)
        )
        wt_final = _open_zarr_array(
            zarr,
            save_path,
            mode="w",
            shape=final_shape,
            chunks=zarr_chunks,
            dtype=np.float32,
            compressor=compressor,
        )
        use_mmap = False
        print(f"Writing full-model phase {phase} directly to Zarr: {save_path}")
    elif has_enough_ram(working_bytes, safety_margin=1.20):
        wt_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
        print(f"Using RAM output for full-model phase {phase} ({working_bytes / (1024**3):.2f} GB working set).")
    else:
        _require_disk_space(folder_path, required_bytes, f"full-model wavelet phase {phase}")
        wt_final = np.lib.format.open_memmap(
            save_path,
            mode="w+",
            dtype=np.float32,
            shape=final_shape,
        )
        use_mmap = True
        print(f"Using disk-backed output for full-model phase {phase}; RAM is below safe working set.")

    temp_flat, temp_flat_path = _safe_work_array(
        (num_filters, num_frames, 1), folder_path, f"full_phase_{phase}_projection"
    )
    temp_flat.fill(0)
    device = resolve_compute_device(prefer_gpu=True)
    video_tensor, video_flat, _, spatial_pixels = _prepare_video_flat(videodata, device)

    total_steps = max(1, len(sigma_indices) * nf)
    completed_steps = 0
    start_time = time.time()
    for out_s, lib_s in enumerate(sigma_indices):
        for f_idx in range(nf):
            check_cancelled(cancel_event)
            if library.ndim >= 7:
                lib_slice = library[:, :, :, lib_s, f_idx]
            elif library.ndim == 6:
                lib_slice = library[:, :, :, lib_s]
            else:
                lib_slice = library[:, :, :, lib_s]

            _project_library_phase(
                video_flat,
                lib_slice,
                phase,
                temp_flat,
                0,
                device,
                spatial_pixels,
                cancel_event=cancel_event,
            )
            spatial = temp_flat[:, :, 0].reshape(lx, ly, num_t, num_frames)
            wt_final[:, :, :, :, out_s, f_idx] = np.transpose(
                spatial,
                (3, 0, 1, 2),
            )
            completed_steps += 1
            print(progress_message(f"Full-model phase {phase}", completed_steps, total_steps, start_time))
            gc.collect()

    del video_flat, video_tensor, spatial
    _release_work_array(temp_flat, temp_flat_path)
    if device == "cuda":
        torch.cuda.empty_cache()

    if output_format == "zarr":
        print(f"Success! Saved full-model Zarr array to {save_path}", end="\n\n")
    elif use_mmap:
        wt_final.flush()
        del wt_final
        print(f"Success! Saved streamed full-model array to {save_path}")
    else:
        print("Saving full-model array to disk...", end="\n\n")
        np.save(save_path, wt_final)
        print(f"Success! Saved full-model array to {save_path}", end="\n\n")


def waveletDecompositionConv(
    videodata,
    phase,
    sigmas,
    folder_path,
    n_orientations,
    phase_offsets=None,
    kernel_cache_path=None,
    frame_chunk_size=None,
    filter_group_size=None,
    output_format="npy",
    output_stem=None,
    zarr_chunks=None,
    cancel_event=None,
    coupled_frequencies=None,
):
    """Decompose a movie using compact Gabor kernels and ``torch.conv2d``.

    This is the coarse RF convolution backend. It mirrors the legacy coarse
    library path by using sigma-coupled frequencies and writing a named
    NPY or Zarr array with shape
    ``(time, nx, ny, n_orientations, n_sigmas)``.
    """
    device = resolve_compute_device(prefer_gpu=True)
    num_frames, ny, nx = videodata.shape
    sigmas = np.asarray(sigmas, dtype=float)
    coupled_frequencies = np.asarray(
        coupled_frequencies if coupled_frequencies is not None else [], dtype=float
    )
    if coupled_frequencies.size and coupled_frequencies.shape != sigmas.shape:
        raise ValueError("Matched coarse frequencies must contain exactly one value per sigma.")
    thetas = np.array([(idx * np.pi) / int(n_orientations) for idx in range(int(n_orientations))])
    phase_offset = _phase_offset_from_index(phase, phase_offsets)
    kernel_cache = _load_convolution_kernel_cache(
        kernel_cache_path,
        "coarse",
        sigmas,
        [],
        n_orientations,
        phase_offsets,
        coupled_frequencies,
    ) if kernel_cache_path else None
    final_shape = (int(num_frames), int(nx), int(ny), int(n_orientations), len(sigmas))
    output_format = str(output_format or "npy").lower()
    if output_format not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported coarse wavelet output format: {output_format}")
    output_stem = output_stem or f"dwt_videodata_{phase}"
    save_path = os.path.join(folder_path, f"{output_stem}.{output_format}")
    required_bytes = _array_bytes(final_shape, np.float32)
    frame_chunk_was_auto = frame_chunk_size is None
    if frame_chunk_was_auto:
        frame_chunk_size = _conv_frame_chunk_size(
            num_frames,
            nx,
            ny,
            filter_count=int(n_orientations),
            output_channels=int(n_orientations),
            device=device,
            kernel_shape=tuple(kernel_cache.shape[-2:]) if kernel_cache is not None else None,
            output_buffer_count=2,
        )
    if filter_group_size is None:
        filter_group_size = _conv_filter_group_size(
            frame_chunk_size, nx, ny, n_orientations, device,
            output_buffer_count=2,
        )
    filter_group_size = max(1, min(int(filter_group_size), len(sigmas)))
    if frame_chunk_was_auto and filter_group_size > 1:
        # Include every simultaneous sigma response in the frame plan.  This
        # makes filter-size/sigma expansion as safe as orientation expansion.
        frame_chunk_size = _conv_frame_chunk_size(
            num_frames,
            nx,
            ny,
            filter_count=int(n_orientations) * filter_group_size,
            output_channels=int(n_orientations) * filter_group_size,
            device=device,
            kernel_shape=tuple(kernel_cache.shape[-2:]) if kernel_cache is not None else None,
            output_buffer_count=2,
        )
    progress = None
    capacity_guard = None

    if output_format == "zarr":
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise ImportError(
                "Zarr coarse-wavelet output requires the 'zarr' and 'numcodecs' packages."
            ) from exc
        capacity_guard = _prepare_compressed_cache_capacity(
            folder_path, required_bytes, f"convolution coarse wavelet phase {phase}"
        )
        if zarr_chunks is None:
            zarr_chunks = _coarse_power_zarr_layout(
                final_shape, frame_chunk_size, filter_group_size,
            )
        zarr_chunks = tuple(
            min(int(dim), max(1, int(chunk)))
            for dim, chunk in zip(final_shape, zarr_chunks)
        )
        progress_kind = json.dumps(
            {"product": "coarse_phase", "phase": int(phase), "sigmas": sigmas.tolist(),
             "coupled_frequencies": coupled_frequencies.tolist(),
             "orientations": int(n_orientations),
             "phase_offsets": list(phase_offsets) if phase_offsets is not None else []},
            sort_keys=True,
        )
        progress = _ConvolutionProgress(save_path, final_shape, progress_kind)
        wt_final = _open_resumable_zarr(
            zarr, save_path, final_shape, zarr_chunks,
            Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), progress,
        )
        use_mmap = False
        print(f"Writing convolution coarse phase {phase} directly to Zarr: {save_path}")
    elif has_enough_ram(required_bytes, safety_margin=1.20):
        wt_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
        print(f"Using RAM output for convolution coarse phase {phase} ({required_bytes / (1024**3):.2f} GB).")
    else:
        _require_disk_space(folder_path, required_bytes, f"convolution coarse wavelet phase {phase}")
        wt_final = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float32, shape=final_shape)
        use_mmap = True
        print(f"Using disk-backed output for convolution coarse phase {phase}; RAM is below safe working set.")

    print(f"Convolution backend device: {device}")
    print(f"Convolution backend frame chunk size: {frame_chunk_size}")
    print(f"Convolution backend sigma group size: {filter_group_size}")

    start_time = time.time()
    telemetry = OperationTelemetry(f"Convolution coarse phase {phase}")
    writer = _AsyncSliceWriter(telemetry)
    total_groups = max(1, math.ceil(len(sigmas) / filter_group_size))

    def tile_key(start, end, group_start):
        return f"g{int(group_start)}:t{int(start)}-{int(end)}"

    try:
        completed_groups = 0
        for group_start in range(0, len(sigmas), filter_group_size):
            check_cancelled(cancel_event)
            group_end = min(group_start + filter_group_size, len(sigmas))
            if _completed_output_interval(progress, group_start, 0, num_frames):
                completed_groups += 1
                print(f"Resume: skipping completed coarse phase {phase} sigma group {completed_groups}/{total_groups}.")
                continue
            group_sigmas = sigmas[group_start:group_end]
            if kernel_cache is not None:
                kernels = kernel_cache[int(phase), group_start:group_end, 0].reshape(-1, *kernel_cache.shape[-2:])
            else:
                kernels = [
                    _gabor_kernel_for_conv(
                        theta,
                        sigma,
                        phase_offset,
                        frequency=(coupled_frequencies[group_start + sigma_offset] if coupled_frequencies.size else None),
                        coupled_frequency=not bool(coupled_frequencies.size),
                    )
                    for sigma_offset, sigma in enumerate(group_sigmas)
                    for theta in thetas
                ]
            print(
                f"Convolution coarse phase {phase}: sigma group "
                f"{completed_groups + 1}/{total_groups} ({group_start + 1}-{group_end} of {len(sigmas)})"
            )
            chunk_start = time.time()
            for chunk_index, (start, end, response) in enumerate(_conv2d_wavelet_group(
                videodata, kernels, device, frame_chunk_size, n_orientations,
                cancel_event=cancel_event, telemetry=telemetry,
            ), start=1):
                def write_response(values, t0=start, t1=end, s0=group_start, s1=group_end):
                    wt_final[t0:t1, :, :, :, s0:s1] = values
                def mark_completed(key=tile_key(start, end, group_start), byte_count=response.nbytes):
                    if progress is not None:
                        progress.mark(key)
                    if capacity_guard is not None:
                        capacity_guard.record_completed_write(byte_count)
                if writer is not None:
                    writer.submit(
                        write_response, response, response.nbytes,
                        on_complete=mark_completed,
                    )
                else:
                    write_start = time.perf_counter()
                    write_response(response)
                    telemetry.add("output", time.perf_counter() - write_start, response.nbytes)
                    mark_completed()
                print(progress_message(
                    f"Convolution coarse phase {phase}", end, num_frames,
                    chunk_start, unit="frames",
                ))
                telemetry.maybe_report()
            completed_groups += 1
            print(progress_message(f"Convolution coarse phase {phase}", completed_groups, total_groups, start_time, unit="groups"))
            gc.collect()
    finally:
        writer_error = None
        try:
            if writer is not None:
                writer.close()
        except Exception as exc:
            # A disk-capacity guard can fail inside the asynchronous writer.
            # Flush completed-tile metadata before surfacing it so the cache is
            # truly resumable rather than appearing corrupt on the next run.
            writer_error = exc
        finally:
            if progress is not None:
                progress.flush()
            if device == "cuda":
                _release_cuda_working_set()
        if writer_error is not None:
            raise writer_error
    if output_format == "zarr":
        progress.discard()
        print(f"Success! Saved convolution coarse Zarr array to {save_path}")
    elif use_mmap:
        wt_final.flush()
        del wt_final
        print(f"Success! Saved streamed convolution coarse array to {save_path}")
    else:
        print("Saving convolution coarse array to disk...", end="\n\n")
        np.save(save_path, wt_final)
        print(f"Success! Saved convolution coarse array to {save_path}", end="\n\n")
    telemetry.report()


def waveletPowerDecompositionConv(
    videodata, sigmas, folder_path, n_orientations, phase_offsets=None,
    kernel_cache_path=None, frame_chunk_size=None, filter_group_size=None,
    output_stem="coarse_rf_power", zarr_chunks=None, cancel_event=None,
    progress_signature=None, rf_neuron_count=None, frequencies=None,
    coupled_frequencies=None, phase_output_stems=None,
    phase_coupled_frequencies=None,
    write_power=True,
):
    """Write coarse wavelet power, optionally retaining its phase pair.

    Coarse RF needs only ``real**2 + imaginary**2``. Real and imaginary
    kernels are fused into one convolution bank, so each movie chunk is read
    and transferred to the GPU once before the power product is written.  When
    ``phase_output_stems`` supplies ``(real_stem, imaginary_stem)``, the same
    convolution also persists the quadrature responses needed by ``run_Model``.
    With an independent power frequency axis, ``phase_coupled_frequencies``
    selects one frequency per sigma for the compact phase pair; every selected
    value must be present in the power frequency list.  Set ``write_power`` to
    ``False`` with ``phase_output_stems`` to write only the Run Model real and
    imaginary pair in one convolution pass.
    """
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ImportError("Direct coarse RF power requires zarr and numcodecs.") from exc
    device = resolve_compute_device(prefer_gpu=True)
    codec_threads = configure_zarr_codec_threads()
    num_frames, ny, nx = videodata.shape
    sigmas = np.asarray(sigmas, dtype=float)
    frequencies = np.asarray(frequencies if frequencies is not None else [], dtype=float)
    coupled_frequencies = np.asarray(
        coupled_frequencies if coupled_frequencies is not None else [], dtype=float
    )
    phase_coupled_frequencies = np.asarray(
        phase_coupled_frequencies if phase_coupled_frequencies is not None else [],
        dtype=float,
    )
    independent_frequencies = bool(frequencies.size)
    phase_output_stems = tuple(phase_output_stems or ())
    write_power = bool(write_power)
    if phase_output_stems and len(phase_output_stems) != 2:
        raise ValueError("phase_output_stems must contain real and imaginary output stems.")
    if not write_power and not phase_output_stems:
        raise ValueError("Phase-only convolution requires real and imaginary output stems.")
    phase_frequency_indices = None
    if phase_output_stems and independent_frequencies:
        if phase_coupled_frequencies.shape != sigmas.shape or np.any(phase_coupled_frequencies <= 0):
            raise ValueError(
                "Independent coarse power needs one positive sigma-coupled frequency per "
                "Run Model phase output."
            )
        phase_frequency_indices = []
        for sigma, phase_frequency in zip(sigmas, phase_coupled_frequencies):
            matches = np.flatnonzero(
                np.isclose(frequencies, phase_frequency, rtol=1e-6, atol=1e-9)
            )
            if len(matches) != 1:
                raise ValueError(
                    "A Run Model sigma-coupled phase frequency is absent or ambiguous in "
                    f"the Coarse RF independent list (sigma={sigma}, frequency={phase_frequency})."
                )
            phase_frequency_indices.append(int(matches[0]))
    if coupled_frequencies.size:
        if independent_frequencies:
            raise ValueError("Choose either matched sigma/frequency pairs or an independent frequency list.")
        if coupled_frequencies.shape != sigmas.shape:
            raise ValueError("Matched coarse frequencies must contain exactly one value per sigma.")
        if np.any(coupled_frequencies <= 0):
            raise ValueError("Matched coarse frequencies must be greater than zero.")
    kernel_frequencies = frequencies if independent_frequencies else np.asarray([0.0], dtype=float)
    n_frequencies = len(kernel_frequencies)
    thetas = np.array([(idx * np.pi) / int(n_orientations) for idx in range(int(n_orientations))])
    kernel_cache = _load_convolution_kernel_cache(
        kernel_cache_path, "coarse", sigmas, frequencies, n_orientations, phase_offsets,
        coupled_frequencies,
    ) if kernel_cache_path else None
    final_shape = (int(num_frames), int(nx), int(ny), int(n_orientations), len(sigmas))
    if independent_frequencies:
        final_shape += (n_frequencies,)
    phase_final_shape = final_shape[:-1] if independent_frequencies else final_shape
    logical_cache_bytes = (_array_bytes(final_shape, np.float32) if write_power else 0) + (
        2 * _array_bytes(phase_final_shape, np.float32) if phase_output_stems else 0
    )
    # A Zarr cache is compressed on disk.  Monitor its measured footprint rather
    # than reserving an additional uncompressed 400+ GiB array up front.
    capacity_guard = _prepare_compressed_cache_capacity(
        folder_path, logical_cache_bytes,
        "coarse RF power and phase caches"
        if write_power and phase_output_stems
        else "Run Model phase caches"
        if phase_output_stems
        else "coarse RF power cache",
    )
    kernel_shape = tuple(kernel_cache.shape[-2:]) if kernel_cache is not None else None
    output_channel_multiplier = 3 if write_power and phase_output_stems else 2 if phase_output_stems else 1
    frame_chunk_was_auto = frame_chunk_size is None
    if frame_chunk_was_auto:
        frame_chunk_size = _conv_frame_chunk_size(
            num_frames,
            nx,
            ny,
            n_channels=2,
            filter_count=2 * int(n_orientations) * n_frequencies,
            output_channels=output_channel_multiplier * int(n_orientations) * n_frequencies,
            device=device,
            kernel_shape=kernel_shape,
            output_buffer_count=2,
        )
    if filter_group_size is None:
        # The group planner accounts for both real/imaginary response channels
        # and the fused power/transfer buffers.
        filter_group_size = _conv_filter_group_size(
            frame_chunk_size,
            nx,
            ny,
            n_orientations,
            device,
            filter_channels=2 * int(n_orientations) * n_frequencies,
            output_channels=output_channel_multiplier * int(n_orientations) * n_frequencies,
            output_buffer_count=2,
        )
    filter_group_size = max(1, min(int(filter_group_size), len(sigmas)))
    if frame_chunk_was_auto and filter_group_size > 1:
        # Re-evaluate once with all requested sigma groups represented in the
        # response and power buffers.  This covers sigma/frequency-like bank
        # expansion as well as a high orientation count.
        frame_chunk_size = _conv_frame_chunk_size(
            num_frames,
            nx,
            ny,
            n_channels=2,
            filter_count=2 * int(n_orientations) * filter_group_size * n_frequencies,
            output_channels=output_channel_multiplier * int(n_orientations) * filter_group_size * n_frequencies,
            device=device,
            kernel_shape=kernel_shape,
            output_buffer_count=2,
        )
    primary_shape = final_shape if write_power else phase_final_shape
    if zarr_chunks is None:
        zarr_chunks = _coarse_power_zarr_layout(
            primary_shape,
            frame_chunk_size,
            filter_group_size,
            rf_neuron_count=rf_neuron_count,
        )
    zarr_chunks = tuple(min(int(dim), max(1, int(chunk))) for dim, chunk in zip(primary_shape, zarr_chunks))
    output_stem = output_stem or ("coarse_rf_power" if write_power else None)
    save_path = os.path.join(folder_path, f"{output_stem}.zarr") if output_stem else None
    # The regular artifact metadata is intentionally written only after every
    # tile completes.  Include the caller's input fingerprint in the separate
    # progress sidecar so a cancelled cache can safely resume, but never be
    # reused after its stimulus/crop parameters changed.
    phase_paths = tuple(
        os.path.join(folder_path, f"{stem}.zarr") for stem in phase_output_stems
    )
    primary_path = save_path if write_power else phase_paths[0]
    phase_zarr_chunks = zarr_chunks[:-1] if independent_frequencies and write_power else zarr_chunks
    progress_kind = json.dumps(
        {"product": "coarse_rf_power" if write_power else "coarse_phase_pair_v2", "sigmas": sigmas.tolist(),
           "orientations": int(n_orientations), "frequencies": frequencies.tolist(),
           "coupled_frequencies": coupled_frequencies.tolist(),
           "phase_coupled_frequencies": phase_coupled_frequencies.tolist(),
           "phase_offsets": list(phase_offsets) if phase_offsets is not None else [],
           "phase_outputs": list(phase_output_stems),
           "resume_signature": progress_signature},
        sort_keys=True,
    )
    progress = _ConvolutionProgress(primary_path, primary_shape, progress_kind)
    compressor, compressor_description = _coarse_power_compressor(Blosc)
    if phase_paths and progress.reusable:
        try:
            def phase_matches(path):
                phase_array = _open_zarr_array(zarr, path, mode="r")
                return tuple(phase_array.shape) == tuple(phase_final_shape)

            phase_resume_ready = all(phase_matches(path) for path in phase_paths)
        except Exception:
            phase_resume_ready = False
        if not phase_resume_ready:
            # The power progress marker is shared with the phase pair.  It is
            # only valid when all outputs have the same completed tiles.
            print("Restarting incomplete coarse cache bundle because a phase output is missing or incompatible.")
            progress.completed.clear()
            progress.reusable = False
            progress._pending = 1
            progress.flush()
    primary_output = _open_resumable_zarr(
        zarr, primary_path, primary_shape, zarr_chunks,
        compressor, progress,
    )
    power = primary_output if write_power else None
    phase_outputs = ()
    if phase_paths:
        phase_mode = "a" if progress.reusable else "w"
        phase_open_kwargs = {"mode": phase_mode}
        if phase_mode == "w":
            phase_open_kwargs.update(
                shape=phase_final_shape,
                chunks=phase_zarr_chunks,
                dtype=np.float32,
                compressor=compressor,
            )
        if write_power:
            phase_outputs = tuple(
                _open_zarr_array(zarr, path, **phase_open_kwargs)
                for path in phase_paths
            )
        else:
            phase_outputs = (
                primary_output,
                _open_zarr_array(zarr, phase_paths[1], **phase_open_kwargs),
            )
    # An interrupted run can be restarted when free VRAM differs. Keep an old
    # *smaller* time tile to retain write alignment, but never restore an old
    # larger tile that the current parameter-aware planner has declared unsafe.
    # The latter would recreate the same OOM on every resume.
    persisted_chunks = tuple(int(value) for value in getattr(primary_output, "chunks", ()) or ())
    if progress.reusable and len(persisted_chunks) == len(primary_shape):
        persisted_time_chunk = max(1, persisted_chunks[0])
        if persisted_time_chunk < int(frame_chunk_size):
            print(
                "Resume: retaining interrupted cache time chunk "
                f"{persisted_time_chunk} instead of newly tuned {frame_chunk_size}."
            )
            frame_chunk_size = persisted_time_chunk
        elif persisted_time_chunk > int(frame_chunk_size):
            print(
                "Resume: using newly safe time chunk "
                f"{frame_chunk_size} instead of interrupted {persisted_time_chunk}; "
                "some existing Zarr chunks will be updated in smaller slices."
            )
        # Keep a smaller-or-equal old sigma tile for resumability.  Never grow
        # a resumed group above the current safe VRAM-derived estimate.
        persisted_sigma_chunk = max(1, persisted_chunks[-2 if independent_frequencies and write_power else -1])
        if persisted_sigma_chunk <= int(filter_group_size):
            filter_group_size = persisted_sigma_chunk
        zarr_chunks = persisted_chunks
    frequency_mode = (
        "frequency list"
        if independent_frequencies
        else "matched sigma/frequency pairs"
        if coupled_frequencies.size
        else "legacy sigma-coupled"
    )
    print(
        f"Direct {'coarse RF power' if write_power else 'Run Model phase pair'}"
        f"{' + phase pair' if write_power and phase_outputs else ''} | "
        f"device={device}, frame chunk={frame_chunk_size}, "
        f"sigma group={filter_group_size}, frequency mode={frequency_mode}"
    )
    if phase_outputs and independent_frequencies:
        print(
            "Run Model phase cache uses the sigma-coupled subset of the "
            "independent Coarse RF frequency list."
        )
    print(
        f"Direct {'coarse RF power' if write_power else 'Run Model phase'} cache layout | chunks={zarr_chunks}; "
        f"compressor={compressor_description}"
        + (f"; Blosc threads={codec_threads}" if codec_threads is not None else "")
    )
    start_time = time.time()
    telemetry = OperationTelemetry("Direct coarse RF power")
    writer = _AsyncSliceWriter(telemetry)
    total_groups = max(1, math.ceil(len(sigmas) / filter_group_size))
    group_records = []
    for group_number, group_start in enumerate(range(0, len(sigmas), filter_group_size), start=1):
        group_end = min(group_start + filter_group_size, len(sigmas))
        if kernel_cache is not None:
            real_kernels = kernel_cache[0, group_start:group_end].reshape(-1, *kernel_cache.shape[-2:])
            imag_kernels = kernel_cache[1, group_start:group_end].reshape(-1, *kernel_cache.shape[-2:])
        else:
            real_phase = _phase_offset_from_index(0, phase_offsets)
            imag_phase = _phase_offset_from_index(1, phase_offsets)
            real_kernels = [
                _gabor_kernel_for_conv(
                    theta,
                    sigma,
                    real_phase,
                    frequency=(coupled_frequencies[group_start + sigma_offset] if coupled_frequencies.size else frequency),
                    coupled_frequency=not independent_frequencies and not bool(coupled_frequencies.size),
                )
                for sigma_offset, sigma in enumerate(sigmas[group_start:group_end])
                for frequency in kernel_frequencies for theta in thetas
            ]
            imag_kernels = [
                _gabor_kernel_for_conv(
                    theta,
                    sigma,
                    imag_phase,
                    frequency=(coupled_frequencies[group_start + sigma_offset] if coupled_frequencies.size else frequency),
                    coupled_frequency=not independent_frequencies and not bool(coupled_frequencies.size),
                )
                for sigma_offset, sigma in enumerate(sigmas[group_start:group_end])
                for frequency in kernel_frequencies for theta in thetas
            ]

        def fused_power_response(
            values,
            group_size=group_end - group_start,
            phase_indices=(
                tuple(phase_frequency_indices[group_start:group_end])
                if phase_frequency_indices is not None else ()
            ),
        ):
            """Prepare only the requested coarse products before PCIe transfer."""
            shaped = values.reshape(
                values.shape[0], 2, group_size, n_frequencies, int(n_orientations), values.shape[2], values.shape[3]
            )
            if independent_frequencies:
                if write_power:
                    power_response = (
                        shaped[:, 0].square() + shaped[:, 1].square()
                    ).permute(0, 5, 4, 3, 1, 2).contiguous()
                    if not phase_outputs:
                        return power_response
                if not phase_outputs:
                    raise RuntimeError("Phase-only convolution requires real and imaginary output stores.")
                real_response = torch.stack(
                    [shaped[:, 0, sigma_index, frequency_index] for sigma_index, frequency_index in enumerate(phase_indices)],
                    dim=1,
                ).permute(0, 4, 3, 2, 1).contiguous()
                imag_response = torch.stack(
                    [shaped[:, 1, sigma_index, frequency_index] for sigma_index, frequency_index in enumerate(phase_indices)],
                    dim=1,
                ).permute(0, 4, 3, 2, 1).contiguous()
                return (power_response, real_response, imag_response) if write_power else (real_response, imag_response)
            if write_power:
                power_response = (
                    shaped[:, 0].square() + shaped[:, 1].square()
                )[:, :, 0].permute(0, 4, 3, 2, 1).contiguous()
                if not phase_outputs:
                    return power_response
            real_response = shaped[:, 0, :, 0].permute(0, 4, 3, 2, 1).contiguous()
            imag_response = shaped[:, 1, :, 0].permute(0, 4, 3, 2, 1).contiguous()
            return (power_response, real_response, imag_response) if write_power else (real_response, imag_response)

        group_records.append(
            ((group_number, group_start, group_end), list(real_kernels) + list(imag_kernels), fused_power_response)
        )

    largest_group = max(1, max(metadata[2] - metadata[1] for metadata, _kernels, _postprocess in group_records))
    coarse_execution_plan = plan_convolution_execution(
        ConvolutionWorkload(
            frame_count=frame_chunk_size,
            spatial_pixels=int(nx) * int(ny),
            filter_channels=2 * int(n_orientations) * largest_group * n_frequencies,
            output_channels=output_channel_multiplier * int(n_orientations) * largest_group * n_frequencies,
            kernel_height=kernel_shape[0] if kernel_shape is not None else 1,
            kernel_width=kernel_shape[1] if kernel_shape is not None else 1,
            activation_dtype_bytes=2 if str(device).startswith("cuda") and amp_enabled() else 4,
            output_buffer_count=2,
        ),
        allow_multi_gpu=(
            str(device).startswith("cuda") and enabled_feature("MULTI_GPU", default=False)
        ),
    )
    print(
        "Direct coarse RF execution policy | "
        f"{coarse_execution_plan.strategy}; {coarse_execution_plan.reason}."
    )

    def tile_key(start, end, group_start):
        return f"g{int(group_start)}:t{int(start)}-{int(end)}"

    def store_power(start, end, group_start, group_end, payload):
        key = tile_key(start, end, group_start)
        payload_bytes = _convolution_output_bytes(payload)
        def write_power(values, t0=start, t1=end, s0=group_start, s1=group_end):
            if phase_outputs:
                if write_power:
                    values, real_values, imag_values = values
                else:
                    real_values, imag_values = values
                phase_outputs[0][t0:t1, :, :, :, s0:s1] = real_values
                phase_outputs[1][t0:t1, :, :, :, s0:s1] = imag_values
            if write_power:
                if independent_frequencies:
                    power[t0:t1, :, :, :, s0:s1, :] = values
                else:
                    power[t0:t1, :, :, :, s0:s1] = values
        def mark_completed(tile_key_value=key, byte_count=payload_bytes):
            progress.mark(tile_key_value)
            capacity_guard.record_completed_write(byte_count)
        if writer is not None:
            writer.submit(write_power, payload, payload_bytes, on_complete=mark_completed)
        else:
            write_start = time.perf_counter()
            write_power(payload)
            telemetry.add("output", time.perf_counter() - write_start, payload_bytes)
            mark_completed()

    try:
        use_time_major = len(group_records) > 1 and coarse_execution_plan.strategy == "single"
        if use_time_major:
            try:
                print("Direct coarse RF power using time-major filter scheduling (one movie read per frame chunk).")
                frame_starts = range(0, num_frames, frame_chunk_size)
                completed_frame_starts = {
                    start for start in frame_starts
                    if all(_completed_output_interval(progress, group_start, start, min(start + frame_chunk_size, num_frames))
                           for _group_number, group_start, _group_end in (record[0] for record in group_records))
                }
                if completed_frame_starts:
                    print(f"Resume: skipping {len(completed_frame_starts)} completed time-major frame chunk(s).")
                for metadata, start, end, power_response in _time_major_convolution_groups(
                    videodata, group_records, device, frame_chunk_size, cancel_event=cancel_event, telemetry=telemetry,
                    skip_frame_starts=completed_frame_starts,
                ):
                    group_number, group_start, group_end = metadata
                    store_power(start, end, group_start, group_end, power_response)
                    if end == num_frames:
                        print(progress_message(
                            "Direct coarse RF power", group_number, total_groups, start_time, unit="groups",
                        ))
                    telemetry.maybe_report()
            except MemoryError as exc:
                print(f"Time-major scheduling exceeded its safe kernel budget; falling back to group-major: {exc}")
                use_time_major = False
        if not use_time_major:
            for metadata, fused_kernels, fused_power_response in group_records:
                group_number, group_start, group_end = metadata
                group_complete = _completed_output_interval(progress, group_start, 0, num_frames)
                if group_complete:
                    print(f"Resume: skipping completed sigma group {group_number}/{total_groups}.")
                    continue
                for chunk_index, (start, end, power_response) in enumerate(_conv2d_wavelet_bank(
                    videodata, fused_kernels, device, frame_chunk_size,
                    cancel_event=cancel_event,
                    telemetry=telemetry,
                    postprocess=fused_power_response,
                    output_channels=output_channel_multiplier * int(n_orientations) * (group_end - group_start) * n_frequencies,
                    execution_plan=coarse_execution_plan,
                ), start=1):
                    store_power(start, end, group_start, group_end, power_response)
                    if chunk_index % 8 == 0 or end == num_frames:
                        print(progress_message(
                            "Direct coarse RF power", end, num_frames, start_time, unit="frames",
                        ))
                    telemetry.maybe_report()
                print(progress_message("Direct coarse RF power", group_number, total_groups, start_time, unit="groups"))
    finally:
        writer_error = None
        try:
            if writer is not None:
                writer.close()
        except Exception as exc:
            writer_error = exc
        finally:
            progress.flush()
            if device == "cuda":
                _release_cuda_working_set()
        if writer_error is not None:
            raise writer_error
    del power, phase_outputs
    progress.discard()
    if write_power:
        print(f"Success! Saved direct coarse RF power Zarr array to {save_path}")
    else:
        print(f"Success! Saved direct Run Model phase Zarr pair to {phase_paths[0]} and {phase_paths[1]}")
    telemetry.report()
    return save_path if write_power else phase_paths[0]


def waveletDecompositionFullConv(
    videodata,
    phase,
    sigmas,
    frequencies,
    folder_path,
    n_orientations,
    phase_offsets=None,
    output_format="npy",
    zarr_chunks=None,
    kernel_cache_path=None,
    frame_chunk_size=None,
    filter_group_size=None,
    cancel_event=None,
    library_sigmas=None,
    progress_signature=None,
):
    """Decompose full-model wavelets with compact Gabor kernels and ``conv2d``.

    Writes ``dwt_videodata2_r`` or ``dwt_videodata2_i`` with the same shape and
    output format as :func:`waveletDecompositionFull`:
    ``(time, nx, ny, n_orientations, n_sigmas, n_frequencies)``.
    """
    device = resolve_compute_device(prefer_gpu=True)
    num_frames, ny, nx = videodata.shape
    sigmas = np.asarray(sigmas, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.size == 0:
        frequencies = np.asarray([0.0], dtype=float)
    thetas = np.array([(idx * np.pi) / int(n_orientations) for idx in range(int(n_orientations))])
    phase_offset = _phase_offset_from_index(phase, phase_offsets)
    cache_sigmas = np.asarray(
        sigmas if library_sigmas is None else library_sigmas,
        dtype=float,
    )
    kernel_cache = _load_convolution_kernel_cache(
        kernel_cache_path,
        "fine",
        cache_sigmas,
        frequencies,
        n_orientations,
        phase_offsets,
    ) if kernel_cache_path else None
    if kernel_cache is not None and library_sigmas is not None:
        kernel_cache = _select_cached_sigma_axis(kernel_cache, cache_sigmas, sigmas)
    final_shape = (
        int(num_frames),
        int(nx),
        int(ny),
        int(n_orientations),
        len(sigmas),
        len(frequencies),
    )

    phase_suffix = "_r" if phase == 0 else "_i"
    output_format = str(output_format or "npy").lower()
    if output_format not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported full-model wavelet output format: {output_format}")
    save_ext = ".zarr" if output_format == "zarr" else ".npy"
    save_path = os.path.join(folder_path, f"dwt_videodata2{phase_suffix}{save_ext}")
    required_bytes = _array_bytes(final_shape, np.float32)
    combo_count = len(sigmas) * len(frequencies)
    frame_chunk_was_auto = frame_chunk_size is None
    if frame_chunk_was_auto:
        frame_chunk_size = _conv_frame_chunk_size(
            num_frames,
            nx,
            ny,
            filter_count=int(n_orientations),
            output_channels=int(n_orientations),
            device=device,
            kernel_shape=tuple(kernel_cache.shape[-2:]) if kernel_cache is not None else None,
            output_buffer_count=2,
        )
    if filter_group_size is None:
        filter_group_size = _conv_filter_group_size(
            frame_chunk_size, nx, ny, n_orientations, device,
            output_buffer_count=2,
        )
    filter_group_size = max(1, min(int(filter_group_size), combo_count))
    if frame_chunk_was_auto and filter_group_size > 1:
        # A full-model group is a Cartesian sigma/frequency group.  Its response
        # bank scales with both parameters, so choose the frame chunk from all
        # simultaneous combinations rather than the first orientation alone.
        frame_chunk_size = _conv_frame_chunk_size(
            num_frames,
            nx,
            ny,
            filter_count=int(n_orientations) * filter_group_size,
            output_channels=int(n_orientations) * filter_group_size,
            device=device,
            kernel_shape=tuple(kernel_cache.shape[-2:]) if kernel_cache is not None else None,
            output_buffer_count=2,
        )
    progress = None
    capacity_guard = None

    if output_format == "zarr":
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise ImportError(
                "Zarr wavelet output requires the 'zarr' and 'numcodecs' packages. "
                "Install project requirements or select 'npy' as the full-model format."
            ) from exc
        os.makedirs(folder_path, exist_ok=True)
        codec_threads = configure_zarr_codec_threads()
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        capacity_guard = _prepare_compressed_cache_capacity(
            folder_path, required_bytes, f"convolution full-model phase {phase}"
        )
        if zarr_chunks is None:
            zarr_chunks = _full_convolution_zarr_layout(
                final_shape, frame_chunk_size, filter_group_size,
            )
        zarr_chunks = tuple(
            min(int(dim), int(max(1, chunk)))
            for dim, chunk in zip(final_shape, zarr_chunks)
        )
        progress_kind = json.dumps(
            {"product": "full_phase_v3", "phase": int(phase), "sigmas": sigmas.tolist(),
             "frequencies": frequencies.tolist(), "orientations": int(n_orientations),
             "phase_offsets": list(phase_offsets) if phase_offsets is not None else [],
             "resume_signature": progress_signature},
            sort_keys=True,
        )
        progress = _ConvolutionProgress(save_path, final_shape, progress_kind)
        wt_final = _open_resumable_zarr(
            zarr, save_path, final_shape, zarr_chunks, compressor, progress,
            required_chunk_axes=(3, 4, 5),
        )
        use_mmap = False
        print(f"Writing convolution full-model phase {phase} directly to Zarr: {save_path}")
        print(
            "Full Model convolution cache layout | "
            f"chunks={zarr_chunks}; sigma/frequency chunks=1/1; "
            "compressor=zstd level 3"
            + (f"; Blosc threads={codec_threads}" if codec_threads is not None else "")
        )
    elif has_enough_ram(required_bytes, safety_margin=1.20):
        wt_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
        print(f"Using RAM output for convolution full-model phase {phase} ({required_bytes / (1024**3):.2f} GB).")
    else:
        _require_disk_space(folder_path, required_bytes, f"convolution full-model wavelet phase {phase}")
        wt_final = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float32, shape=final_shape)
        use_mmap = True
        print(f"Using disk-backed output for convolution full-model phase {phase}; RAM is below safe working set.")

    print(f"Convolution backend device: {device}")
    print(f"Convolution backend frame chunk size: {frame_chunk_size}")
    print(f"Convolution backend sigma/frequency group size: {filter_group_size}")

    combinations = [(s_idx, f_idx) for s_idx in range(len(sigmas)) for f_idx in range(len(frequencies))]
    total_steps = max(1, math.ceil(len(combinations) / filter_group_size))
    completed_steps = 0
    start_time = time.time()
    telemetry = OperationTelemetry(f"Convolution full-model phase {phase}")
    writer = _AsyncSliceWriter(telemetry)

    def tile_key(start, end, group_start):
        return f"g{int(group_start)}:t{int(start)}-{int(end)}"

    try:
      for group_start in range(0, len(combinations), filter_group_size):
        check_cancelled(cancel_event)
        combo_group = combinations[group_start:group_start + filter_group_size]
        if _completed_output_interval(progress, group_start, 0, num_frames):
            completed_steps += 1
            print(f"Resume: skipping completed full-model phase {phase} group {completed_steps}/{total_steps}.")
            continue
        if kernel_cache is not None:
            kernels = np.concatenate(
                [kernel_cache[int(phase), s_idx, f_idx] for s_idx, f_idx in combo_group],
                axis=0,
            )
        else:
            kernels = [
                _gabor_kernel_for_conv(theta, sigmas[s_idx], phase_offset, frequency=frequencies[f_idx])
                for s_idx, f_idx in combo_group
                for theta in thetas
            ]
        print(
            f"Convolution full-model phase {phase}: sigma/frequency group "
            f"{completed_steps + 1}/{total_steps} ({len(combo_group)} combinations)"
        )
        chunk_start = time.time()
        for chunk_index, (start, end, response) in enumerate(_conv2d_wavelet_group(
            videodata,
            kernels,
            device,
            frame_chunk_size,
            n_orientations,
            cancel_event=cancel_event,
            telemetry=telemetry,
        ), start=1):
            def write_response(values, t0=start, t1=end, pairs=tuple(combo_group)):
                for local_idx, (s_idx, f_idx) in enumerate(pairs):
                    wt_final[t0:t1, :, :, :, s_idx, f_idx] = values[..., local_idx]
            def mark_completed(key=tile_key(start, end, group_start), byte_count=response.nbytes):
                if progress is not None:
                    progress.mark(key)
                if capacity_guard is not None:
                    capacity_guard.record_completed_write(byte_count)
            if writer is not None:
                writer.submit(
                    write_response, response, response.nbytes,
                    on_complete=mark_completed,
                )
            else:
                write_start = time.perf_counter()
                write_response(response)
                telemetry.add("output", time.perf_counter() - write_start, response.nbytes)
                mark_completed()
            print(
                progress_message(
                    f"Convolution full-model phase {phase}",
                    end,
                    num_frames,
                    chunk_start,
                    unit="frames",
                )
            )
            telemetry.maybe_report()
        completed_steps += 1
        print(progress_message(f"Convolution full-model phase {phase}", completed_steps, total_steps, start_time, unit="groups"))
        gc.collect()
    finally:
        writer_error = None
        try:
            if writer is not None:
                writer.close()
        except Exception as exc:
            writer_error = exc
        finally:
            if progress is not None:
                progress.flush()
            if device == "cuda":
                _release_cuda_working_set()
        if writer_error is not None:
            raise writer_error
    if output_format == "zarr":
        progress.discard()
        print(f"Success! Saved convolution full-model Zarr array to {save_path}", end="\n\n")
    elif use_mmap:
        wt_final.flush()
        del wt_final
        print(f"Success! Saved streamed convolution full-model array to {save_path}")
    else:
        print("Saving convolution full-model array to disk...", end="\n\n")
        np.save(save_path, wt_final)
        print(f"Success! Saved convolution full-model array to {save_path}", end="\n\n")
    telemetry.report()


def getTrueRF(idx, rfs, L):
    """Function for getTrueRF.

    Args:
        idx: Input value for this operation.
        rfs: Input value for this operation.
        L: Input value for this operation.
    """
    rf=rfs[idx, :, :, :]#.swapaxes(0, 1)
    # rf = skimage.transform.resize(rf, (135, 54, 8),order=5, anti_aliasing=True)
    rfv=rf.reshape(1, -1)@L[:, :, :, 2, 0, :].reshape(-1,7290)

    plt.figure()
    plt.imshow(rfv.reshape(54, 135)[5:-5, 5:-5],  vmin=-np.max(rfv), vmax=np.max(rfv) ,cmap='coolwarm')#vmin=-0.0014, vmax=0.0014,

