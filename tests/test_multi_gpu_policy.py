"""Tests for the conservative synchronous multi-GPU selection policy."""

from types import SimpleNamespace

from waven.runtime import performance
from waven.runtime.performance import (
    ConvolutionWorkload,
    plan_convolution_execution,
    select_compatible_multi_gpu_ids,
)


def _gpu(device_id, name, capability, score, vram_gib):
    return {
        "id": device_id,
        "name": name,
        "capability": capability,
        "throughput_score": score,
        "vram_bytes": vram_gib * 1024**3,
    }


def _high_orientation_workload(frames=120):
    return ConvolutionWorkload(
        frame_count=frames,
        spatial_pixels=137 * 77,
        filter_channels=320,  # real + imaginary banks for 160 orientations
        output_channels=160,
        kernel_height=91,
        kernel_width=91,
        output_buffer_count=2,
    )


def test_mismatched_architectures_are_not_combined_in_dataparallel():
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            _gpu(0, "RTX 3060 Ti", (8, 6), 38_000, 8),
            _gpu(1, "GTX 1660 Ti", (7, 5), 24_000, 6),
        )
    )

    assert ids == (0,)
    assert rejected and "compute capability" in rejected[0]


def test_similarly_matched_cards_are_retained_for_dataparallel():
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            _gpu(0, "RTX A5000", (8, 6), 64_000, 24),
            _gpu(1, "RTX A5000", (8, 6), 62_000, 24),
        )
    )

    assert ids == (0, 1)
    assert rejected == ()


def test_live_free_vram_selects_the_primary_gpu_before_device_index():
    """A busy cuda:0 must not become primary merely because of its index."""
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            {**_gpu(0, "RTX A5000", (8, 6), 64_000, 24), "free_vram_bytes": 4 * 1024**3},
            {**_gpu(1, "RTX A5000", (8, 6), 63_000, 24), "free_vram_bytes": 20 * 1024**3},
        )
    )

    assert ids == (1, 0)
    assert rejected == ()


def test_large_throughput_gap_is_not_combined_even_with_matching_architecture():
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            _gpu(0, "Fast", (8, 6), 80_000, 16),
            _gpu(1, "Slow", (8, 6), 40_000, 16),
        )
    )

    assert ids == (0,)
    assert "throughput" in rejected[0]


def test_gpu_descriptor_accepts_property_objects_without_clock_rate(monkeypatch):
    properties = SimpleNamespace(
        name="Clockless test GPU",
        major=8,
        minor=6,
        total_memory=8 * 1024**3,
        multi_processor_count=20,
    )
    monkeypatch.setattr(
        performance.torch.cuda,
        "get_device_properties",
        lambda _device_id: properties,
    )

    descriptor = performance._gpu_descriptor(0)

    assert descriptor["throughput_score"] == 20
    assert descriptor["capability"] == (8, 6)


def test_high_orientation_workload_uses_gather_free_frame_sharding():
    """High response/output memory benefits from splitting frames, not gather."""
    descriptors = (
        {**_gpu(0, "RTX A5000", (8, 6), 64_000, 24), "free_vram_bytes": 20 * 1024**3},
        {**_gpu(1, "RTX A5000", (8, 6), 63_000, 24), "free_vram_bytes": 20 * 1024**3},
    )

    plan = plan_convolution_execution(
        _high_orientation_workload(), allow_multi_gpu=True, descriptors=descriptors,
    )

    assert plan.strategy == "frame_parallel"
    assert plan.devices == ("cuda:0", "cuda:1")
    assert plan.frames_per_device == 60


def test_multi_gpu_policy_rejects_tiny_or_memory_starved_slices():
    """Avoid multi-GPU overhead and peer OOMs despite matching hardware."""
    roomy = (
        {**_gpu(0, "RTX A5000", (8, 6), 64_000, 24), "free_vram_bytes": 20 * 1024**3},
        {**_gpu(1, "RTX A5000", (8, 6), 63_000, 24), "free_vram_bytes": 20 * 1024**3},
    )
    tiny = plan_convolution_execution(
        _high_orientation_workload(frames=12), allow_multi_gpu=True, descriptors=roomy,
    )
    assert tiny.strategy == "single"
    assert "frames per GPU" in tiny.reason

    starved = (
        roomy[0],
        {**roomy[1], "free_vram_bytes": 512 * 1024**2},
    )
    low_memory = plan_convolution_execution(
        _high_orientation_workload(), allow_multi_gpu=True, descriptors=starved,
    )
    assert low_memory.strategy == "single"
    assert "lacks free VRAM" in low_memory.reason
