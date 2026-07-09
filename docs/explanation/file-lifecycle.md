# File lifecycle

Large files are intentionally reused across runs.

## Durable outputs

| Output | Typical shape or contents | dtype/format | Keep? |
| --- | --- | --- | --- |
| coarse Gabor library | `(coarse_nx, coarse_ny, n_orientations, n_sigmas, n_phases, coarse_nx * coarse_ny)` | `.npy`, `float16` | Yes, if config axes match. |
| fine Gabor library | `(NX, NY, n_orientations, n_sigmas, n_frequencies, n_phases, NX * NY)` | `.npy`, `float16` | Yes, but it can be very large. |
| `<movie>_coarse_downsampled.npy` | `(n_frames, coarse_ny, coarse_nx)` | `bool` | Yes, cheap and useful for resume. |
| `<movie>_downsampled.npy` | `(n_frames, NY, NX)` | `bool` | Yes, cheap relative to wavelets. |
| `dwt_downsampled_videodata.npy` | `(3, n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `.npy`, `float32` | Yes, this is the durable coarse RF cache. |
| `dwt_videodata2_r.npy` / `.zarr` | `(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)` | `float32` | Yes when running full models. |
| `dwt_videodata2_i.npy` / `.zarr` | same as real phase | `float32` | Yes when running full models. |
| aligned `spikes.npy` and `pos.npy` | spikes: `(n_trials, n_frames, n_neurons)`; positions: `(n_neurons, 2 or 3)` | numeric `.npy` | Yes, these make reruns reproducible. |
| `plot_cache.pkl.gz` | precomputed GUI plotting payload | compressed pickle | Optional; keep for faster GUI startup. |
| `model_results/` | parameters, predictions, metrics, exports | mixed arrays and metadata | Yes for analysis records. |
| explicit exports | PNG/SVG/NPZ/JSON/Pickle bundles | mixed | Yes for figures and paper provenance. |

## Temporary outputs

| Output | Why it exists | Removal rule |
| --- | --- | --- |
| `dwt_videodata_0.npy`, `dwt_videodata_1.npy` | Intermediate coarse real/imaginary phase files used to build the durable cache. | Can be regenerated from the coarse downsampled movie and coarse library. |
| `dwt_r_downsampled.mmap`, `dwt_i_downsampled.mmap`, `dwt_c_downsampled.mmap` | Low-RAM scratch arrays for coarse cache construction. | Remove after successful cache creation. |
| plot-cache temporary files | Atomic writes for GUI cache updates. | Remove after successful cache write. |
| recovery checkpoint folders | Resume metadata for long GUI tasks. | Removed after success/cancel; kept after failure for debugging. |

Interrupted long tasks can usually be resumed by rerunning the same button.

## Practical storage advice

Keep durable outputs on a local SSD when possible. Network drives can make
wavelet decomposition look stalled because every chunk read/write crosses the
network. If disk space is limited, delete full-model phase files before deleting
libraries or downsampled movies; the phase files are usually the largest and can
be regenerated from smaller inputs.
