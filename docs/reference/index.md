# Reference

The API reference combines short human-written orientation notes with generated
`mkdocstrings` signatures. Use it when you need exact call signatures, but read
the tables on each page first: they explain the array shapes, units, and
ownership boundaries that signatures alone cannot convey.

## Shape glossary

| Symbol | Meaning | Common source |
| --- | --- | --- |
| `n_trials` | repeated stimulus presentations | alignment output |
| `n_frames` | stimulus frames after alignment/truncation | movie, spikes, wavelets |
| `n_neurons` | recorded cells or units | Suite2p/ephys input |
| `analysis_x`, `analysis_y` | movie-metadata dimensions scaled by the GUI percentage | stimulus metadata |
| `coarse_nx`, `coarse_ny` | derived coarse RF grid | `AnalysisConfig.coarse_nx/coarse_ny` |
| `n_orientations` | orientation bins over 0-180 degrees | `N_thetas` |
| `n_sigmas` | Gabor size bins | `Sigmas` or `Sigmas Full Model` |
| `n_frequencies` | spatial-frequency bins | `Frequencies` |

## Most important arrays

| Object | Shape | dtype | Read this as |
| --- | --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | floating numeric | neural response time series |
| downsampled movie | `(n_frames, analysis_y, analysis_x)` | `bool` | image-space stimulus frames |
| Coarse RF power | `(n_frames, coarse_x, coarse_y, n_orientations, n_sigmas)` | `float32` | power-only Zarr feature cache used by RF correlation |
| full-model phase | `(n_frames, analysis_x, analysis_y, n_orientations, n_sigmas, n_frequencies)` | `float32` | dense real or imaginary wavelet coefficients |
| RF tensor | `(n_neurons, coarse_nx, coarse_ny, n_orientations, n_sigmas, n_frequencies)` | numeric | correlation per neuron and feature |

Movie arrays are image-like and use `(y, x)` spatial order. Wavelet, RF, and
Gabor-library arrays use feature order `(x, y)`.

## Page map

| Page | Start here when you need |
| --- | --- |
| [Configuration](config.md) | typed config objects, default axes, path fields |
| [Pipeline](pipeline.md) | high-level orchestration and dataclass outputs |
| [Wavelets](wavelets.md) | Gabor libraries, downsampling, wavelet decomposition |
| [Stimulus](stimulus.md) | loading/reusing coarse and full-model wavelet caches |
| [Analysis](analysis.md) | RF tensors, OSI/gOSI, model fitting, trial statistics |
| [Storage](storage.md) | `.npy`/Zarr loading and conversion helpers |
| [Runtime](runtime.md) | chunk sizes, RAM checks, cancellation, keep-awake behavior |
| [Time Alignment](time-alignment.md) | conversion from acquisition timing to frame-aligned spikes |
| [Spike-Triggered Averaging](spike-triggered-averaging.md) | ephys raw-count STAs, shuffle screening, and phase-sensitive Gabor fits |
| [Neural Data](neural-data.md) | Suite2p/ephys loading utilities |
| [GUI](gui.md) | interactive app entry points and long-running task plumbing |
| [Compatibility Modules](compatibility.md) | historical import paths retained for older scripts |
