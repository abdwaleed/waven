# Model input, output, and scientific contracts

This reference specifies the inputs and outputs used by **Run Model** and **Run Full Model**. A model can run without raising an exception while still being scientifically invalid if a stimulus cache, neural cache, or feature grid is from a different experiment.

## Shared inputs

| Input | Type | Description | Scientific reasoning |
| --- | --- | --- | --- |
| aligned neural responses | numeric array, `(trials, frames, neurons)` | Ephys values are frame-bin firing rates in Hz; two-photon values are aligned activity. | Repeated trials permit a fit/holdout split on the same visual time base. |
| real and imaginary wavelets | paired `float32` NPY/Zarr arrays | Same shape and frame clock; preserve Gabor phase. | A phase-aware nonlinear model needs both quadrature components, unlike coarse RF power. |
| preferred RF parameters | numeric array, at least `(x, y, orientation, sigma)` by neuron | Coarse RF feature indices. The GUI supplies raw and spatially smoothed versions. | Coarse correlation limits nonlinear fitting to visually plausible features instead of searching an intractable full space. |
| training and test trial indices | non-overlapping integer lists | Every index must identify an available neural trial. | Prevents evaluating a fit on the trials used to train it. |
| frames per minute | positive integer, `round(movie_fps * 60)` | Derived from movie metadata, not a manually entered clock. | Converts a requested duration in minutes into the experiment’s exact number of frames. |

All model actions validate array rank, paired real/imaginary shape, shared frame count, preferred-index bounds, non-empty train/test splits, and the absence of overlap between training and holdout trials before fitting.

## Run Model (coarse)

When Coarse RF uses an independent frequency list, it may search every
sigma-by-frequency combination. Run Model intentionally remains a compact
five-dimensional phase model: it keeps the selected x/y/orientation/sigma seed
and rebuilds its carrier from the configured cycles-per-sigma relationship.
It does not choose frequency index zero or copy the RF frequency axis. When
the coupled carrier values are present in the RF list, cache preparation reuses
that convolution response; otherwise it writes the small real/imaginary phase
pair in a separate safe pass.

| Field | Contract |
| --- | --- |
| input wavelets | real and imaginary arrays, `(frames, coarse_x, coarse_y, orientations, sigmas)` |
| input RF indices | coarse-grid indices for every requested neuron |
| output predictions | numeric array, one held-out prediction time course per requested neuron |
| output nonlinear parameters | numeric fitted response-function parameters per neuron |
| output rho/phase parameters | feature-response summaries used by the nonlinear model |
| output metrics | held-out fit metrics, including explained-variance and correlation quantities |
| output interpolators | callable/non-serializable fit objects retained for compatible legacy consumers |

For each selected feature, the model converts real/imaginary wavelet values to amplitude and phase, computes frame-to-frame phase drift, fits its nonlinearity only on training trials, and evaluates that fitted function on held-out trials. With the single-wavelet option, the inhibitory feature is zeroed but the fitted nonlinearity is still used for prediction; it is not replaced by an unscaled raw wavelet trace.

## Run Full Model (local refinement)

| Field | Contract |
| --- | --- |
| input wavelets | real and imaginary arrays, `(frames, full_x, full_y, orientations, full_sigmas, frequencies)` |
| extra input | visual coverage `(left, right, top, bottom)` in degrees plus neuron positions `(neurons, 2+)` |
| output refined parameters | for each requested neuron, raw and smoothed local best `(x, y, orientation, sigma, frequency)` features |
| output predictions/metrics | held-out nonlinear prediction and its scalar evaluation metrics |
| output selectivity payload | legacy correlation-derived orientation summary for the refined feature; it is distinct from GUI firing-rate OSI/gOSI |
| durable output | NPY result arrays and interpolator pickle in `Full Model Save Path/model_results` |

The full model maps the coarse seed to the full feature grid, searches a local spatial neighbourhood, selects the stronger real or imaginary phase correlation at each step, then fits/evaluates the response model as above. It never mixes the full-model cache with a coarse cache. A missing Zarr pair falls back to compatible NPY memory maps; a mismatched shape produces a clear error rather than silently indexing a different feature axis.

## Orientation, STA, and model quantities are different

The individual-neuron orientation-correlation curve is a diagnostic slice of the RF tensor. The adjacent firing-rate curve is a weighted mean of the aligned neural response and is the sole source of GUI OSI/gOSI. Both curves use the same sorted orientation centres and repeat their 0-degree value at 180 degrees only for periodic visualization. The duplicate display point is excluded from the selectivity calculation, so it cannot change OSI or gOSI.

PSTH-weighted STA maps are diagnostic outputs, not model inputs. They are rate-weighted averages of preceding stimulus frames and are labelled by lag using movie FPS. The highest-variance STA lag is descriptive, not a significance test or a fitted receptive-field estimate.

## Runtime switches

The GPU, multi-GPU, compilation, batch, and writer switches change execution scheduling only. **Use compatible GPUs** first verifies device capability, memory, and an available-throughput proxy; unavailable device fields fall back to a conservative score. **Compile stable convolution kernels** falls back to the eager convolution runner if compilation setup *or the first compiled call* fails. Therefore these switches must not change requested stimulus features, array layouts, model inputs, or the scientific definition of outputs.
