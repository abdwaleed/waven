# Compatibility modules

These modules preserve older import paths. New code should prefer the lowercase
domain packages used elsewhere in the reference.

Compatibility pages can be noisy because they expose historical modules with
large APIs. Treat them as migration references:

| Historical module | Prefer new code from |
| --- | --- |
| `waven.WaveletGenerator` | `waven.wavelets.filters`, `waven.wavelets.decomposition` |
| `waven.LoadPinkNoise` | `waven.stimulus.wavelet_cache`, `waven.stimulus.full_model` |
| `waven.Analysis_Utils` | `waven.analysis.*` and `waven.pipeline` |
| `waven.zebraGUI` | `waven.app.gui` |

If you are writing a new script, start with [Pipeline](pipeline.md). Use this
page only when an older notebook imports one of these names.

::: waven.WaveletGenerator

::: waven.LoadPinkNoise

::: waven.Analysis_Utils

::: waven.zebraGUI
