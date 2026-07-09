# Pipeline intuition

The core idea is to express a visual stimulus in a feature space that resembles
localized oriented filters. Each Gabor filter asks: "Was there structure at this
location, orientation, size, phase, and frequency at this time?"

The workflow then compares those feature time courses to neural activity.

1. **Gabor libraries** define candidate visual features.
2. **Wavelet decomposition** projects the movie onto those features over time.
3. **Coarse RF analysis** finds the feature that best correlates with each
   neuron.
4. **Tuning curves** slice the RF tensor around the preferred feature.
5. **OSI/gOSI** summarize orientation tuning strength.
6. **Simple and full models** fit nonlinear relationships between selected
   wavelet variables and neural responses.

The coarse path is designed to be fast enough for screening. The full model is
more expensive but searches at higher spatial and feature resolution.
