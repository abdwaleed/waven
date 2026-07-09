# File lifecycle

Large files are intentionally reused across runs.

## Durable outputs

- coarse and fine Gabor libraries (`.npy` or `.zarr`);
- `<movie>_coarse_downsampled.npy`;
- `<movie>_downsampled.npy`;
- `dwt_downsampled_videodata.npy`;
- `dwt_videodata2_r.npy` / `.zarr`;
- `dwt_videodata2_i.npy` / `.zarr`;
- aligned `spikes.npy` and `pos.npy`;
- `plot_cache.pkl.gz`;
- full-model `model_results/`;
- explicit exports.

## Temporary outputs

- coarse real/imaginary phase files after the durable cache is written;
- low-RAM memory-map scratch files;
- plot-cache temporary files;
- recovery checkpoint folders after successful completion.

Interrupted long tasks can usually be resumed by rerunning the same button.
