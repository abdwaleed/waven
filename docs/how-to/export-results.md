# Export results

The GUI has four export buttons:

- **Export All Displayed Results**
- **Export All Neurons Tab**
- **Export Individual Neuron Tab**
- **Export All Individual Neurons**

Each exported graph can include:

- PNG image;
- SVG vector file;
- `.npy`, `.zarr`, or both for reusable arrays;
- Python pickle with richer payload data;
- JSON manifest describing files, axes, and array keys.

Use exports when you need a figure and the numerical data that produced it.
The selected array format does not alter PNG, SVG, JSON, or pickle outputs. The
all-individual-neurons export creates one directory per neuron and can therefore
be large for recordings with many cells or units.
