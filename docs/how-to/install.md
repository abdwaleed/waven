# Install waven

## Analysis environment

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
pip install -e .
```

## Documentation environment

```bash
python -m pip install -e ".[docs]"
```

This installs MkDocs, Material for MkDocs, and mkdocstrings.

If extras installation is awkward in your shell, use:

```bash
python -m pip install -r requirements-docs.txt
```

## Hardware notes

Use local SSD/NVMe storage for movies, wavelets, and model outputs. Network
drives are often the limiting factor for this project.

Recommended starting point:

- at least 64 GB RAM;
- at least 100 GB free disk space for a real experiment;
- CUDA GPU when running large wavelet/model jobs.
