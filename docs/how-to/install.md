# Install and launch

This page covers installation only. Choose the environment that matches the
computer; scientific configuration begins in
[First GUI Analysis](../tutorials/first-gui-analysis.md).

## Prerequisites

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Conda or Miniconda | Recommended installation | The checked-in environment targets Python 3.10. |
| 32 GB RAM recommended | Experiment-sized sessions | Smaller sessions can run with less; cache products stay disk-backed. |
| Local SSD/NVMe storage | Durable caches | Strongly recommended for stimulus and wavelet cache throughput. |
| NVIDIA CUDA GPU | **STRONGLY RECOMMENDED** acceleration | CPU fallback is supported. CUDA is not required to launch the GUI. |

## CUDA-enabled installation (recommended default installation)

`environment.yml` includes PyTorch CUDA 12.1 packages. Use it on a machine
with a compatible NVIDIA driver. Run the following in the **root directory** of this project:

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
python -m pip install -e .
python ui.py
```

If `libmamba` is unavailable, omit `--solver libmamba`. `python ui.py` should
be run from the repository root; Don't worry about configuring `pipeline_config.json`
before launch; GUI inputs are accepted.

## CPU-only installation

Create a Python 3.10 environment and install the CPU requirements. Run the following in
the **root directory** of this project:

```bash
conda create -n waven-cpu python=3.10 pip
conda activate waven-cpu
python -m pip install -r requirements-cpu.txt
python ui.py
```

CPU-only runs are valid but convolution and large model preparations can take
substantially longer (**NOT** recommended). Keep the initial Gabor lists and 
movie sampling modest to verify the workflow before an experiment-sized run.

## Verify the interpreter and launch

Run these commands in the activated environment:

```bash
python -c "import sys, waven; print(sys.executable); print(waven.__version__)"
python ui.py
```

If startup fails because a dependency is missing, `ui.py` displays the Python
interpreter it used and writes unexpected startup tracebacks to
`waven_launch_error.log` in the repository root. Install packages into that
same interpreter rather than a different system Python.

## Build the documentation locally

Run the following in the **root directory** of this project:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

Open the local address printed by MkDocs. Documentation links are validated as
part of the project checks described in the maintainability guide.
