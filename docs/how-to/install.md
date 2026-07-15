# Install waven

This page is for both a first-time analyst and a developer modifying the code.
Use the repository's `environment.yml`: it pins a compatible scientific stack,
Tk GUI support, Zarr, and the CUDA-enabled PyTorch build used by the current
wavelet backend.

## First-time setup (Windows, macOS, or Linux)

1. Install [Miniconda or Anaconda](https://docs.conda.io/projects/miniconda/).
2. Open an Anaconda Prompt or terminal in the repository root—the folder that
   contains `environment.yml` and `ui.py`.
3. Create and activate the environment:

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
python -m pip install -e .
```

`-e` means *editable*: imports point at `src/waven`, so a developer can change
code without reinstalling the package.

4. Launch the GUI from that same root:

```bash
python ui.py
```

If the GUI reports a missing package, confirm the interpreter is the activated
environment with `python -c "import sys; print(sys.executable)"`, then rerun
`python -m pip install -e .`.

## GPU and CPU installations

The checked-in Conda environment requests CUDA 12.1 PyTorch. It is the
recommended route for NVIDIA systems. The application remains functional on a
CPU-only computer: convolution and Coarse RF fall back safely, but long wavelet
and model jobs will take longer. Do not install CUDA separately merely to run
the GUI; PyTorch supplies the needed runtime when the driver is compatible.

For a CPU-only developer environment, create the Conda environment and replace
the PyTorch dependency with the CPU build appropriate for your platform before
running `pip install -e .`. Keep `numpy` below 2.0 when using Suite2p/Numba
components.

## Developer workflow

Run these checks before handing off a change:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('src').rglob('*.py')]; print('syntax OK')"
```

For a GUI change, also start `python ui.py`, select a small test movie, and run
the staged smoke path: prepare stimulus cache → neural cache → Gabor assets →
the required wavelet product → Coarse RF. Never validate a large cache change
by first running an experiment-scale movie.

## Documentation environment

The editable docs extra installs MkDocs, Material, and mkdocstrings:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

Use `python -m mkdocs build` for a static site in `site/`. If extras quoting is
awkward in a Windows shell, use `python -m pip install -r requirements-docs.txt`.

## Hardware and storage preflight

Use a local SSD/NVMe for the movie, `cache/`, and `output/` folders; a network
drive often becomes the bottleneck even with a fast GPU. Start with at least
64 GB RAM and 100 GB free storage for a real experiment, then use the logical
size estimates in the Wavelets tab before creating full-model products. Waven
keeps queues bounded and falls back from GPU tiles to CPU when required, but
storage capacity remains a user responsibility.

## CPU-only systems

Waven's current convolution and Coarse RF implementations automatically
fall back to CPU when CUDA is unavailable. On a system without an NVIDIA GPU,
install the CPU PyTorch wheel rather than the CUDA requirements:

```powershell
python -m pip install -r requirements-cpu.txt
```

The GPU-only legacy helpers also use the same safe device resolver. CUDA
options in the GUI remain saved preferences and have no effect until a CUDA GPU
is available.
