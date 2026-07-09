# Build the documentation

Install documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

If your shell has trouble with extras quoting on Windows, install the docs stack
directly:

```bash
python -m pip install -r requirements-docs.txt
```

Serve locally:

```bash
python -m mkdocs serve
```

Build static output:

```bash
python -m mkdocs build
```

The rendered site is written to `site/`, as configured in `mkdocs.yml`.

!!! tip "Windows PATH"
    The `mkdocs.exe` script may be installed into
    `C:\Users\<you>\AppData\Roaming\Python\Python313\Scripts`, which is not
    always on `PATH`. Using `python -m mkdocs ...` avoids that problem.
