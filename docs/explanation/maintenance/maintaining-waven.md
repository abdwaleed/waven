# Maintaining WavEn

The codebase is organised around explicit ownership boundaries so a change to a
button label, cache format, or analysis calculation does not silently alter
unrelated stages. Use [System Architecture and Code Ownership](system-architecture.md)
to locate the correct module before editing.

## Safe change workflow

1. Identify the stage contract: inputs, output shape/format, consumer, and
   cache provenance.
2. Make pure numerical/storage changes outside `app/gui.py` whenever possible.
3. For a GUI setting, update widget state, worker snapshots, JSON save/load,
   and the configuration documentation together.
4. For a long-running operation, accept and check the cancellation event at
   meaningful chunk/neuron/iteration boundaries.
5. Preserve user-created files and unrelated dirty-worktree changes.
6. Validate syntax and formatting before handoff.

## Current maintenance boundaries

| Change type | First place to inspect | Typical companion change |
| --- | --- | --- |
| New scientific parameter | `config.py`, relevant analysis/wavelet module | GUI field, provenance/fingerprint, docs. |
| New cache product | producing wavelet/analysis module + `storage/` | consumer validation, disk estimate, docs. |
| New GUI task | `app/gui.py` | worker snapshot, progress/cancel wiring, notification, docs. |
| Export behavior | `app/gui.py`, `gui_support/export_selection.py` | flat-name contract and export docs. |
| Hardware option | `app/gui.py` runtime specs + `runtime/performance.py` | process environment application and configuration docs. |
| Neural workflow | `data/neural.py`, `time_alignment.py`, workflow adapter | neural cache validation and input-contract docs. |

## Validation baseline

Run the checks that match the change:

```bash
python -m compileall -q src tests
git diff --check
python -m mkdocs build --strict
```

Run focused tests when `pytest` is installed in the active environment. GUI
changes also merit a manual launch with a small cache because widget layout,
thread scheduling, and available hardware are runtime concerns.

## Documentation is part of the interface

The documentation deliberately has one ordered onboarding route and focused
reference pages. Avoid duplicating field explanations across pages. Instead,
put first-use reasoning in the onboarding guide, exact field/type/requiredness
in configuration or data-contract references, and implementation ownership in
source architecture. Remove or redirect stale documentation when an interface
is simplified.
