# GUI workflow reference

The GUI-first walkthrough, field meanings, recommended controls, active cache
selection, analysis, and export workflow now live in
[GUI Onboarding](../tutorials/gui-onboarding.md). Start there for normal use.

## Task behavior

Long actions run outside the Tk event loop so the application can repaint and
report progress. When a task, configuration action, or export finishes, fails,
or cancels, WavEn flashes the taskbar and plays a system sound.

**Cancel** is cooperative: WavEn checks between durable chunks, neurons, or
model iterations. It stops as soon as the current interruptible operation
finishes rather than corrupting a cache. Completed tiles and valid caches remain
reusable; prefer Cancel to force-closing the application.

For cache reuse, versioning, and format planning, see
[Reusable Caches](storage.md). For saving a GUI setup, see
[Restore GUI Settings](../how-to/prepare-configuration.md).
