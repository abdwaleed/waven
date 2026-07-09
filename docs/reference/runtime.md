# Runtime

Runtime helpers keep long scientific jobs practical. They estimate safe chunk
sizes, detect usable RAM/GPU resources, support cancellation, and keep the host
awake during long GUI tasks.

## Why this matters

Wavelet decomposition streams through arrays that can be hundreds of gigabytes
logically. Chunk sizing controls peak RAM and GPU memory use. The goal is to
process enough frames/filters per batch for speed without forcing the operating
system to page memory or crash the Python process.

| Helper area | Used by | Controls |
| --- | --- | --- |
| performance | wavelet and cache builders | RAM checks, chunk sizes, GPU/CPU route |
| task control | GUI and long loops | cancellation checks, progress messages |
| keep awake | GUI long tasks | prevents sleep during active processing |

::: waven.runtime.performance

::: waven.runtime.task_control

::: waven.runtime.keep_awake
