# First GUI Analysis

This guide gets the files right before you start clicking through the GUI.
WavEn is GUI-first: enter the paths and settings in the application. Save a
`pipeline_config.json` file only after a working session if you want to restore
that setup later.

<br>

## 1. What you need before opening the GUI

You need three things:

1. A writable experiment folder, conventionally named `my_experiment` (more info below).
2. One intended stimulus video.
3. Either fresh raw neural data (ephys or 2-photon) or an existing aligned neural cache.

The GUI creates all wavelet, Gabor, neural-cache, plot, model, and recovery
products. Do not pre-create them unless you are deliberately reusing an
existing cache.

### Inputs and Outputs

Critical to the inputs and outputs of this app is the `my_experiment` folder, which you
input into the app as `Project Root Folder`. Its
structure is as follows:

```text
my_experiment/
├── input/                              # You provide or point the GUI here
│   ├── raw_data/                       # Ephys or 2-photon source files
│   ├── stimulus_movie/                 # One intended stimulus video
│   └── neural_cache/                   # Optional existing aligned spikes/positions
├── cache/                              # GUI-created and reusable
│   ├── gabor/                          # Compact convolution kernels
│   └── wavelets/
│       ├── coarse/                     # Stimulus cache + Coarse RF versions
│       └── full/                       # Optional Full Model versions
└── output/                             # GUI-created session outputs
    ├── plots/
    ├── models/
    └── recovery_cache/
```

You do not have to name the folder `my_experiment`. However, the program will automatically create and follow
the filepath conventions listed in `my_experiment` when reading and generating its outputs. You might ask,
"Where do I place inputs then?" We will cover this in the section below. Suffice it to say that for all inputs,
you either manually create `input/{FOLDER_NAME}/` and paste the data or input the paths to your data in the GUI
and let the program handle `input/{FOLDER_NAME}/` for you. Please note that for the latter case, the program does
NOT copy your data but rather stores a reference JSON file to it.

| Area | You provide it? | What belongs there | Why it matters |
| --- | --- | --- | --- |
| `input/raw_data/` | Yes, for a fresh recording | Ephys session files or a 2-photon Timeline/Suite2p dataset. | Used once to create frame-aligned neural responses. |
| `input/stimulus_movie/` | Yes | One intended stimulus movie. | Its frame count, pixel size, and FPS are authoritative metadata for every later stage. |
| `input/neural_cache/` | Optional | A matching aligned `spikes` + `pos` pair. | Lets you analyse previously prepared neural data without repeating raw-data alignment. |
| `cache/` | No | Gabor kernels, downsampled stimulus, and versioned wavelet products. | Reusable when the stimulus-side settings still match. |
| `output/` | No | Plots, exported figures, fitted models, and recovery checkpoints. | Belongs to the current neural-data analysis. |

**NOTE:** The selected folder must remain available when you run the pipeline.

<br>

## 2. Put the stimulus movie in one clear location

Place one intended movie in `input/stimulus_movie/`, or choose folder path in GUI
with **Stimulus Movie Folder**. WavEn reads the
movie's width, height, frame count, and FPS automatically from the file. You enter the visual
coverage in degrees; you do not enter the movie metadata by hand.

<br>

## 3A. Raw-data layout for a fresh ephys session (2-photon section below)

Select **Ephys** in the GUI and point **Raw Data Folder** at the session root.
The current ephys loader expects:

```text
input/raw_data/
├── session_units.pkl                   # One .pkl file at the selected root containing spike_sorted data
└── digital_inputs/                     # Name is flexible; nesting is allowed
    └── Din3.dat                        # The photodiode/TTL port, for example 3; multiple DINs can be listed here
                                        # since DIN selection is done in the GUI
```

### Notes

+ The digital-input directory does **not** have to be named `digital_inputs`. WavEn finds
folders by the presence of `Din<port>.dat` inside.

+ Creating a standalone `digital_inputs/` folder is recommended for organization but is not required.

+ You specify the correct DIN port. The program can list ports but cannot identify which one
was physically connected to the photodiode**.

<br>

## 3B. Raw-data layout for a fresh 2-photon session

Select **2-photon** and point **Raw Data Folder** at a root containing the
experiment identifier you enter in the GUI. WavEn needs completed Suite2p
output and the Cortex Lab Timeline file; raw TIFFs alone are not sufficient.

```text
input/raw_data/
└── subject/
    └── 2026-05-23/
        ├── 3/
        │   ├── 2026-05-23_3_subject_Timeline.mat
        │   └── suite2p/                # Accepted location
        │       ├── plane0/
        │       │   ├── spks.npy
        │       │   ├── iscell.npy
        │       │   └── stat.npy
        │       └── plane1/ ...         # Include every configured plane
        └── suite2p/                    # Also accepted by the default discovery
            └── plane0/ ...
```

| Required item | Location rule | Used for |
| --- | --- | --- |
| `Timeline.mat` | Expected below `subject/date/experiment-number/`, matching **Experiment ID**. | Aligns neural frames with the stimulus timing. |
| Suite2p `planeN/` directories | Choose the Suite2p output root; it must contain `plane0`, `plane1`, and so on. | Supplies neural activity and ROI information. |
| `spks.npy`, `iscell.npy`, `stat.npy` | Required in every configured plane. | Supplies activity, cell mask, and ROI position data. |
| Resolution and Number of Planes | Entered in the GUI. | Defines position scaling and which Suite2p planes must be present. |

Use **Advanced 2-photon data discovery** to select a Suite2p output folder when
it lives outside the conventional experiment directory. Keep the Timeline file
available in the raw-data roots configured for the session.

<br>

## 4. Reuse caches (Skip if you have no caches to use)

Reusing a stimulus cache is advisable when you analyse new neural raw data
against the **same visual stimulus** and using the same **downsampling amount**
and **gabor parameters**. It avoids repeating the most expensive video and wavelet work.

You may reuse the downsampled stimulus cache, Gabor-kernel cache, and Coarse RF wavelet cache
when all of these match:

+ the underlying stimulus video;
+ visual and analysis coverage;
+ the sampling grid derived from those choices; and
+ the requested Gabor settings: orientations, sigmas, frequencies, phases, and
  frequency mode.

**THE GOOD NEWS:** New raw neural data does **not** invalidate those stimulus-derived products.
Create a new aligned neural cache for the new recording, choose the matching
Coarse RF cache version, and run analysis. WavEn validates compatibility before
it uses a selected cache, so it stops rather than silently mixing a mismatched
movie or filter bank with your neural data.

**IMPORTANT:** The Coarse RF Cache Library Folder can hold several cache versions. In the GUI,
choose the active version from **Active cache versions** before **Run Coarse RF
Analysis**. If preparation encounters a different configuration, choose **No**
in the replacement dialog to preserve the current version and create a new
timestamped cache folder. See [Reusable Caches](../reference/storage.md) for
the complete cache map and selection rules.

<br>

## 5. Continue in the GUI

You are ready to proceed through the GUI. The next page contains a visual walkthrough from A to Z.

Up Next: [GUI Onboarding](gui-onboarding.md).
