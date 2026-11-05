# AP40 inline app and ML classifier

Everything needed at AP40 (I11, 4 to 6 February 2027) to sort, integrate and merge 2D PXRD frames while the run is going.
The classifier is the frozen **Version 1.2**, trained on **real I11 diffraction data only** (2,556 labelled frames: 1,023 hits and 1,533 non-hits; XGBoost, mean of 5 seeds).

```
AP40_inline_app_and_ML_classifier/
  inline_app/            the Streamlit app, the worker, the live viewer and the rehearsal tool
    fep2d_app_inline.py  the app (tabs 0 to 4, sidebar)
    inline_worker.py     background worker: watch folder, classify, integrate, log
    live_pattern.py      shared code for run segments and the cumulative pattern
    pages/Live_viewer.py pop-out live 1D pattern window (folder must be called "pages")
    replay_test.py       rehearsal: replays saved frames into a folder at beamline pace
    fep2d_pipeline.py    maths shared with the offline app (unchanged copy)
    fep2d_v3.py          maths shared with the offline app (unchanged copy)
  ML_version1_2/         the frozen classifier (code, config, 5 model files, freeze record)
  docs/Inline_App_Startup_Guide.md
  requirements.txt
```

`fep2d_pipeline.py` and `fep2d_v3.py` also exist in `src/KRAIC_I11_1/fep_subtraction/`. These are the copies the app was tested with. Do not edit one without the other.

## Install

```
pip install -r requirements.txt
```

## Run the app

```
cd AP40_inline_app_and_ML_classifier\inline_app
streamlit run fep2d_app_inline.py
```

In the sidebar set the run name, the output folder and the watched folder (where the subtraction script writes `I11-1-<collection number>_subtracted.<npy|hdf|nxs>`), load the .poni and mask in tab 0, check the classifier line reads **Version 1.2 (frozen), cut-offs 0.01 / 0.9**, then press Start worker. The full walk-through is in `docs/Inline_App_Startup_Guide.md`.

## How the classifier is found and loaded

The app looks for the classifier folder, in this order: the `ML_V1_DIR` environment variable, `inline_app/ML_version1_2`, then `ML_version1_2` next to `inline_app` (this repo layout), then the AP40 pack on the thesis drive. You can also type the folder into the sidebar box. The sidebar shows the version, training size and cut-offs it actually found, and blocks the worker if the folder is not usable.

To use it by hand:

```python
import sys
sys.path.insert(0, "AP40_inline_app_and_ML_classifier/ML_version1_2")
from classify_v1 import ClassifierV1
clf = ClassifierV1()                   # loads the folder it sits in
label, score = clf.classify(frame)     # frame: full corrected 2D array, or a path to the file
# label: "hit", "non_hit" or "unsure"; score = probability of hit (0 to 1)
# cut-offs come from config.json: score < 0.01 non_hit, score >= 0.90 hit, otherwise unsure
```

`frame_guard.py`, `spike_guard.py` and `run_start_basic.py` in the same folder are the safety checks. They can only move a frame to unsure or error, never to hit. Details are in `ML_version1_2/README_V1_2.md` and `FREEZE_RECORD_V1_2.md`. The training table (`data/training_frames_v1_2.csv`) is not in this repo; it stays on the thesis drive.

## Rehearse without a beamtime

```
python replay_test.py --src <folder of corrected frames> --dest <empty watched folder> --interval 5 --mode mixed --format hdf --suffix "" --n 40
```

## Status (5 Nov 2026)

Rehearsed on a laptop with 40 real AP39 CBZ frames replayed at 5 s intervals, including half-written files. The worker kept up with no backlog, and the live viewer, run events, new-run prompt, segment merge and stop and restart worked. Not yet tested against the engineer's live script at the beamline: confirm on day one that its file names match the sidebar pattern.
