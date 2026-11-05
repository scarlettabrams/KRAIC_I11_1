# Inline App: Start-up Guide and Git Upload (5 Nov 2026)

Online copy: "Inline App: Start-up Guide and Git Upload" (Claude Docs).

## Where every file lives
Working copy (run from here): E:\2D_sub_appV4_trial\ML+AppDevelopment\beamline_pack_AP40\inline_app
Git copy:                     <your clone of KRAIC_I11_1>\AP40_inline_app_and_ML_classifier

inline_app\
  fep2d_app_inline.py   inline_worker.py   live_pattern.py   replay_test.py
  fep2d_pipeline.py     fep2d_v3.py
  pages\Live_viewer.py          <- folder must be called exactly "pages"
beamline_pack_AP40\ML_version1_2   <- frozen classifier, one folder up

## One-off set-up
1. streamlit --version  (must be 1.37 or newer)
2. pip install -U streamlit plotly pybaselines pyFAI h5py xgboost scipy pandas matplotlib
   (the pack's requirements.txt does not list pybaselines or plotly; the app needs both)
3. Copy all of beamline_pack_AP40 to the beamline laptop, keeping ML_version1_2 and inline_app side by side.
4. Keep the output folder on the laptop's local drive during a run; copy to the hard drive afterwards.
5. Allow pop-ups for localhost so the live viewer opens in its own window.

## Start-up each session
cd /d "E:\2D_sub_appV4_trial\ML+AppDevelopment\beamline_pack_AP40\inline_app"
streamlit run fep2d_app_inline.py

1. Sidebar: run name, output folder, watched folder.
2. Tab 0: load .poni and mask (new reference here if needed; 46 frames recommended, 30 minimum).
3. Check the classifier line: Version 1.2 (frozen), cut-offs 0.01 / 0.9.
4. File pattern: prefix I11-1-, suffix _subtracted (optional), types npy, hdf, nxs. Day one: confirm against the engineer's real file names.
5. Press Start worker (status "Worker running" within ~20 s).
6. Click "Open the live 1D pattern in a new window"; move it to another screen.
The app remembers sidebar settings: check run name and watched folder each time.

## During a run
Tab 1 Watcher | Tab 2 sorting (your label overrides the classifier) | Tab 3 integrate + baseline (segment dropdown) | Tab 4 pattern stacker.
Conditions changed: sidebar > Run events > choose, note, Mark it. Closes the segment and saves its final cumulative pattern in 04_merged.
New reference needed (e.g. antisolvent ratio) or run finished: type the new run name, wait for backlog 0, Start new run, then: new reference in tab 0, restart the subtraction script, set the watched folder, Start worker.
Worker stopped or app closed: start it again; it resumes and skips frames already scored.

## Uploading to git
cd /d "C:\Users\ezxsa27\I11_PXRD_SSA"
git status
git add inline_app
git commit -m "Inline app: <what changed>"
git remote -v        (nothing printed = no remote; also copy to the hard drive)
git push
git status           (should say up to date)

Do not commit frames or outputs. Leave out inline_last_session.json and __pycache__ (stage named files instead of the folder).
Keep the pack copy and the repo copy identical: copy any fixed file to both before committing.

## If something goes wrong
- Worker won't start: the red message names what is missing (poni, mask, watched folder, classifier).
- Classifier not Version 1.2: point the box at beamline_pack_AP40\ML_version1_2.
- Counts stay 0: file names don't match the pattern, or wrong watched folder.
- Viewer won't open: allow pop-ups, or type /Live_viewer after the app address.
- Worker not responding: press Start worker again; it resumes.
- Start new run greyed out: wait until the backlog is 0.

The classifier behind the hit labels is the frozen Version 1.2, trained on real diffraction data.
