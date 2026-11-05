# FREEZE RECORD: Version 1.2 (27 Oct 2026, 11:58)
**Decision (Scarlett, 27 Oct 11:57):** Version 1.2 is the frozen release candidate for AP40, with the lower cut-off set to 0.01 (upper 0.90 unchanged). V1 and V1.1 stay as fallbacks. Frozen means: no label, model file, feature or cut-off changes without a new version number and her decision.

**Contents.** 2,556 rows (1,023 hits, 1,533 non-hits), real I11 data only, 24 features, XGBoost 300 trees depth 3 lr 0.05, 5 seeds averaged, cut-offs 0.01 / 0.90. Labels: V1 training rows + 26 Oct decisions + 27 Oct decisions (all newer, C10 = hit).

**Evidence (all on the drive).**
- Feature tests (leave-one-run-out, V1.2): hits lost 6 (0.59%) at 0.02 and 4 (0.39%) at 0.01; hand checks 14.2% and 17.9%.
- Held-out end-to-end test 06 (DL-methionine R008, model never saw the system), at 0.02: 0 false hits, 3 of 189 hits lost (1.6%), 12.5% hand checks. At 0.01 (post hoc): 0 hits lost, 13.9% hand checks.
- Blind check 27 Oct: controls 10/10; questionable-frame labels are noisy (8/20 agreement with the newer labels; Scarlett judged the app zoom view clearer than the static images, so the newer labels stand).
- Run-start check (test 04): works as designed.

**Known risks.** V1.2 may lean towards non-hit on subtle single-spot frames (lost hits in test 06 scored 0.012 to 0.019); the lower cut-off of 0.01 was chosen after seeing test 06 (supported independently by the all-runs sweep); one held-out run of one system only; no beamline dry run yet.
**Safety net.** Unsure band 0.01 to 0.90; per-frame guard; run-start check at the start of each run; recommended lost-hit check (list hits scoring 0.005 to 0.02 on a known run) in the Thursday dry run.

**File fingerprints (md5).**
| file | md5 |
|---|---|
| classify_v1.py | 0823ac2a72d706f999bb75a503eba8e3 |
| features_v1.py | c570f37424cca121a08762fbdf356a5f |
| frame_guard.py | 15c9296b4ee63cb43c80fca555bafd9a |
| spike_guard.py | 91f194c1399a62d233755870239938ce |
| run_start_basic.py | 20f56b77b4e755b08f97efbe98759d9e |
| envelope_v1.json | f34b27d236f436c9e0ff8e5799fa2bec |
| template_run007.npz | f516baed786a81af0854fa114e83cd97 |
| config.json | 3fd16f65e3c71943be4810ae8fdf7376 |
| record.json | 277ffbe4131743b76c2e352dc2bac984 |
| model/xgb_seed0.json | 0df86cb6df50fc8a736d59b01e0b958c |
| model/xgb_seed1.json | ae49be5bccf9491210bfaf611ca84ef4 |
| model/xgb_seed2.json | cd327de22d249e434a5fd69b3908514a |
| model/xgb_seed3.json | ad55b94b2d04e1f10c535ef1d0caf1bf |
| model/xgb_seed4.json | 31cbe9a6323984608fd995558acd0146 |
| data/training_frames_v1_2.csv | 007222ddec02628987d5362460e4fab7 |
