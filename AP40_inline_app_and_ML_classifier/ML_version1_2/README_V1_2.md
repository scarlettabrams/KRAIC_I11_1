# Version 1.2 (27 Oct 2026) - FROZEN RELEASE CANDIDATE (27 Oct 11:57, Scarlett). Lower cut-off 0.01. See FREEZE_RECORD_V1_2.md. Beamline dry run still to do.
Version 1.1 + the 27 Oct decision sheet (all 38 frames marked 'newer'). 31 frames / 39 rows changed label. C10 (pixium_146649) was held back first (its 26 Oct audit said non_hit); on 27 Oct 11:32 Scarlett confirmed the newer labels stand, so C10 is now hit and V1.2 was retrained (earlier build: 1,021 hits; model files replaced).
2,556 rows, 1,023 hits, 1,533 non-hits. Same XGBoost settings, cut-offs 0.02 / 0.90 unchanged. Real data only.
This folder is also a complete app-ready ML folder (point the app's 'ML_version1 folder' box at it).
Tests (features only, 3 seeds; each version scored on its own labels), V1.1 -> V1.2 (final, with C10 = hit):
- leave-one-run-out: hits lost 10 (0.98%) -> 6 (0.59%); unsure share 16.8% -> 14.2%; false hits 13 -> 14
- lower cut-off 0.01: hits lost 7 -> 4; unsure 22.7% -> 17.9%
- leave-one-system-out: DL-methionine hits lost 7 -> 4, false hits 4 -> 0; Carbamazepine 0 -> 1 (false hits 13 -> 6); Glycine 0 -> 1; L-histidine 3 -> 3
BLIND CHECK (27 Oct, blind_check_27oct): Scarlett labelled 30 frames cold from static images. Controls 10/10. On the 20 questionable frames she agreed with the 27 Oct newer labels on 8 and with the earlier labels on 12 (called 19 hits). Her view: the app's zoom view used for the newer labels is far clearer than the static images, so the newer labels stand (decision 27 Oct 11:28 and 11:32). Open risk: V1.2 may lean towards non-hit on subtle single-spot frames; the safety net is the unsure band and the lower cut-off option.

HELD-OUT TEST 06 (DL-methionine R008, model 1.2-noDLM): at 0.02 lost 3 of 189 hits (1.6%), 0 false hits, 12.5% hand checks; at 0.01 0 lost, 13.9% hand checks. Decision: freeze V1.2 with lower cut-off 0.01.
