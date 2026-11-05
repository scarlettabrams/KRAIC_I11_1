"""Run-start check, basic version (checks and warns only; it NEVER changes the classifier, the cut-offs or any label).
Give it N corrected 2D frames of a run that should be EMPTY (the frames used to build the reference, or first frames you have confirmed are non-hit) (arrays or paths read by the caller). It returns OK / WARN / FAIL with plain-words reasons.
Envelope = ranges of the 24 features seen in real non-hit training frames (envelope_v1.json, made by make_envelope()).
    from run_start_check import check_run_start
    res = check_run_start(frames, clf)      # frames: list of 2D arrays, ideally the first 10 of the run
    print(res['status'], res['reasons'])"""
import os, json
import numpy as np
from features_v1 import features, CY, CX, H

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ['frac_z_gt3', 'z_kurt', 'n_peaks_z4', 'max_z']      # noise level / spot level; fixed in advance, not tuned
N_KEY_OUT = 2          # WARN if this many key medians are outside the training non-hit range
HIT_FRAC_WARN = 0.5    # WARN if this fraction of the first frames score as hits
MIN_VALID = 0.5        # FAIL if less than this fraction of the crop is non-zero
LO_Q, HI_Q = 1, 99

def make_envelope(training_csv, out_json, exclude=None):
    import pandas as pd
    d = pd.read_csv(training_csv); d = d[d.y == 0]
    if exclude is not None: d = d[~exclude(d)]
    env = {c: [float(np.percentile(d[c], LO_Q)), float(np.percentile(d[c], HI_Q))] for c in KEY}
    env['_n_nonhit_frames'] = int(len(d))
    if out_json: json.dump(env, open(out_json, 'w'), indent=1)
    return env

BIN = 10; IOU_WARN = 0.8
def zero_map(a):
    c = np.asarray(a)[CY-H:CY+H, CX-H:CX+H]
    z = (c == 0).reshape(2*H//BIN, BIN, 2*H//BIN, BIN).mean(axis=(1, 3)) > 0.5
    return z

def make_template(frame, out_npz=None):
    z = zero_map(frame)
    if out_npz: np.savez_compressed(out_npz, zero=z)
    return z

def mask_overlap(a, template):
    z = zero_map(a); u = (z | template).sum()
    return 1.0 if u == 0 else float((z & template).sum() / u)

COV_WARN = 0.97
def calibration_map(mask_npy):
    """The detector mask saved with the calibration file (tab 0), as the same coarse map used for frames. True = masked pixel."""
    m = np.load(mask_npy) > 0
    return m[CY-H:CY+H, CX-H:CX+H].reshape(2*H//BIN, BIN, 2*H//BIN, BIN).mean(axis=(1, 3)) > 0.5

def mask_coverage(a, cal):
    """Share of the calibration mask's pixels that are empty (zero) in the frame. 1.0 = every masked pixel is empty. One-sided: extra
    empty pixels (extra 2theta masks added at subtraction) are allowed."""
    z = zero_map(a); return float((z & cal).sum() / max(cal.sum(), 1))

def frame_sanity(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] < CY + H or a.shape[1] < CX + H:
        return 'FAIL', 'frame shape %s is too small for the central crop' % (a.shape,), 0.0
    c = a[CY-H:CY+H, CX-H:CX+H]
    if not np.isfinite(c).all(): return 'FAIL', 'frame has NaN or infinite values', 0.0
    v = float((c != 0).mean())
    if v == 0: return 'FAIL', 'frame is all zero (shutter closed, or nothing written)', 0.0
    if v < MIN_VALID: return 'FAIL', 'only %.0f%% of the frame has data (expected most of it)' % (100 * v), v
    return 'OK', '', v

def check_run_start(frames, clf, envelope=None, feats=None, template=None, calibration=None):
    """feats: optional precomputed list of feature dicts (used in testing); frames may then be None."""
    env = envelope or json.load(open(os.path.join(HERE, 'envelope_v1.json')))
    reasons, status = [], 'OK'
    def bump(s):
        nonlocal status
        if s == 'FAIL' or (s == 'WARN' and status == 'OK'): status = s
    F, scores, ious, covs = [], [], [], []
    n = len(feats) if feats is not None else len(frames)
    for i in range(n):
        if feats is None:
            s, why, v = frame_sanity(frames[i])
            if s == 'FAIL':
                bump('FAIL'); reasons.append('frame %d: %s' % (i + 1, why)); continue
            f = features(frames[i])
        else: f = feats[i]
        if feats is None and template is not None:
            iou = mask_overlap(frames[i], template); ious.append(iou)
        if feats is None and calibration is not None:
            covs.append(mask_coverage(frames[i], calibration))
        F.append(f); scores.append(clf.score_features(f))
    if not F:
        return dict(status='FAIL', reasons=reasons + ['no usable frames'], n=n, scores=[], medians={})
    sc = np.array(scores); nusable = len(F)
    if nusable < n:
        bump('WARN'); reasons.append('%d of %d frames could not be used' % (n - nusable, n))
    hf = float((sc >= clf.upper).mean()); uf = float(((sc >= clf.lower) & (sc < clf.upper)).mean())
    if hf >= HIT_FRAC_WARN:
        bump('WARN'); reasons.append('%.0f%% of the first frames score as hits: the reference may be wrong, or the run starts with crystals' % (100 * hf))
    # unsure fraction is reported only: empty frames of an unseen compound often score 0.02 to 0.15 (unsure) as normal
    if ious and float(np.median(ious)) < IOU_WARN:
        bump('WARN'); reasons.append('the empty-pixel pattern (beam stop, gaps, masks) overlaps the reference pattern by only %.0f%%: beam centre, mask or detector setup may have changed' % (100 * float(np.median(ious))))
    if covs and float(np.median(covs)) < COV_WARN:
        bump('WARN'); reasons.append('only %.0f%% of the calibration mask pixels are empty in these frames: the frames were not made with this calibration mask (wrong calibration file, or the beam centre has moved)' % (100 * float(np.median(covs))))
    med = {c: float(np.median([f[c] for f in F])) for c in KEY}
    out = [c for c in KEY if not (env[c][0] <= med[c] <= env[c][1])]
    if len(out) >= N_KEY_OUT:
        bump('WARN'); reasons.append('frame statistics outside the usual range of non-hit frames: ' + ', '.join(out))
    return dict(status=status, reasons=reasons or ['first frames look like the training data'], n=n, iou_median=(float(np.median(ious)) if ious else None), cov_median=(float(np.median(covs)) if covs else None), hit_frac=hf, unsure_frac=uf,
                scores=[round(float(s), 4) for s in sc], medians=med, outside=out)
