"""The 24 per-frame features used by Version 1 (identical to extract_features_v2.py, but takes an array as well as a file).
A corrected 2D frame goes in; a dict of 24 numbers comes out. Central 1400 x 1400 crop around (CY, CX)."""
import numpy as np
from scipy import ndimage as ndi
CY, CX, H = 1437, 1440, 700
R0, R1 = 60, 700
FEATURE_ORDER = None   # set from record.json by the classifier

def crop_from_file(path):
    import h5py
    with h5py.File(path, 'r') as h:
        return h['entry/data/data'][0, CY-H:CY+H, CX-H:CX+H].astype(np.float32)

def features(frame):
    """frame: the full corrected 2D frame (2D array) OR an already-cropped 1400 x 1400 array."""
    a = np.asarray(frame)
    if a.shape != (2 * H, 2 * H):
        a = a[CY-H:CY+H, CX-H:CX+H]
    a = a.astype(np.float32)
    valid = (a != 0)
    yy, xx = np.mgrid[-H:H, -H:H]
    r = np.hypot(yy, xx); th = np.arctan2(yy, xx)
    ann = (r >= R0) & (r <= R1)
    w = valid.astype(np.float32)
    k = 41
    n = ndi.uniform_filter(w, k) + 1e-6
    mu = ndi.uniform_filter(a * w, k) / n
    var = ndi.uniform_filter(a * a * w, k) / n - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-6))
    z = (a - mu) / sd
    z[~valid] = 0
    ok = ann & (n > 0.8)
    mx = ndi.maximum_filter(z, size=7)
    peaks = (z == mx) & ok & (z > 4)
    zp = z[peaks]; rp = r[peaks]; tp = th[peaks]
    f = {}
    for t in (4, 6, 8, 10, 15, 25):
        f['n_peaks_z%d' % t] = int((zp > t).sum())
    f['max_z'] = float(zp.max()) if len(zp) else 0.0
    top = np.sort(zp)[::-1]
    f['mean_top5_z'] = float(top[:5].mean()) if len(top) else 0.0
    f['sum_z_gt6'] = float(zp[zp > 6].sum())
    s6 = peaks & (z > 6)
    rr = r[s6]; tt = th[s6]
    f['radial_std'] = float(rr.std()) if len(rr) > 1 else 0.0
    f['radial_range'] = float(np.ptp(rr)) if len(rr) > 1 else 0.0
    if len(tt):
        C = np.cos(tt).mean(); S = np.sin(tt).mean(); f['azim_R'] = float(np.hypot(C, S))
    else:
        f['azim_R'] = 0.0
    bins = np.histogram(rr, bins=6, range=(R0, R1))[0] if len(rr) else np.zeros(6)
    for i, b in enumerate(bins): f['n_rbin%d' % i] = int(b)
    lab, nl = ndi.label((z > 6) & ann)
    f['n_blobs_z6'] = int(nl)
    if nl:
        sizes = np.bincount(lab.ravel())[1:]
        f['blob_area_med'] = float(np.median(sizes)); f['blob_area_max'] = float(sizes.max())
    else:
        f['blob_area_med'] = 0.0; f['blob_area_max'] = 0.0
    zz = z[ok]
    f['z_kurt'] = float(((zz - zz.mean()) ** 4).mean() / (zz.var() ** 2 + 1e-9))
    f['frac_z_gt3'] = float((zz > 3).mean())
    ys, xs = np.nonzero(s6)
    if len(ys) > 1:
        P = np.c_[ys, xs].astype(np.float32)[:300]
        D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); np.fill_diagonal(D, 1e9)
        f['nn_med'] = float(np.median(D.min(1)))
    else:
        f['nn_med'] = 0.0
    return f
