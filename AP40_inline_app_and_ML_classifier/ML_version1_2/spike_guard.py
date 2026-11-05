"""Hot-pixel (isolated single-pixel spike) guard. Counts pixels that are far above the local background (z>Z) whose 8 neighbours are all
much lower (<NB x the spike height above background). Real diffraction spots spread over several pixels; hot pixels do not."""
import numpy as np
from scipy import ndimage as ndi
CY, CX, H = 1437, 1440, 700
def spike_count(frame, Z=15.0, NB=0.2, k=41):
    a = np.asarray(frame)
    if a.shape != (2*H, 2*H): a = a[CY-H:CY+H, CX-H:CX+H]
    a = a.astype(np.float32); valid = a != 0; w = valid.astype(np.float32)
    n = ndi.uniform_filter(w, k) + 1e-6
    mu = ndi.uniform_filter(a*w, k)/n; var = ndi.uniform_filter(a*a*w, k)/n - mu*mu
    sd = np.sqrt(np.maximum(var, 1e-6)); z = (a-mu)/sd; z[~valid] = 0
    h = a - mu
    cand = (z > Z) & (n > 0.8)
    if not cand.any(): return 0
    nb = ndi.maximum_filter(h, size=3, footprint=np.array([[1,1,1],[1,0,1],[1,1,1]], bool))
    return int((cand & (nb < NB*h)).sum())
