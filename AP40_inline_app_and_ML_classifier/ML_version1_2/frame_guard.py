"""Per-frame guard (Version 1, 23 Oct). It never changes a hit to a non-hit: any flag only sends a confident call to 'unsure' (hand check).
Rules (all thresholds set from the fault tests + real-run diagnostics, see TESTING_STATUS.md):
  FAIL    frame sanity failed (all zero, <50% data, NaN)         -> label 'error'
  SPIKES  >= SPIKE_WARN isolated single-pixel spikes (hot pixels) -> unsure
  SCALE   subtraction scale outside 0.5-2x the run's recent median -> unsure
  FIT     fit_pixel_frac more than 0.08 below the run median      -> unsure
  NEG     neg_frac more than 0.12 above the run median            -> unsure
The three run-relative rules need MIN_HIST earlier frames from the same run; before that only FAIL and SPIKES act."""
import numpy as np
from collections import deque
from spike_guard import spike_count
SPIKE_WARN = 15; SCALE_LO, SCALE_HI = 0.5, 2.0; FIT_DROP = 0.08; NEG_RISE = 0.12; MIN_HIST = 5; HIST = 30
class RunGuard:
    def __init__(self): self.h = deque(maxlen=HIST)
    def check(self, corr, info=None, sanity_ok=True, update=True):
        flags = []
        if not sanity_ok: return ['FAIL']
        sp = spike_count(corr)
        if sp >= SPIKE_WARN: flags.append('SPIKES(%d)' % sp)
        if info is not None and len(self.h) >= MIN_HIST:
            m = np.median(np.array(self.h), axis=0)
            sc, fit, neg = info['scale'], info['fit_pixel_frac'], info['neg_frac']
            r = sc / m[0] if m[0] > 0 else np.inf
            if r < SCALE_LO or r > SCALE_HI: flags.append('SCALE(x%.2f)' % r)
            if m[1] - fit > FIT_DROP: flags.append('FIT(-%.2f)' % (m[1] - fit))
            if neg - m[2] > NEG_RISE: flags.append('NEG(+%.2f)' % (neg - m[2]))
        if info is not None and update and not flags: self.h.append((info['scale'], info['fit_pixel_frac'], info['neg_frac']))
        return flags
def final_label(label, flags):
    if 'FAIL' in flags: return 'error'
    if flags and label in ('hit', 'non_hit'): return 'unsure'
    return label
