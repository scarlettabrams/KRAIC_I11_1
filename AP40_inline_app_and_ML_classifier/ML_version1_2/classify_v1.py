"""Version 1 classifier: corrected 2D frame in, hit / non_hit / unsure out.
Real I11 data only. XGBoost (mean of 5 seeds); the random forest backup is selectable with model='rf'.
Cut-offs are settings (config.json), not part of the trained model, so they can change without retraining.

    from classify_v1 import ClassifierV1
    clf = ClassifierV1()                      # loads the folder next to this file
    label, score = clf.classify(frame)        # frame: 2D array (full corrected frame) or path to the .hdf
    clf.classify_many(paths, log_csv='scores_log.csv')
score = probability of hit (0 to 1). label: score < lower -> non_hit; score >= upper -> hit; else unsure."""
import os, json, time, csv, pickle, datetime
import numpy as np
from features_v1 import features, crop_from_file

HERE = os.path.dirname(os.path.abspath(__file__))

class ClassifierV1:
    def __init__(self, folder=HERE, model='xgb'):
        self.rec = json.load(open(os.path.join(folder, 'record.json')))
        self.cfg = json.load(open(os.path.join(folder, 'config.json')))
        self.order = self.rec['features']
        self.model_name = model
        if model == 'xgb':
            from xgboost import XGBClassifier
            self.models = []
            for s in self.rec['xgboost']['seeds']:
                m = XGBClassifier(); m.load_model(os.path.join(folder, 'model', 'xgb_seed%d.json' % s)); self.models.append(m)
        else:
            self.models = [pickle.load(open(os.path.join(folder, 'model', 'rf_backup.pkl'), 'rb'))]
    @property
    def lower(self): return self.cfg['lower']
    @property
    def upper(self): return self.cfg['upper']
    def score_features(self, f):
        x = np.array([[f[c] for c in self.order]], dtype=np.float64)
        return float(np.mean([m.predict_proba(x)[:, 1] for m in self.models]))
    def label_for(self, score):
        return 'non_hit' if score < self.lower else ('hit' if score >= self.upper else 'unsure')
    def classify(self, frame):
        if isinstance(frame, (str, os.PathLike)): frame = crop_from_file(frame)
        s = self.score_features(features(frame))
        return self.label_for(s), s
    def classify_many(self, paths, log_csv=None):
        out = []
        for p in paths:
            t = time.time()
            try: lab, s = self.classify(p); err = ''
            except Exception as e: lab, s, err = 'error', float('nan'), str(e)[:80]
            out.append(dict(frame=str(p), label=lab, score=round(s, 5), seconds=round(time.time() - t, 3), error=err,
                            version=self.rec['version'] + ('' if self.model_name == 'xgb' else '-rf'), lower=self.lower, upper=self.upper,
                            time=datetime.datetime.now().isoformat(timespec='seconds')))
        if log_csv:
            new = not os.path.exists(log_csv)
            with open(log_csv, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(out[0].keys()));
                if new: w.writeheader()
                w.writerows(out)
        return out
