"""Version 1.2 = Version 1.1 (26 Oct decisions applied) + the 27 Oct decision sheet (Label_decision_sheet_27Oct.xlsx).
Sheet result: all 38 frames 'newer'. 'newer' label = the later of Scarlett's two judgements (D rows: fresh blind labels in test 03; C rows: test 02 labels).
C10 pixium_146649: 'newer' (hit) contradicted her 26 Oct audit (non_hit, A07); on 27 Oct 11:32 Scarlett confirmed the newer labels stand, so it is now hit. No frames held back.
Usage: python train_v1_2.py <training_frames_v1.csv> <sheet.xlsx> <out_dir>"""
import sys, json, hashlib, datetime, platform, glob
import numpy as np, pandas as pd, openpyxl, xgboost, sklearn
sys.argv_saved = sys.argv; src, sheet, out = sys.argv[1:4]
sys.argv = ['x', src, '', out]
import train_v1_1 as T
d = T.d.copy()
ws = openpyxl.load_workbook(sheet).active
rows = [r for r in ws.iter_rows(min_row=4, values_only=True) if r[0] and len(str(r[0])) == 3 and str(r[0])[0] in 'DC']
HOLD = set()  # C10 pixium_146649 released 27 Oct 11:32: Scarlett confirmed newer labels stand (hit)
changed = []
for r in rows:
    stem, newer, dec = r[2], r[5], str(r[9]).strip().lower()
    if stem in HOLD or dec != 'newer': continue
    m = d.stem == stem
    cur = d.loc[m, 'label_final'].unique().tolist()
    if newer not in cur or len(cur) != 1:
        d.loc[m, 'label_final'] = newer; d.loc[m, 'y'] = int(newer == 'hit'); d.loc[m, 'label_source'] = 'manual_decision_27Oct26'
        changed.append((r[0], stem, cur, newer, int(m.sum())))
pos, neg = int(d.y.sum()), int((1 - d.y).sum())
if __name__ == '__main__':
    import os; os.makedirs(out + '/model', exist_ok=True); os.makedirs(out + '/data', exist_ok=True)
    for s in range(5): T.fit(d, s).save_model(f'{out}/model/xgb_seed{s}.json')
    d.drop(columns='stem').to_csv(f'{out}/data/training_frames_v1_2.csv', index=False)
    md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
    rec = dict(version='1.2', trained=datetime.datetime.now().isoformat(timespec='seconds'), base='V1.1 (26 Oct decisions) + 27 Oct decision sheet',
        frames_label_changed_27oct=[(a, b, c, e) for a, b, c, e, f in changed], rows_label_changed=int(sum(f for *_, f in changed)),
        held_back=sorted(HOLD), n_frames=len(d), n_hits=pos, n_non_hits=neg, scale_pos_weight=round(neg / pos, 4), features=T.FE,
        xgboost=dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seeds=[0,1,2,3,4], combine='mean of 5 seeds'),
        cutoffs=dict(lower=0.02, upper=0.90, note='unchanged'), libraries=dict(python=platform.python_version(), xgboost=xgboost.__version__, sklearn=sklearn.__version__),
        input_md5={'training_frames_v1.csv': md5(src), 'Label_decision_sheet_27Oct.xlsx': md5(sheet)},
        model_md5={f'xgb_seed{s}.json': md5(f'{out}/model/xgb_seed{s}.json') for s in range(5)})
    json.dump(rec, open(f'{out}/record_v1_2.json', 'w'), indent=1, default=str)
    print(len(changed), 'frames changed;', rec['rows_label_changed'], 'rows'); [print(c) for c in changed]; print(len(d), pos, neg)
