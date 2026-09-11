"""KIPAN/KIRC, GBMLGG/LGG 전용 2-클래스 서브모델(클래스 가중치 적용).
`specialist_submodel.py`(v2, 가중치 없음)에 `class_weights()`(`main.py`에 이미 있는 함수)를
`sample_weight`로 추가한 버전. 04_todo_day2.md STEP 4 참고.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, TARGET, FeatureMaker, XGB_PARAMS, SEED, twin_groups, load_train, class_weights

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_row = gid.map(label_sets).apply(lambda s: len(s) > 1)

main_per_class = {"KIRC": 0.2811, "KIPAN": 0.4432, "GBMLGG": 0.5052, "LGG": 0.4348}


def run_specialist(pair):
    sub = train[train[TARGET].isin(pair)].copy()
    le = LabelEncoder()
    y = le.fit_transform(sub[TARGET])
    groups = twin_groups(sub)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(sub), 2))
    for tri, vai in skf.split(sub, y, groups):
        tr_part, va_part = sub.iloc[tri], sub.iloc[vai]
        fm = FeatureMaker("v4").fit(tr_part)
        Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
        w = class_weights(y[tri])
        model = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr, y[tri], sample_weight=w)
        oof[vai] = model.predict_proba(Xva)
    pred = le.inverse_transform(oof.argmax(1))
    sub = sub.reset_index(drop=True)
    sub["pred"] = pred
    sub["is_twin"] = is_twin_row.loc[train[TARGET].isin(pair)].reset_index(drop=True)
    return sub


print("[1/1] 전용 서브모델(가중치 적용) 학습 중...")
summary = {}
for pair in [("KIRC", "KIPAN"), ("GBMLGG", "LGG")]:
    res = run_specialist(pair)
    print(f"\n=== {pair[0]} vs {pair[1]} 전용 모델 (balanced) ===")
    for cls in pair:
        d = res[res[TARGET] == cls]
        twin_acc = (d[d.is_twin].pred == cls).mean() if d.is_twin.any() else float("nan")
        nontwin = d[~d.is_twin]
        nontwin_acc = (nontwin.pred == cls).mean() if len(nontwin) else float("nan")
        print(f"{cls}: 비쌍둥이 정확도 {nontwin_acc*100:.1f}% (n={len(nontwin)}) | "
              f"쌍둥이 정확도 {twin_acc*100:.1f}% | 메인모델 F1(참고) {main_per_class.get(cls)}")
        summary[cls] = dict(nontwin_acc=round(float(nontwin_acc), 4), twin_acc=round(float(twin_acc), 4),
                            n_nontwin=int(len(nontwin)), main_model_f1=main_per_class.get(cls))

out_dir = ROOT / "6. experiments" / "2026-09-10_approach11_v3_specialist_submodel_balanced"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "result.json").write_text(json.dumps(dict(features="v4", weighted=True, per_class=summary),
                                                 indent=2, ensure_ascii=False))
print(f"\nsaved: {out_dir}")
