"""approach23 v18: TP53 burden-conditional enrichment feature를 기존 v4+mild_col 파이프라인에
추가해서 OOF로 검증. 새 모델/서드모델 없음 — 기존 XGB에 컬럼 하나만 추가.
feature = TP53 변이여부(0/1) - E[TP53 변이확률 | burden bin](그 fold의 tr_part로만 추정).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from main import ROOT, SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker
from features.features import gene_columns, TARGET

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
is_tp53 = (train["TP53"] != "WT").to_numpy().astype(float)
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]

OUT = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
OUT.mkdir(parents=True, exist_ok=True)

BINS_EDGES = [0, 1, 4, 8, 16, 31, 101, 397, 10**9]
burden_bin = np.digitize(burden, BINS_EDGES[1:-1])

BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi<10**9 else '+'}" for lo, hi in BINS]


def macro_by_bin(pred_full):
    row = {}
    for (lo, hi), nm in zip(BINS, bin_names):
        m = (burden >= lo) & (burden <= hi)
        row[nm] = round(f1_score(y[m], pred_full[m], average="macro", labels=sorted(set(y[m].tolist()))), 4)
    row["전체"] = round(f1_score(y, pred_full, average="macro", labels=sorted(set(y.tolist()))), 4)
    return row


groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y, groups))

oof_pred = np.full(n, -1)
importances = []
for k, (tri, vai) in enumerate(folds):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)

    # burden bin별 TP53 기대확률: tr_part로만 추정
    expected = pd.Series(is_tp53[tri]).groupby(burden_bin[tri]).mean()
    exp_tr = pd.Series(burden_bin[tri]).map(expected).to_numpy()
    exp_va = pd.Series(burden_bin[vai]).map(expected).fillna(expected.mean()).to_numpy()
    Xtr = Xtr.copy(); Xva = Xva.copy()
    Xtr["tp53_burden_enrich"] = is_tp53[tri] - exp_tr
    Xva["tp53_burden_enrich"] = is_tp53[vai] - exp_va

    model = xgb.XGBClassifier(**PARAMS).fit(Xtr, y[tri])
    oof_pred[vai] = model.predict(Xva)
    fi = dict(zip(Xtr.columns, model.feature_importances_))
    importances.append(fi.get("tp53_burden_enrich", 0.0))
    rank = sorted(fi.items(), key=lambda kv: -kv[1])
    rank_pos = [i for i, (nm, _) in enumerate(rank) if nm == "tp53_burden_enrich"][0]
    print(f"fold{k}: tp53_burden_enrich importance={fi.get('tp53_burden_enrich',0):.5f} "
          f"(전체 {len(fi)}개 피처 중 {rank_pos+1}위)", flush=True)

np.save(OUT / "oof_pred_tp53norm.npy", oof_pred)
res = macro_by_bin(oof_pred)
print("\n결과 (baseline_9cha와 비교는 exp1 summary 참고):")
print(pd.Series(res))
print(f"\ntp53_burden_enrich 평균 importance: {np.mean(importances):.5f}")

pc = f1_score(y, oof_pred, average=None, labels=range(len(classes)))
pd.Series(pc, index=classes).to_csv(OUT / "per_class_f1_tp53norm.csv")
pd.Series(res).to_csv(OUT / "exp2_summary.csv")
print(f"\nsaved: {OUT}")
