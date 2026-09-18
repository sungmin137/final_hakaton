"""approach23 v22: pathway(LoF/missense 분리) feature의 half-cross validation.
기존 5-fold(SEED=42, twin_groups) 분할을 그대로 재사용해 even folds{0,2,4} <-> odd folds{1,3}
양방향으로 학습/평가. 새 feature 추가·튜닝 없음, test 미사용. 재현성만 확인.
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

from main import ROOT, SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker, TARGET
from features.features import gene_columns
from approach23_synthetic_aug.spectrum_ref import spectrum_features
from approach23_synthetic_aug.pathway_module_features import pathway_module_features

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]

BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi<10**9 else '+'}" for lo, hi in BINS]


def macro_by_bin(y_true, pred, mask_all=None):
    row = {}
    for (lo, hi), nm in zip(BINS, bin_names):
        m = (burden >= lo) & (burden <= hi)
        if mask_all is not None:
            m = m & mask_all
        if m.sum() == 0:
            row[nm] = float("nan"); continue
        row[nm] = round(f1_score(y_true[m], pred[m], average="macro", labels=sorted(set(y_true[m].tolist()))), 4)
    m = mask_all if mask_all is not None else np.ones(len(y_true), dtype=bool)
    row["전체"] = round(f1_score(y_true[m], pred[m], average="macro", labels=sorted(set(y_true[m].tolist()))), 4)
    return row


groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_of = np.full(n, -1)
for k, (_, vai) in enumerate(skf.split(train, y, groups)):
    fold_of[vai] = k

EVEN = {0, 2, 4}; ODD = {1, 3}


def build_X(tr_part, va_part):
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr_v4, Xva_v4 = fm.transform(tr_part), fm.transform(va_part)
    sp_tr = spectrum_features(tr_part, genes).reset_index(drop=True)
    sp_va = spectrum_features(va_part, genes).reset_index(drop=True)
    Xtr_base = pd.concat([Xtr_v4.reset_index(drop=True), sp_tr], axis=1)
    Xva_base = pd.concat([Xva_v4.reset_index(drop=True), sp_va], axis=1)
    pw_tr = pathway_module_features(tr_part, genes).reset_index(drop=True)
    pw_va = pathway_module_features(va_part, genes).reset_index(drop=True)
    Xtr_pw = pd.concat([Xtr_base, pw_tr], axis=1)
    Xva_pw = pd.concat([Xva_base, pw_va], axis=1)
    return Xtr_base, Xva_base, Xtr_pw, Xva_pw


results = {}
for name, train_folds, eval_folds in [("A_even_train_odd_eval", EVEN, ODD), ("B_odd_train_even_eval", ODD, EVEN)]:
    tri = np.flatnonzero(np.isin(fold_of, list(train_folds)))
    vai = np.flatnonzero(np.isin(fold_of, list(eval_folds)))
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    Xtr_base, Xva_base, Xtr_pw, Xva_pw = build_X(tr_part, va_part)

    m_base = xgb.XGBClassifier(**PARAMS).fit(Xtr_base, y[tri])
    pred_base = m_base.predict(Xva_base)
    m_pw = xgb.XGBClassifier(**PARAMS).fit(Xtr_pw, y[tri])
    pred_pw = m_pw.predict(Xva_pw)

    y_va = y[vai]
    burden_va_mask = np.zeros(n, dtype=bool); burden_va_mask[vai] = True
    # macro_by_bin은 전체 y/burden 배열 기준이라 vai만 골라 별도 계산
    def bin_metrics(pred_arr):
        full_pred = np.full(n, -1); full_pred[vai] = pred_arr
        return macro_by_bin(y, full_pred, mask_all=burden_va_mask)

    res_base = bin_metrics(pred_base)
    res_pw = bin_metrics(pred_pw)
    results[name] = dict(base=res_base, pw=res_pw, pred_base=pred_base, pred_pw=pred_pw, vai=vai)
    print(f"{name}: base_전체={res_base['전체']} pw_전체={res_pw['전체']} Δ={res_pw['전체']-res_base['전체']:+.4f}", flush=True)

print("\n" + "=" * 78)
print("구간별 비교")
rows = []
for name in results:
    b, p = results[name]["base"], results[name]["pw"]
    for nm in bin_names + ["전체"]:
        rows.append(dict(direction=name, bin=nm, base=b[nm], pathway=p[nm], delta=round(p[nm]-b[nm], 4)))
df = pd.DataFrame(rows)
print(df.pivot(index="bin", columns="direction", values="delta").to_string())

print("\n" + "=" * 78)
print("클래스별 변화 (양방향)")
for name in results:
    vai = results[name]["vai"]
    y_va = y[vai]
    f1_base = f1_score(y_va, results[name]["pred_base"], average=None, labels=range(len(classes)))
    f1_pw = f1_score(y_va, results[name]["pred_pw"], average=None, labels=range(len(classes)))
    diff = pd.Series(f1_pw - f1_base, index=classes).sort_values()
    print(f"\n{name} 최악5/최선5:")
    print("  최악:", ", ".join(f"{c}({diff[c]:+.3f})" for c in diff.head(5).index))
    print("  최선:", ", ".join(f"{c}({diff[c]:+.3f})" for c in diff.tail(5).index))

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
df.to_csv(out_dir / "v22_pathway_halfcross.csv", index=False)
print(f"\nsaved: {out_dir}")
