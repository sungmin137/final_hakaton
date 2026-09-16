"""approach23 v23: v4s(v4+치환스펙트럼) 기준선에 BLOSUM62 분포 요약(표준편차, 보수적 치환 비율)을 추가해
정직 OOF(group-twins)로 검증. 60유전자 제한 없이 전체 missense 대상. test 미사용.
기존 KnowledgeFeatures의 kf_bl_mean(평균)·kf_severe_ratio(심각 비율)와 겹치는 항목은 제외 (중복 감사 완료).
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
from approach23_synthetic_aug.blosum_dist_features import blosum_dist_features

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]

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

conditions = ["v4s_base", "v4s_plus_blosum_dist"]
oof = {c: np.full(n, -1) for c in conditions}
importances = []

for k, (tri, vai) in enumerate(folds):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr_v4, Xva_v4 = fm.transform(tr_part), fm.transform(va_part)

    sp_tr = spectrum_features(tr_part, genes).reset_index(drop=True)
    sp_va = spectrum_features(va_part, genes).reset_index(drop=True)
    Xtr_base = pd.concat([Xtr_v4.reset_index(drop=True), sp_tr], axis=1)
    Xva_base = pd.concat([Xva_v4.reset_index(drop=True), sp_va], axis=1)

    m = xgb.XGBClassifier(**PARAMS).fit(Xtr_base, y[tri])
    oof["v4s_base"][vai] = m.predict(Xva_base)

    bd_tr = blosum_dist_features(tr_part, genes).reset_index(drop=True)
    bd_va = blosum_dist_features(va_part, genes).reset_index(drop=True)
    Xtr_bd = pd.concat([Xtr_base, bd_tr], axis=1)
    Xva_bd = pd.concat([Xva_base, bd_va], axis=1)
    m2 = xgb.XGBClassifier(**PARAMS).fit(Xtr_bd, y[tri])
    oof["v4s_plus_blosum_dist"][vai] = m2.predict(Xva_bd)

    imp = pd.Series(m2.feature_importances_, index=Xtr_bd.columns)
    importances.append(imp[["bld_std", "bld_conservative_ratio"]])

    print(f"fold{k} 완료 (BLOSUM분포 피처 {bd_tr.shape[1]}열)", flush=True)

print("\n" + "=" * 78)
print("결과 비교 (정직 OOF, group-twins)")
tbl = pd.DataFrame([macro_by_bin(oof[c]) for c in conditions], index=conditions)
print(tbl.to_string())

f1_base = f1_score(y, oof["v4s_base"], average=None, labels=range(len(classes)))
f1_bd = f1_score(y, oof["v4s_plus_blosum_dist"], average=None, labels=range(len(classes)))
diff = pd.Series(f1_bd - f1_base, index=classes).sort_values()
print("\nclass별 변화 최악5 / 최선5:")
print("  최악:", ", ".join(f"{cl}({diff[cl]:+.3f})" for cl in diff.head(5).index))
print("  최선:", ", ".join(f"{cl}({diff[cl]:+.3f})" for cl in diff.tail(5).index))
changed = oof["v4s_plus_blosum_dist"] != oof["v4s_base"]
print(f"예측 바뀐 행: {changed.sum()} ({changed.mean():.1%})")

print("\nfeature importance (bld_std, bld_conservative_ratio), fold 평균:")
imp_mean = pd.concat(importances, axis=1).mean(axis=1)
print(imp_mean.to_string())

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
tbl.to_csv(out_dir / "v23_blosum_distribution_summary.csv")
diff.to_csv(out_dir / "v23_blosum_distribution_class_diff.csv")
print(f"\nsaved: {out_dir}")
