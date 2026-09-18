"""approach23 v24: v4s(v4+치환스펙트럼) 기준선에 consequence 조합 구조(csq_n_types, csq_all4_present)
2개만 추가해 정직 OOF(group-twins)로 검증. test 미사용. LoF비중(spt_trunc와 중복)은 제외.
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
from approach23_synthetic_aug.consequence_composition_features import consequence_composition_features

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

conditions = ["v4s_base", "v4s_plus_csq_composition"]
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

    csq_tr = consequence_composition_features(tr_part, genes).reset_index(drop=True)
    csq_va = consequence_composition_features(va_part, genes).reset_index(drop=True)
    Xtr_csq = pd.concat([Xtr_base, csq_tr], axis=1)
    Xva_csq = pd.concat([Xva_base, csq_va], axis=1)
    m2 = xgb.XGBClassifier(**PARAMS).fit(Xtr_csq, y[tri])
    oof["v4s_plus_csq_composition"][vai] = m2.predict(Xva_csq)

    imp = pd.Series(m2.feature_importances_, index=Xtr_csq.columns)
    importances.append(imp[["csq_n_types", "csq_all4_present"]])

    print(f"fold{k} 완료 (csq 피처 {csq_tr.shape[1]}열)", flush=True)

print("\n" + "=" * 78)
print("결과 비교 (정직 OOF, group-twins)")
tbl = pd.DataFrame([macro_by_bin(oof[c]) for c in conditions], index=conditions)
print(tbl.to_string())

f1_base = f1_score(y, oof["v4s_base"], average=None, labels=range(len(classes)))
f1_csq = f1_score(y, oof["v4s_plus_csq_composition"], average=None, labels=range(len(classes)))
diff = pd.Series(f1_csq - f1_base, index=classes).sort_values()
print("\nclass별 변화 최악5 / 최선5:")
print("  최악:", ", ".join(f"{cl}({diff[cl]:+.3f})" for cl in diff.head(5).index))
print("  최선:", ", ".join(f"{cl}({diff[cl]:+.3f})" for cl in diff.tail(5).index))
changed = oof["v4s_plus_csq_composition"] != oof["v4s_base"]
print(f"예측 바뀐 행: {changed.sum()} ({changed.mean():.1%})")

print("\nfeature importance (csq_n_types, csq_all4_present), fold 평균:")
imp_mean = pd.concat(importances, axis=1).mean(axis=1)
print(imp_mean.to_string())

print("\ncsq_n_types 분포(train 전체):")
full_csq = consequence_composition_features(train, genes)
print(full_csq["csq_n_types"].value_counts().sort_index().to_string())

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
tbl.to_csv(out_dir / "v24_consequence_composition_summary.csv")
diff.to_csv(out_dir / "v24_consequence_composition_class_diff.csv")
print(f"\nsaved: {out_dir}")
