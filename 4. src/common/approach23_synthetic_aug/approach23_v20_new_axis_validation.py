"""approach23 v20: v4s(v4+치환스펙트럼) 기준선에 신규 정보축 후보 D(스펙트럼 엔트로피/집중도),
E(hotspot 연속거리)를 추가해서 정직 OOF(group-twins)로 비교. test 미사용, 새 모델/라우팅 없음.
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
from approach23_synthetic_aug.new_axis_features import spectrum_concentration_features, HotspotDistance

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

conditions = ["v4s_base", "v4s_plus_D_entropy", "v4s_plus_E_hotspotdist"]
oof = {c: np.full(n, -1) for c in conditions}

for k, (tri, vai) in enumerate(folds):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr_v4, Xva_v4 = fm.transform(tr_part), fm.transform(va_part)

    sp_tr = spectrum_features(tr_part, genes)
    sp_va = spectrum_features(va_part, genes)
    Xtr_base = pd.concat([Xtr_v4.reset_index(drop=True), sp_tr.reset_index(drop=True)], axis=1)
    Xva_base = pd.concat([Xva_v4.reset_index(drop=True), sp_va.reset_index(drop=True)], axis=1)

    m = xgb.XGBClassifier(**PARAMS).fit(Xtr_base, y[tri])
    oof["v4s_base"][vai] = m.predict(Xva_base)

    concD_tr = spectrum_concentration_features(sp_tr).reset_index(drop=True)
    concD_va = spectrum_concentration_features(sp_va).reset_index(drop=True)
    Xtr_D = pd.concat([Xtr_base, concD_tr], axis=1)
    Xva_D = pd.concat([Xva_base, concD_va], axis=1)
    mD = xgb.XGBClassifier(**PARAMS).fit(Xtr_D, y[tri])
    oof["v4s_plus_D_entropy"][vai] = mD.predict(Xva_D)

    hs = HotspotDistance(min_carriers=3, near_threshold=10).fit(tr_part, genes)
    hsE_tr = hs.transform(tr_part, genes).reset_index(drop=True)
    hsE_va = hs.transform(va_part, genes).reset_index(drop=True)
    Xtr_E = pd.concat([Xtr_base, hsE_tr], axis=1)
    Xva_E = pd.concat([Xva_base, hsE_va], axis=1)
    mE = xgb.XGBClassifier(**PARAMS).fit(Xtr_E, y[tri])
    oof["v4s_plus_E_hotspotdist"][vai] = mE.predict(Xva_E)

    print(f"fold{k} 완료 (hotspot 보유 유전자 수: {len(hs.hotspots)})", flush=True)

print("\n" + "=" * 78)
print("결과 비교 (정직 OOF, group-twins)")
tbl = pd.DataFrame([macro_by_bin(oof[c]) for c in conditions], index=conditions)
print(tbl.to_string())

for c in conditions[1:]:
    f1_base = f1_score(y, oof["v4s_base"], average=None, labels=range(len(classes)))
    f1_c = f1_score(y, oof[c], average=None, labels=range(len(classes)))
    diff = pd.Series(f1_c - f1_base, index=classes).sort_values()
    print(f"\n{c} vs base, class별 변화 최악5 / 최선5:")
    print("  최악:", ", ".join(f"{cl}({diff[cl]:+.3f})" for cl in diff.head(5).index))
    print("  최선:", ", ".join(f"{cl}({diff[cl]:+.3f})" for cl in diff.tail(5).index))
    changed = oof[c] != oof["v4s_base"]
    print(f"  전체 대비 예측 바뀐 행: {changed.sum()} ({changed.mean():.1%})")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
tbl.to_csv(out_dir / "v20_new_axis_summary.csv")
print(f"\nsaved: {out_dir}")
