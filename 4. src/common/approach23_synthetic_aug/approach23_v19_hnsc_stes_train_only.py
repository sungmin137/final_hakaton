"""approach23 v19: HNSC/STES 서브모델(혜림님 원안, approach19)을 v7("9차") 기준선 위에서
train OOF confusion(60건, test/LB 근거 아님)으로만 재정당화하여 정직 OOF로 재검증.
방법(LR, top-40 유전자, 0.70 임계값)은 원안(approach14_cw_plus/blend.py hnsc_stes_router) 그대로 재사용.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import xgboost as xgb

from main import ROOT, SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker, TARGET, ID

train = load_train()
genes = [c for c in train.columns if c not in (ID, TARGET)]
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]
PAIR = ("HNSC", "STES")
a_id, b_id = le.transform(list(PAIR))


def hnsc_stes_router_fit_apply(tr_part, va_pred_id, va_genes_mat, C=0.1, confidence=0.7, top_k=40):
    """approach14_cw_plus/blend.py의 hnsc_stes_router와 동일 로직, train(tr_part)으로만 fit."""
    ytr = le.transform(tr_part[TARGET])
    Xtr = (tr_part[genes].to_numpy() != "WT").astype(np.int8)
    ptr = np.flatnonzero(np.isin(ytr, (a_id, b_id)))
    top = np.argsort(np.abs(Xtr[ptr][ytr[ptr] == a_id].mean(0) - Xtr[ptr][ytr[ptr] == b_id].mean(0)))[-top_k:]
    ref = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED).fit(
        Xtr[ptr][:, top], ytr[ptr])
    p_ref = ref.predict_proba(va_genes_mat[:, top])
    use = np.isin(va_pred_id, (a_id, b_id)) & (p_ref.max(1) >= confidence)
    out = va_pred_id.copy()
    out[use] = ref.classes_[p_ref.argmax(1)][use]
    return out, use


groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y, groups))

oof_main = np.full(n, -1)
oof_routed = np.full(n, -1)
route_hits = []

for k, (tri, vai) in enumerate(folds):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
    model = xgb.XGBClassifier(**PARAMS).fit(Xtr, y[tri])
    main_pred = model.predict(Xva)
    oof_main[vai] = main_pred

    Xva_genes = (va_part[genes].to_numpy() != "WT").astype(np.int8)
    routed, use = hnsc_stes_router_fit_apply(tr_part, main_pred, Xva_genes)
    oof_routed[vai] = routed
    route_hits.append(use.sum())
    print(f"fold{k}: 라우팅 후보(main pred HNSC/STES) 중 확신>=0.7로 개입한 행 {use.sum()}개 / "
          f"바뀐 행 {(routed!=main_pred).sum()}개", flush=True)

BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi<10**9 else '+'}" for lo, hi in BINS]
burden = (train[genes] != "WT").sum(axis=1).to_numpy()


def macro_by_bin(pred_full):
    row = {}
    for (lo, hi), nm in zip(BINS, bin_names):
        m = (burden >= lo) & (burden <= hi)
        row[nm] = round(f1_score(y[m], pred_full[m], average="macro", labels=sorted(set(y[m].tolist()))), 4)
    row["전체"] = round(f1_score(y, pred_full, average="macro", labels=sorted(set(y.tolist()))), 4)
    return row


print("\n" + "=" * 78)
print("결과: v7(main, features v4+mild_col) vs +HNSC/STES 라우팅(train-only 재정당화)")
tbl = pd.DataFrame([macro_by_bin(oof_main), macro_by_bin(oof_routed)], index=["v7_main", "v7+HNSC_STES_router"])
print(tbl.to_string())

changed = oof_routed != oof_main
n_changed = changed.sum()
improve = changed & (oof_main != y) & (oof_routed == y)
harm = changed & (oof_main == y) & (oof_routed != y)
print(f"\n총 변경 행: {n_changed} (fold별 개입 총합 {sum(route_hits)})")
print(f"개선(main틀림->routed맞음): {improve.sum()}")
print(f"훼손(main맞음->routed틀림): {harm.sum()}")

f1_main = f1_score(y, oof_main, average=None, labels=range(len(classes)))
f1_routed = f1_score(y, oof_routed, average=None, labels=range(len(classes)))
diff = pd.Series(f1_routed - f1_main, index=classes)
print("\nHNSC/STES 자체 F1 변화:")
print(f"  HNSC: {f1_main[a_id]:.4f} -> {f1_routed[a_id]:.4f} ({diff['HNSC']:+.4f})")
print(f"  STES: {f1_main[b_id]:.4f} -> {f1_routed[b_id]:.4f} ({diff['STES']:+.4f})")
print("\n다른 클래스 중 영향 큰 것(상위 5개):")
other = diff.drop(["HNSC", "STES"]).sort_values(key=np.abs, ascending=False)
print(other.head(5).to_string())

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
tbl.to_csv(out_dir / "v19_hnsc_stes_router_summary.csv")
print(f"\nsaved: {out_dir}")
