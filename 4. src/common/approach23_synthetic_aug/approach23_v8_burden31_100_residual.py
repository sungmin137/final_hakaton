"""approach23 v8: burden 31-100 구간 잔여 취약 클래스 감사 (approach23/anomaly-scan 마지막 단계).
학습/CV/제출 없음 — 기존 OOF(9차/v2/16차 블렌드) + test 관찰만 재사용.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix as sk_cm

from main import ROOT, DATA, load_train
from features.features import gene_columns, TARGET

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")
genes = gene_columns(train)
burden_tr = (train[genes] != "WT").sum(axis=1).to_numpy()
burden_te = (test[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
true_lab = train[TARGET].to_numpy()

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
oof_9cha = np.load(ROOT / "6. experiments/2026-09-11_v4_xgb_mild_col_grp/oof_proba.npy")
oof_v2 = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/oof_proba.npy")
oof_blend = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
pred_9cha = np.array(classes)[(oof_9cha * s3).argmax(1)]
pred_v2 = np.array(classes)[(oof_v2 * s3).argmax(1)]
pred_blend = np.array(classes)[(oof_blend * s3).argmax(1)]

m31 = (burden_tr >= 31) & (burden_tr <= 100)
mte31 = (burden_te >= 31) & (burden_te <= 100)
print(f"train 31-100: {m31.sum()}행 ({m31.mean():.1%})  test 31-100: {mte31.sum()}행 ({mte31.mean():.1%})")

print("\n" + "=" * 78)
print("1~3. 31-100 구간 class별 F1/precision/recall (9차/v2/16차블렌드), support>=15만")
rows = []
for c in classes:
    na = ((true_lab == c) & m31).sum()
    if na < 15:
        continue
    yb = (true_lab[m31] == c).astype(int)
    r = {"class": c, "n_actual": na}
    for name, pred in [("9cha", pred_9cha), ("v2", pred_v2), ("blend", pred_blend)]:
        pb = (pred[m31] == c).astype(int)
        r[f"recall_{name}"] = round(recall_score(yb, pb), 3)
        r[f"prec_{name}"] = round(precision_score(yb, pb, zero_division=0), 3)
        r[f"f1_{name}"] = round(f1_score(yb, pb, zero_division=0), 3)
    rows.append(r)
df = pd.DataFrame(rows).sort_values("f1_blend")
print(df[["class", "n_actual", "f1_9cha", "f1_v2", "f1_blend"]].to_string(index=False))

print("\n" + "=" * 78)
print("4. v3(blend) 대비 9차 개선/악화 분류 (A: 9차보다 크게 개선 / B: 계속 취약 / C: trade-off)")
df["delta_blend_vs_9cha"] = df["f1_blend"] - df["f1_9cha"]
print(df[["class", "n_actual", "f1_9cha", "f1_blend", "delta_blend_vs_9cha"]].sort_values("delta_blend_vs_9cha").to_string(index=False))

# 잔여 취약 후보: blend F1이 여전히 낮고(<0.5), 9차 대비 개선폭이 작은(<0.05) 클래스
weak = df[(df.f1_blend < 0.5) & (df.delta_blend_vs_9cha.abs() < 0.06)]
print(f"\n잔여 취약 후보(F1<0.5, v3 개선폭<0.06, support>=15): {weak['class'].tolist()}")

print("\n" + "=" * 78)
print("2(train/test relevance). 후보 클래스별 train 전체/31-100 표본, burden 분포, test 예측 비중")
te_pred_share_31 = pd.Series(pred_blend).where(pd.Series(mte31).values).value_counts(normalize=True) if False else None
# test는 라벨이 없으니 16차 test 예측(blend_te) 재사용
p7 = np.load(ROOT / "6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy") / s3
p7 = p7 / p7.sum(1, keepdims=True)
p2 = np.load(ROOT / "6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy")
z = np.log(p7 + 1e-9) * 0.5 + np.log(p2 + 1e-9) * 0.5
blend_te = np.exp(z - z.max(1, keepdims=True)); blend_te /= blend_te.sum(1, keepdims=True)
pred_te17 = np.array(classes)[(blend_te * s3).argmax(1)]
te31_pred_share = pd.Series(pred_te17[mte31]).value_counts(normalize=True)

for c in df["class"]:
    n_all = (true_lab == c).sum()
    n_31 = ((true_lab == c) & m31).sum()
    b = burden_tr[(true_lab == c) & m31]
    te_share = te31_pred_share.get(c, 0.0)
    print(f"{c}: train 전체={n_all}, 31-100 표본={n_31}({n_31/max(n_all,1):.1%}), "
          f"31-100 내 burden mean={b.mean():.1f}, test 31-100 예측비중={te_share:.1%}")

print("\n" + "=" * 78)
print("3(confusion). 잔여 취약 후보의 31-100 구간 최대 confusion edge (blend OOF 기준)")
for c in weak["class"]:
    mask = m31 & (true_lab == c)
    others = pd.Series(pred_blend[mask]).value_counts()
    print(f"{c} (n={mask.sum()}): " + ", ".join(f"{k}={v}" for k, v in others.items() if k != c))

print("\n" + "=" * 78)
print("5. 잔여 취약 후보 vs 최대 confusion 상대 클래스, gene frequency 차이 + test-train shift")
gene_diff_te = pd.read_csv(ROOT / "6. experiments/2026-09-14_approach23_synthetic_gbmlgg_lgg/v3_gene_freq_diff.csv", index_col=0).iloc[:, 0]
is_mut = (train[genes] != "WT")
for c in weak["class"]:
    mask_c = m31 & (true_lab == c)
    if mask_c.sum() < 15:
        continue
    others = pd.Series(pred_blend[mask_c]).value_counts()
    rival = [k for k in others.index if k != c][0] if len(others) > (1 if c in others.index else 0) else None
    if rival is None:
        continue
    mask_r = m31 & (true_lab == rival)
    fc = is_mut[mask_c].mean(); fr = is_mut[mask_r].mean()
    sep = (fc - fr).sort_values(key=np.abs, ascending=False).head(6)
    print(f"\n{c}(n={mask_c.sum()}) vs 최대 오답상대 {rival}(n={mask_r.sum()}) — 구분 유전자 상위 6개 + test-train 전체 shift")
    for g in sep.index:
        print(f"  {g}: {c}={fc[g]:.3f} {rival}={fr[g]:.3f} sep={sep[g]:+.3f}  (test-train diff 전체={gene_diff_te.get(g, float('nan')):+.4f})")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
df.to_csv(out_dir / "v8_burden31_100_residual.csv", index=False)
print(f"\nsaved: {out_dir}")
