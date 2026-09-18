"""approach23 v15: burden 31-100 전용 nearest-neighbor 분류기 — OOF로 기존 모델(16차 블렌드) 보완 여부 검증.
CV 재실행/새 XGB 학습 없음 — train 31-100 profile만으로 만든 가벼운 similarity-vote 분류기를
main.py와 동일한 group-twins fold(SEED=42)로 out-of-fold 평가한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from main import ROOT, DATA, SEED, twin_groups, load_train
from features.features import gene_columns, TARGET

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")
genes = gene_columns(train)
is_mut_tr = (train[genes] != "WT")
is_mut_te = (test[genes] != "WT")
burden_tr = is_mut_tr.sum(axis=1).to_numpy()
burden_te = is_mut_te.sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n_classes = len(classes)

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
blend_oof = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
probaA_scaled = blend_oof * s3; probaA_scaled /= probaA_scaled.sum(1, keepdims=True)  # Model A (기존 16차 OOF, 26-class)

m31 = (burden_tr >= 31) & (burden_tr <= 100)
BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi<10**9 else '+'}" for lo, hi in BINS]

# ================================================================ Model B: burden 31-100 전용 NN 분류기 (fold별 OOF)
groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
probaB_31 = np.zeros((m31.sum(), n_classes))   # 31-100 train행에 대해서만 채움
idx31 = np.flatnonzero(m31)
idx31_pos = {gi: pi for pi, gi in enumerate(idx31)}
K = 5

X_all = is_mut_tr[genes].to_numpy(dtype=np.float32)
for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
    tri_set = set(tri.tolist())
    pool_idx = np.array([i for i in idx31 if i in tri_set])       # 이 fold의 학습쪽 31-100 행만 (도너 풀)
    val_idx = np.array([i for i in idx31 if i not in tri_set])    # 이 fold의 검증쪽 31-100 행
    if len(pool_idx) == 0 or len(val_idx) == 0:
        continue
    Xp, Xv = X_all[pool_idx], X_all[val_idx]
    inter = Xv @ Xp.T
    sp, sv = Xp.sum(1), Xv.sum(1)
    union = sv[:, None] + sp[None, :] - inter
    jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    yk = y[pool_idx]
    topk = np.argsort(-jacc, axis=1)[:, :K]
    for r, vi in enumerate(val_idx):
        neigh = topk[r]
        w = jacc[r, neigh]
        if w.sum() == 0:
            w = np.ones_like(w)
        proba = np.zeros(n_classes)
        for wj, cj in zip(w, yk[neigh]):
            proba[cj] += wj
        proba /= proba.sum()
        probaB_31[idx31_pos[vi]] = proba
    print(f"fold{k}: pool={len(pool_idx)} val={len(val_idx)} 완료", flush=True)

predA_31 = probaA_scaled[idx31].argmax(1)
predB_31 = probaB_31.argmax(1)
y31 = y[idx31]

print("\n" + "=" * 78)
print("Model A(기존 16차) vs Model B(31-100 NN) — 31-100 OOF 비교")
accA = (predA_31 == y31).mean(); accB = (predB_31 == y31).mean()
f1A = f1_score(y31, predA_31, average="macro", labels=sorted(set(y31)))
f1B = f1_score(y31, predB_31, average="macro", labels=sorted(set(y31)))
print(f"Model A: acc={accA:.3f} macroF1={f1A:.3f}")
print(f"Model B: acc={accB:.3f} macroF1={f1B:.3f}")
agree = (predA_31 == predB_31).mean()
print(f"A/B agreement: {agree:.3f}")

a_wrong_b_right = (predA_31 != y31) & (predB_31 == y31)
a_right_b_wrong = (predA_31 == y31) & (predB_31 != y31)
print(f"A 틀림 & B 맞음(보완 가능): {a_wrong_b_right.sum()}행")
print(f"A 맞음 & B 틀림(훼손 위험): {a_right_b_wrong.sum()}행")

print("\n" + "=" * 78)
print("alpha 블렌드 스윕 (31-100 행에서만 A/B 확률 블렌드, 다른 구간은 A 그대로) — burden 구간별 macro F1")
final_proba_full = probaA_scaled.copy()
results = []
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    fp = probaA_scaled.copy()
    blended_31 = (1 - alpha) * probaA_scaled[idx31] + alpha * probaB_31
    fp[idx31] = blended_31
    pred_full = fp.argmax(1)
    row = {"alpha": alpha}
    for (lo, hi), name in zip(BINS, bin_names):
        mb = (burden_tr >= lo) & (burden_tr <= hi)
        yt, yp = y[mb], pred_full[mb]
        row[name] = round(f1_score(yt, yp, average="macro", labels=sorted(set(yt.tolist()))), 4)
    row["전체"] = round(f1_score(y, pred_full, average="macro", labels=sorted(set(y.tolist()))), 4)
    results.append(row)
res_df = pd.DataFrame(results)
cols = ["alpha", "전체"] + bin_names
print(res_df[cols].to_string(index=False))

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
res_df.to_csv(out_dir / "v15_nn31_100_alpha_sweep.csv", index=False)
np.save(out_dir / "v15_probaB_31_oof.npy", probaB_31)
print(f"\nsaved: {out_dir}")
