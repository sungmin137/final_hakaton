"""approach23 v7: burden=0(무변이) 94행에서 THYM prior/sink 여부 확인. 재학습 없음 — 기존 group-twins
OOF만 사용. THYM을 제외하면 나머지 클래스가 서로 구분되는지, 아니면 THYM을 빼도 여전히 붕괴하는지 확인.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix as sk_cm

from main import ROOT, load_train
from features.features import gene_columns, TARGET

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
m0 = burden == 0
print(f"burden=0 행 수: {m0.sum()}")

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
blend_oof = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
pred_blend = (blend_oof * s3).argmax(1)
pred_blend_lab = np.array(classes)[pred_blend]
true_lab = train[TARGET].to_numpy()

print("\n" + "=" * 78)
print("1. burden=0 전체 confusion matrix (16차 블렌드 OOF)")
sub_true, sub_pred = true_lab[m0], pred_blend_lab[m0]
present = sorted(set(sub_true.tolist()) | set(sub_pred.tolist()))
cm = pd.DataFrame(sk_cm(sub_true, sub_pred, labels=present), index=present, columns=present)
print(cm.to_string())

print("\n" + "=" * 78)
print("2. THYM 정답 -> 다른 클래스 오분류")
thym_true = m0 & (true_lab == "THYM")
print(f"THYM 실제(n={thym_true.sum()}): " + pd.Series(pred_blend_lab[thym_true]).value_counts().to_string())

print("\n3. 다른 클래스 정답 -> THYM 오분류")
other_to_thym = m0 & (true_lab != "THYM") & (pred_blend_lab == "THYM")
print(f"THYM으로 잘못 예측된 행(n={other_to_thym.sum()}): " + pd.Series(true_lab[other_to_thym]).value_counts().to_string())

print("\n" + "=" * 78)
print("4~5. THYM 제외 전/후 macro F1, class별 support/F1 (burden=0, 16차 블렌드)")
mask_noThym_true = m0 & (true_lab != "THYM")
print(f"THYM 포함 burden=0 macro F1(present labels): "
      f"{f1_score(sub_true, sub_pred, average='macro', labels=present):.4f} (n={m0.sum()})")
present_noThym = sorted(set(true_lab[mask_noThym_true].tolist()))
f1_noThym = f1_score(true_lab[mask_noThym_true], pred_blend_lab[mask_noThym_true], average="macro", labels=present_noThym)
print(f"THYM '실제 라벨'만 제외(THYM으로 예측되는 건 그대로 오답 처리) macro F1: {f1_noThym:.4f} (n={mask_noThym_true.sum()})")

per_class = []
for c in present:
    nt = ((true_lab == c) & m0).sum()
    if nt == 0:
        continue
    yb = (true_lab[m0] == c).astype(int); pb = (pred_blend_lab[m0] == c).astype(int)
    f1c = f1_score(yb, pb, zero_division=0)
    per_class.append((c, nt, f1c))
pc_df = pd.DataFrame(per_class, columns=["class", "n_actual(burden=0)", "f1"]).sort_values("f1")
print("\nclass별 support/F1 (burden=0):")
print(pc_df.to_string(index=False))

print("\n" + "=" * 78)
print("6. 여러 실험에서 동일 현상 반복 확인 (group_twins 저장 OOF 전부)")
rows = []
for rp in sorted(glob.glob(str(ROOT / "6. experiments" / "**" / "result.json"), recursive=True)):
    d = json.load(open(rp))
    if not d.get("group_twins"):
        continue
    npy = Path(rp).parent / "oof_proba.npy"
    if not npy.exists():
        continue
    proba = np.load(npy)
    if proba.shape[0] != len(train):
        continue
    if "khs_tp53_local" in rp:
        continue
    pred = proba.argmax(1)
    pred_lab = np.array(classes)[pred]
    f_with = f1_score(true_lab[m0], pred_lab[m0], average="macro", labels=sorted(set(true_lab[m0])))
    f_without = f1_score(true_lab[mask_noThym_true], pred_lab[mask_noThym_true], average="macro",
                          labels=sorted(set(true_lab[mask_noThym_true])))
    rows.append((Path(rp).parent.name, d.get("features"), round(f_with, 4), round(f_without, 4), round(f_without - f_with, 4)))
rep_df = pd.DataFrame(rows, columns=["experiment", "features", "burden0_with_THYM", "burden0_without_THYM_true", "delta"])
print(rep_df.to_string(index=False))
print(f"\n델타 평균: {rep_df['delta'].mean():.4f}  표준편차: {rep_df['delta'].std():.4f}  "
      f"(양수면 THYM 제외 후 나머지 클래스 구분력이 살아난다는 뜻)")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
rep_df.to_csv(out_dir / "v7_burden0_thym_repetition.csv", index=False)
print(f"\nsaved: {out_dir}")
