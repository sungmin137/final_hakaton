"""0~10 mutation 구간 전용 오답 구조 진단. 재학습 없음 — 기존 16차 블렌드 OOF(blend_w05_oof.npy)만 사용.
confusion, 클래스별 recall/precision/F1, 최대 오답 이동 방향, 세부 구간(0/1-3/4-6/7-10)별 재확인.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "4. src/common"))

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from main import ROOT, load_train
from features.features import gene_columns

D_PREV = ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp"
SCALE_PATH = ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"

train = load_train()
genes = gene_columns(train)
n_mut = (train[genes] != "WT").sum(1).to_numpy()
le = LabelEncoder().fit(train["SUBCLASS"])
y = le.transform(train["SUBCLASS"])
classes = list(le.classes_)
s3 = np.array([json.load(open(SCALE_PATH))[c] for c in classes])

blend = np.load(D_PREV / "blend_w05_oof.npy")
pred_all = (blend * s3).argmax(1)


def report(mask, label):
    n = mask.sum()
    yt, yp = y[mask], pred_all[mask]
    print(f"\n{'='*70}\n{label} (n={n})\n{'='*70}")

    cm = confusion_matrix(yt, yp, labels=range(len(classes)))
    # 클래스별 precision/recall/F1 (0으로 나누기 방지)
    rows = []
    for i, c in enumerate(classes):
        n_actual = (yt == i).sum()
        n_pred = (yp == i).sum()
        tp = cm[i, i]
        recall = tp / n_actual if n_actual > 0 else np.nan
        precision = tp / n_pred if n_pred > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall and not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0) else np.nan
        rows.append(dict(class_=c, n_actual=n_actual, n_pred=n_pred, recall=recall, precision=precision, f1=f1))
    df = pd.DataFrame(rows)
    df_show = df[df.n_actual > 0].sort_values("f1")
    print(f"\n[클래스별 recall/precision/F1] (F1 낮은 순, n_actual>0만)")
    print(df_show.to_string(index=False, formatters={"recall": "{:.2f}".format, "precision": "{:.2f}".format, "f1": "{:.2f}".format}))

    # 최대 오답 이동 방향 (off-diagonal top 15)
    print(f"\n[최대 오답 이동 방향 top 15]")
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], cm[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for a, b, c in pairs[:15]:
        print(f"  {a:8s} -> {b:8s} : {c}건")

    return df


df_all = report(n_mut <= 10, "0~10 구간 전체")

for lo, hi, name in [(0, 0, "0개(무변이)"), (1, 3, "1~3개"), (4, 6, "4~6개"), (7, 10, "7~10개")]:
    mask = (n_mut >= lo) & (n_mut <= hi)
    report(mask, f"{name}")
