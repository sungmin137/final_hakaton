"""approach23 v5: anomaly 후보 중 실제 test 레버리지가 큰 것을 좁힌다 (3단계).

학습/CV/제출 전부 금지. 기존 OOF(blend_w05_oof.npy), 기존 test 관찰(v3_test_shift_row_level.csv,
v3_gene_freq_diff.csv), class_x_approach_f1.csv/burden_x_approach_f1.csv/confusion_edge_repetition.csv
(approach23_v4의 산출물)만 재사용한다.

목적: "train/test distribution shift가 있고 + 그 영역에서 OOF가 약하고 + 여러 접근에서 반복되고 +
작은 개입으로 검증 가능한" 교집합을 찾는다. 특히 이미 postprocessing으로 해결된 것(hypermut,
STES 31-100 일반)과 이미 approach11이 다룬 것(GBMLGG/LGG 마커)을 제외하고, 아직 안 건드린
후보(KIRC/KIPAN을 burden 관점에서)와 31-100 구간 내부의 잔여 취약 클래스를 확인한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

from main import ROOT, load_train
from features.features import gene_columns, TARGET

OUT = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
OUT_SYNTH = ROOT / "6. experiments" / "2026-09-14_approach23_synthetic_gbmlgg_lgg"
train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)

blend_oof = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
pred_oof = (blend_oof * s3).argmax(1)

row_te = pd.read_csv(OUT_SYNTH / "v3_test_shift_row_level.csv")

# ================================================================ 1. 31-100 구간 내부, 클래스별 OOF 취약도
print("=" * 78)
print("1. burden 31-100 구간, 클래스별 OOF recall/precision (16차 블렌드)")
m = (burden >= 31) & (burden <= 100)
yt, yp = y[m], pred_oof[m]
rows = []
for i, c in enumerate(classes):
    na = (yt == i).sum()
    if na == 0:
        continue
    rec = recall_score(yt == i, yp == i)
    prec = precision_score(yt == i, yp == i, zero_division=0)
    rows.append((c, na, rec, prec))
df31 = pd.DataFrame(rows, columns=["class", "n_actual_in_band", "recall", "precision"]).sort_values("recall")
print(df31.to_string(index=False))
print("\n(참고) test 31-100 구간 예측 분포(v3 스캔 재사용): STES 480, COAD 93, LUAD 71, LUSC 62 (총 969행)")

# ================================================================ 2. KIRC/KIPAN: burden 관점에서 train/test shift가 있는가
print("\n" + "=" * 78)
print("2. KIRC/KIPAN — burden 관점 train/test shift 확인 (GBMLGG/LGG와 동일한 렌즈)")
for c in ("KIRC", "KIPAN"):
    b = burden[train[TARGET] == c]
    print(f"train {c} (n={len(b)}): mean={b.mean():.1f} median={np.median(b):.1f} std={b.std():.1f} "
          f"p25={np.percentile(b,25):.1f} p75={np.percentile(b,75):.1f}")

nn_kirc = row_te[row_te.nn_train_class == "KIRC"]
nn_kipan = row_te[row_te.nn_train_class == "KIPAN"]
print(f"\ntest 중 최근접 train 이웃=KIRC: {len(nn_kirc)}행, burden mean={nn_kirc.burden.mean():.1f} median={nn_kirc.burden.median():.1f}")
print(f"test 중 최근접 train 이웃=KIPAN: {len(nn_kipan)}행, burden mean={nn_kipan.burden.mean():.1f} median={nn_kipan.burden.median():.1f}")
print("(비교) test 중 최근접 train 이웃=GBMLGG/LGG의 burden도 같이 확인:")
for c in ("GBMLGG", "LGG"):
    sub = row_te[row_te.nn_train_class == c]
    print(f"  nn={c}: {len(sub)}행, burden mean={sub.burden.mean():.1f} median={sub.burden.median():.1f}")

# KIRC vs KIPAN 오분류 OOF에서 burden 겹침 확인 (GBMLGG/LGG처럼 8-15류 경계가 있는지)
print("\nOOF에서 KIRC/KIPAN 정답 vs 오답의 burden 분포 (GBMLGG/LGG와 동일한 분석 반복)")
for c in ("KIRC", "KIPAN"):
    mask = (train[TARGET].to_numpy() == c)
    correct = mask & (pred_oof == y)
    wrong = mask & (pred_oof != y)
    for name, mm in (("정답", correct), ("오답", wrong)):
        b = burden[mm]
        if len(b) == 0:
            print(f"{c} {name}: 0행"); continue
        print(f"{c} {name} (n={len(b)}): mean={b.mean():.1f} median={np.median(b):.1f} std={b.std():.1f}")

# ================================================================ 3. KIRC/KIPAN 유전자 빈도 shift (train vs test, 데이터 기반)
print("\n" + "=" * 78)
print("3. KIRC/KIPAN을 가장 잘 가르는 유전자(train 빈도차 기준, 사전 목록 없이 데이터로 선정) 상위 10개의 test shift")
is_mut = (train[genes] != "WT")
f_kirc = is_mut[train[TARGET] == "KIRC"].mean()
f_kipan = is_mut[train[TARGET] == "KIPAN"].mean()
sep = (f_kirc - f_kipan).sort_values(key=np.abs, ascending=False)
gene_diff_te = pd.read_csv(OUT_SYNTH / "v3_gene_freq_diff.csv", index_col=0).iloc[:, 0]  # test-train diff, 유전자 인덱스
top_sep_genes = sep.head(10).index.tolist()
print(f"{'gene':10s} {'KIRC freq':>10s} {'KIPAN freq':>10s} {'sep':>7s}  {'test-train diff(전체)':>20s}")
for g in top_sep_genes:
    d = gene_diff_te.get(g, float("nan"))
    print(f"{g:10s} {f_kirc[g]:10.3f} {f_kipan[g]:10.3f} {sep[g]:7.3f}  {d:20.4f}")

print(f"\nsaved: (별도 저장 없음, 화면 출력만)")
