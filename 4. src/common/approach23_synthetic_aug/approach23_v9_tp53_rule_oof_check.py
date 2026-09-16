"""approach23 v9: TP53/31-100/STES-LUSC 후처리 후보의 OOF 타당성 검증.
학습/CV/제출 전부 금지. 16차 블렌드 OOF(blend_w05_oof.npy) + train 원본만 사용.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from main import ROOT, load_train
from features.features import gene_columns, TARGET

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_)
true_lab = train[TARGET].to_numpy()
tp53_mut = (train["TP53"] != "WT").to_numpy()

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
blend_oof = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
proba_scaled = blend_oof * s3
proba_scaled = proba_scaled / proba_scaled.sum(1, keepdims=True)
pred = np.array(classes)[proba_scaled.argmax(1)]
conf = proba_scaled.max(1)

m31 = (burden >= 31) & (burden <= 100)

# ================================================================ 2. candidate set 정의
print("=" * 78)
print("2. candidate = burden 31-100 & TP53+ & pred in {STES,LUSC}")
cand = m31 & tp53_mut & np.isin(pred, ["STES", "LUSC"])
n = cand.sum()
correct = (true_lab[cand] == pred[cand])
print(f"n={n}")
print(f"실제 정답이 STES/LUSC인 비율: {np.isin(true_lab[cand], ['STES','LUSC']).mean():.3f} "
      f"({np.isin(true_lab[cand], ['STES','LUSC']).sum()}/{n})")
print(f"실제 정답이 STES/LUSC가 아닌 비율: {(~np.isin(true_lab[cand], ['STES','LUSC'])).mean():.3f}")
print("\n실제 정답 class 분포:")
print(pd.Series(true_lab[cand]).value_counts().to_string())
print("\nconfidence 분포:")
print(pd.Series(conf[cand]).describe(percentiles=[.25, .5, .75]).round(3).to_string())

# ================================================================ 3. STES vs LUSC 분리
print("\n" + "=" * 78)
print("3. STES(Group A) vs LUSC(Group B) 분리")
for grp, c in [("A(pred=STES)", "STES"), ("B(pred=LUSC)", "LUSC")]:
    mm = cand & (pred == c)
    if mm.sum() == 0:
        print(f"{grp}: 0행"); continue
    corr = (true_lab[mm] == c)
    print(f"\n{grp}: n={mm.sum()}, correct={corr.sum()}({corr.mean():.1%}), incorrect={(~corr).sum()}({(~corr).mean():.1%})")
    print("  실제 정답 class 분포:")
    print("  " + pd.Series(true_lab[mm]).value_counts().to_string().replace("\n", "\n  "))
    print(f"  confidence: mean={conf[mm].mean():.3f} median={np.median(conf[mm]):.3f} "
          f"p25={np.percentile(conf[mm],25):.3f} p75={np.percentile(conf[mm],75):.3f}")

# ================================================================ 4. confidence 구간별 정답률
print("\n" + "=" * 78)
print("4. candidate set 내 confidence 3분위별 정답률")
q = pd.qcut(conf[cand], 3, labels=["low", "mid", "high"])
acc_by_q = pd.Series(correct).groupby(q.to_numpy(), observed=True).agg(["mean", "count"])
print(acc_by_q.round(3).to_string())

# ================================================================ 5. TP53 negative control (같은 31-100 조건, TP53 유무만 다름)
print("\n" + "=" * 78)
print("5. negative control: 31-100 구간에서 TP53+ vs TP53- (OOF)")
for label, mask in [("TP53+", m31 & tp53_mut), ("TP53-", m31 & ~tp53_mut)]:
    pred_stes_lusc = np.isin(pred[mask], ["STES", "LUSC"])
    true_stes_lusc = np.isin(true_lab[mask], ["STES", "LUSC"])
    pred_share = pred_stes_lusc.mean()
    true_share = true_stes_lusc.mean()
    if pred_stes_lusc.sum() > 0:
        wrong_rate = (~(true_lab[mask][pred_stes_lusc] == pred[mask][pred_stes_lusc])).mean()
    else:
        wrong_rate = float("nan")
    print(f"{label} (n={mask.sum()}): STES/LUSC 예측비율={pred_share:.3f}  실제 STES/LUSC 비율={true_share:.3f}  "
          f"STES/LUSC로 예측된 것 중 오분류율={wrong_rate:.3f}")

# ================================================================ 6. TP53 특이성 -- 다른 shift 유전자와 비교
print("\n" + "=" * 78)
print("6. TP53 특이성 확인: 다른 test-train shift 상위 유전자와 동일 비교")
other_genes = ["AHNAK", "TCHH", "APC", "CACNA1A", "KMT2D", "CPEB2", "KDM6B"]
for g in other_genes:
    if g not in train.columns:
        continue
    g_mut = (train[g] != "WT").to_numpy()
    mask = m31 & g_mut
    if mask.sum() < 10:
        print(f"{g}: 표본 부족(n={mask.sum()}) 생략"); continue
    pred_share = np.isin(pred[mask], ["STES", "LUSC"]).mean()
    true_share = np.isin(true_lab[mask], ["STES", "LUSC"]).mean()
    print(f"{g}+ (31-100 내 n={mask.sum()}): STES/LUSC 예측비율={pred_share:.3f}  실제비율={true_share:.3f}")
print(f"(비교) TP53+ (31-100 내 n={(m31&tp53_mut).sum()}): STES/LUSC 예측비율="
      f"{np.isin(pred[m31&tp53_mut], ['STES','LUSC']).mean():.3f}  실제비율={np.isin(true_lab[m31&tp53_mut], ['STES','LUSC']).mean():.3f}")

# ================================================================ 7. 회수 가능한 오류 규모
print("\n" + "=" * 78)
print("7. candidate 중 오분류 행의 실제 정답 방향 (대체 클래스 후보)")
wrong_mask = cand & (true_lab != pred)
print(f"candidate {n}행 중 오분류 {wrong_mask.sum()}행({wrong_mask.mean():.1%})")
print("오분류 행의 실제 정답 분포 (대체 방향 후보):")
print(pd.Series(true_lab[wrong_mask]).value_counts().to_string())
