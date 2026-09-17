"""approach23 v11: TP53&APC&31-100 -> allowed-set 규칙의 최종 사전검증. 학습/제출/threshold sweep 금지.
기존 OOF(16차 블렌드 + 28개 group-twins OOF) + train 원본만 사용.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

from main import ROOT, SEED, twin_groups, load_train
from features.features import gene_columns, TARGET

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
true_lab = train[TARGET].to_numpy()

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
blend_oof = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
proba_scaled = blend_oof * s3; proba_scaled /= proba_scaled.sum(1, keepdims=True)
pred17 = np.array(classes)[proba_scaled.argmax(1)]
conf17 = proba_scaled.max(1)

m31 = (burden >= 31) & (burden <= 100)
cand51 = m31 & (train["TP53"] != "WT").to_numpy() & (train["APC"] != "WT").to_numpy()
print(f"TP53&APC&31-100 train 행 수: {cand51.sum()}")
ALLOWED = {"STES", "COAD", "HNSC", "LUSC", "OV"}

# fold 재현 (main.py와 동일 SEED, group_twins split)
groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_of = np.full(len(train), -1)
for k, (_, vai) in enumerate(skf.split(train, y, groups)):
    fold_of[vai] = k

idx51 = np.flatnonzero(cand51)

print("\n" + "=" * 78)
print("1. 51행 true class별 요약")
df51 = pd.DataFrame({
    "idx": idx51, "true": true_lab[idx51], "pred17": pred17[idx51], "conf17": conf17[idx51],
    "fold": fold_of[idx51], "outside_allowed": ~np.isin(pred17[idx51], list(ALLOWED)),
    "correct": true_lab[idx51] == pred17[idx51],
})
summary = df51.groupby("true").agg(n=("true", "size"), correct=("correct", "sum"),
                                    outside=("outside_allowed", "sum")).reset_index()
summary["incorrect"] = summary.n - summary.correct
summary["outside_rate"] = (summary.outside / summary.n).round(3)
print(summary[["true", "n", "correct", "incorrect", "outside", "outside_rate"]].to_string(index=False))

print("\n" + "=" * 78)
print("2. allowed-set 밖 4행 상세")
out4 = df51[df51.outside_allowed]
print(out4.to_string(index=False))
print(f"\n같은 fold에 몰려있는가: fold 분포 = {out4.fold.value_counts().to_dict()}")
print(f"confidence 분포: {out4.conf17.tolist()} (전체 51행 conf 평균 {df51.conf17.mean():.3f}, 중앙값 {df51.conf17.median():.3f})")
print("\n각 행의 주요 변이(WT 아닌 유전자, TP53/APC 제외 상위 몇 개):")
for i in out4.idx:
    row = train.loc[i, genes]
    muts = row[row != "WT"].index.tolist()
    other = [g for g in muts if g not in ("TP53", "APC")]
    print(f"  idx={i} true={true_lab[i]} pred={pred17[i]} burden={burden[i]} 변이유전자수={len(muts)} 예시={other[:8]}")

# ================================================================ 3. 28개 group-twins OOF 교차확인
print("\n" + "=" * 78)
print("3. 28개 group-twins OOF에서 이 51행(및 특히 4행)의 예측 재현성")
runs = []
for rp in sorted(glob.glob(str(ROOT / "6. experiments" / "**" / "result.json"), recursive=True)):
    d = json.load(open(rp))
    if not d.get("group_twins") or "khs_tp53_local" in rp:
        continue
    npy = Path(rp).parent / "oof_proba.npy"
    if not npy.exists():
        continue
    proba = np.load(npy)
    if proba.shape[0] != len(train):
        continue
    runs.append((Path(rp).parent.name, np.array(classes)[proba.argmax(1)]))
print(f"비교 대상 OOF {len(runs)}개 (스케일 미적용 raw argmax 기준 — 모델 간 일관성만 확인)")

rows_summary = []
for i in idx51:
    preds_i = [p[i] for _, p in runs]
    n_out = sum(1 for p in preds_i if p not in ALLOWED)
    n_in = len(preds_i) - n_out
    most_common = pd.Series(preds_i).value_counts().idxmax()
    rows_summary.append((i, true_lab[i], n_in, n_out, most_common))
cross_df = pd.DataFrame(rows_summary, columns=["idx", "true", "n_in_allowed", "n_out_allowed", "most_common_pred"])
print("\n전체 51행 요약(밖으로 예측된 모델 수 상위 10행):")
print(cross_df.sort_values("n_out_allowed", ascending=False).head(10).to_string(index=False))

print("\n특히 원래 4개 후보 행의 28개 모델별 예측 재현성:")
for i in out4.idx:
    preds_i = [p[i] for _, p in runs]
    vc = pd.Series(preds_i).value_counts()
    n_out = sum(1 for p in preds_i if p not in ALLOWED)
    print(f"idx={i} true={true_lab[i]} 17차pred={pred17[i]} | 28개 모델 예측분포: {vc.to_dict()} | 밖으로 나간 모델 수={n_out}/{len(runs)}")

# ================================================================ 4. allowed-set 독립성 점검
print("\n" + "=" * 78)
print("4. allowed-set {STES,COAD,HNSC,LUSC,OV}의 독립성")
print("이 집합은 지난 분석에서 '이 51행(TP53&APC&31-100)의 실제 라벨 분포'로부터 직접 정의됐다.")
print("즉 51행 중 4행이 밖으로 나갔는지 확인하는 이번 검증은, 그 4행의 정답이 '같은 51행 집단의 다수 라벨'과 "
      "같은 카테고리에 속하는지를 재확인하는 것과 사실상 같아 완전히 독립적이지 않다.")
print("hypermut_rule도 동일한 방식(초과변이 train 행의 실제 라벨 집합)으로 allowed-set을 정의했었지만, "
      "그때는 test에서만 최종 검증했다 — 여기서는 아직 test 라벨이 없어 이 순환성을 원천적으로 못 피한다.")
