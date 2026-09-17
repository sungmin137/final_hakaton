"""train-only 구조 감사: multi-token cell 내부 same/mixed consequence-type 각도.
모델 학습 없음. test.csv 미사용.
"""
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
ID, TARGET = "ID", "SUBCLASS"

train = pd.read_csv(ROOT / "1. info" / "data" / "train.csv")
genes = [c for c in train.columns if c not in (ID, TARGET)]
G = train[genes].to_numpy()
n_rows, n_cols = G.shape


def mut_type(tok: str) -> str:
    if tok.endswith("*"):
        return "nonsense"
    if "fs" in tok:
        return "frameshift"
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
    if m:
        return "synonymous" if m.group(1) == m.group(3) else "missense"
    return "other"


def position(tok: str):
    m = re.match(r"^[A-Za-z](\d+)", tok)
    return m.group(1) if m else None


# -------------------- 1. same/mixed 전체 비율 재확인 --------------------
same_flags = []          # per multi-token cell: True=same-type, False=mixed
row_idx_of_cell = []      # sample row index for each multi-token cell
gene_of_cell = []
types_of_cell = []

for i, j in zip(*np.nonzero(G != "WT")):
    toks = G[i, j].split(" ")
    if len(toks) < 2:
        continue
    types = [mut_type(t) for t in toks]
    same = len(set(types)) == 1
    same_flags.append(same)
    row_idx_of_cell.append(i)
    gene_of_cell.append(genes[j])
    types_of_cell.append(types)

same_flags = np.array(same_flags)
n_multi = len(same_flags)
n_same = same_flags.sum()
n_mixed = n_multi - n_same

print("=" * 78)
print("1. same/mixed consequence-type 재확인 (전체 비율)")
print(f"multi-token 셀 총 개수: {n_multi}")
print(f"same-type: {n_same} ({n_same/n_multi:.4%})")
print(f"mixed-type: {n_mixed} ({n_mixed/n_multi:.4%})")

# -------------------- 2. missense-only 내부 동일 위치 충돌 --------------------
print("\n" + "=" * 78)
print("2. missense 내부 동일 위치 충돌 여부")

missense_only_cells = []
for idx in range(n_multi):
    types = types_of_cell[idx]
    if all(t == "missense" for t in types):
        missense_only_cells.append(idx)

n_missense_only = len(missense_only_cells)
print(f"missense-only multi-token 셀 개수: {n_missense_only}")

collision_cells = 0
identical_dup_cells = 0
for idx in missense_only_cells:
    i = row_idx_of_cell[idx]
    j = genes.index(gene_of_cell[idx])
    toks = G[i, j].split(" ")
    positions = [position(t) for t in toks]
    # 동일 position인데 서로 다른 치환(different outcome) -> 충돌
    pos_to_toks = defaultdict(list)
    for t, p in zip(toks, positions):
        pos_to_toks[p].append(t)
    for p, tlist in pos_to_toks.items():
        if len(tlist) >= 2:
            if len(set(tlist)) == 1:
                identical_dup_cells += 1  # 완전히 동일한 토큰 중복 (예: "R58Q R58Q")
            else:
                collision_cells += 1      # 같은 위치, 다른 치환 (예: "R58Q R58W")

print(f"같은 position, 다른 substitution 충돌 셀: {collision_cells}")
print(f"같은 position, 완전 동일 토큰 중복 셀: {identical_dup_cells}")
if collision_cells > 0:
    # 예시 몇 개 출력
    shown = 0
    for idx in missense_only_cells:
        i = row_idx_of_cell[idx]
        j = genes.index(gene_of_cell[idx])
        toks = G[i, j].split(" ")
        positions = [position(t) for t in toks]
        pos_to_toks = defaultdict(list)
        for t, p in zip(toks, positions):
            pos_to_toks[p].append(t)
        if any(len(set(tl)) > 1 for tl in pos_to_toks.values() if len(tl) >= 2):
            print(f"  예: gene={gene_of_cell[idx]}, tokens={toks}")
            shown += 1
        if shown >= 5:
            break

# -------------------- 3. 샘플별 same/mixed 비율 & burden 층화 class 비교 --------------------
print("\n" + "=" * 78)
print("3. 샘플별 same/mixed 비율의 class 차이 (burden 층화)")

df_cells = pd.DataFrame({
    "row": row_idx_of_cell,
    "same": same_flags,
})
per_sample = df_cells.groupby("row")["same"].agg(["sum", "count"]).rename(
    columns={"sum": "n_same", "count": "n_multi_genes"})
per_sample["frac_same"] = per_sample["n_same"] / per_sample["n_multi_genes"]

n_samples_with_multi = len(per_sample)
print(f"multi-token gene을 1개 이상 가진 샘플 수: {n_samples_with_multi} "
      f"({n_samples_with_multi/n_rows:.2%} of train)")

burden = (train[genes] != "WT").sum(axis=1).to_numpy()
per_sample["burden"] = burden[per_sample.index]
per_sample["class"] = train[TARGET].to_numpy()[per_sample.index]

BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi < 10**9 else '+'}" for lo, hi in BINS]


def bin_of(b):
    for (lo, hi), nm in zip(BINS, bin_names):
        if lo <= b <= hi:
            return nm
    return "?"


per_sample["burden_bin"] = per_sample["burden"].apply(bin_of)

print(f"\nburden bin별 샘플 수 (multi-token 보유 샘플 중):")
print(per_sample["burden_bin"].value_counts().reindex(bin_names).to_string())

print("\nburden bin별 frac_same 전체 평균/표준편차:")
print(per_sample.groupby("burden_bin")["frac_same"].agg(["mean", "std", "count"]).reindex(bin_names).to_string())

# bin 내부에서 class별 평균 frac_same의 분산 vs bin 내부 전체 분산 비교(경량 ANOVA류)
print("\nbin별 class-평균 frac_same 분산(between) vs 표본 분산(total) 비율:")
rows = []
for nm in bin_names:
    sub = per_sample[per_sample["burden_bin"] == nm]
    if len(sub) < 30:
        rows.append((nm, len(sub), np.nan, np.nan, np.nan))
        continue
    grp = sub.groupby("class")["frac_same"]
    class_means = grp.mean()
    class_counts = grp.count()
    # class 5개 미만 표본은 제외(노이즈)
    valid = class_counts[class_counts >= 5].index
    if len(valid) < 2:
        rows.append((nm, len(sub), np.nan, np.nan, np.nan))
        continue
    cm = class_means.loc[valid]
    cc = class_counts.loc[valid]
    grand_mean = np.average(cm, weights=cc)
    between_var = np.average((cm - grand_mean) ** 2, weights=cc)
    total_var = sub["frac_same"].var()
    ratio = between_var / total_var if total_var > 0 else np.nan
    rows.append((nm, len(sub), len(valid), between_var, ratio))

btab = pd.DataFrame(rows, columns=["burden_bin", "n_samples", "n_classes_valid", "between_var", "between/total_var"])
print(btab.to_string(index=False))

print("\n전체(burden 무시) class별 frac_same 평균 (참고, burden 교란 있음):")
overall_class = per_sample.groupby("class")["frac_same"].agg(["mean", "count"]).sort_values("mean")
print(overall_class.to_string())

print("\n" + "=" * 78)
print("스크립트 종료 (모델 학습 없음, train-only)")
