import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
from features.features import gene_columns

TRAIN = "/Users/admin/Desktop/해커톤_암종분류/branch_khs/1. info/data/train.csv"
train = pd.read_csv(TRAIN)
genes = gene_columns(train)
M = (train[genes].to_numpy() != "WT")
classes = sorted(train["SUBCLASS"].unique())
carrier_counts = M.sum(axis=0)
freq_df = pd.Series(carrier_counts, index=genes)

print("n_rows", len(train), "n_genes_total", len(genes), "n_classes", len(classes))
print("\nclass sizes:")
print(train["SUBCLASS"].value_counts().sort_index().to_string())

print("\nFrequency distribution of gene carrier counts (total across all 6201 rows):")
for q in [1,2,5,10,15,20,30,50,100,200]:
    print(f"  genes with carrier_count >= {q}: {int((freq_df>=q).sum())}")
print(freq_df.describe())

CUTOFF = 15
sel_genes = [g for g,c in zip(genes, carrier_counts) if c >= CUTOFF]
print(f"\nSelected genes (>= {CUTOFF} total carriers): {len(sel_genes)} / {len(genes)}")

gi_all = {g:i for i,g in enumerate(genes)}
sel_idx = [gi_all[g] for g in sel_genes]
Msel = M[:, sel_idx]
y = train["SUBCLASS"].values

def rate_matrix(mask_rows, M_, y_, classes_):
    sub_M = M_[mask_rows]; sub_y = y_[mask_rows]
    out = np.zeros((M_.shape[1], len(classes_)))
    for ci, c in enumerate(classes_):
        idx = sub_y == c
        n = idx.sum()
        out[:, ci] = sub_M[idx].sum(axis=0) / n if n else np.nan
    return out

rate_full = rate_matrix(np.ones(len(train), bool), Msel, y, classes)
rate_df = pd.DataFrame(rate_full, index=sel_genes, columns=classes)
print("\nGene x class rate matrix shape:", rate_df.shape)
rate_df.to_csv("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/rate_matrix.csv")

eps = 1e-9
rows = []
for g in sel_genes:
    r = rate_df.loc[g].values
    mean_r = r.mean()
    n_classes_above = int((r > 3*mean_r).sum()) if mean_r > 0 else 0
    p = r / (r.sum() + eps)
    ent = -np.nansum(np.where(p>0, p*np.log(p+eps), 0))
    order = np.argsort(-r)
    top1c, top2c = classes[order[0]], classes[order[1]]
    top1r, top2r = r[order[0]], r[order[1]]
    gap = top1r - top2r
    rows.append(dict(gene=g, entropy=ent, n_classes_above_3xmean=n_classes_above,
                      top1_class=top1c, top1_rate=top1r, top2_class=top2c, top2_rate=top2r, gap=gap,
                      carrier_count=int(carrier_counts[gi_all[g]])))
profile_df = pd.DataFrame(rows).set_index("gene")

def classify(row):
    if row["n_classes_above_3xmean"] >= 8:
        return "범용"
    if row["gap"] < 0.02 and row["n_classes_above_3xmean"] <= 3:
        return "애매(2class)"
    if row["n_classes_above_3xmean"] <= 3:
        return "특이"
    return "기타"

profile_df["type"] = profile_df.apply(classify, axis=1)
print("\nEntropy stats:")
print(profile_df["entropy"].describe())
print("\nQ1/Q2/Q4 profile type counts:")
print(profile_df["type"].value_counts())
print("\nGap distribution (top1_rate - top2_rate):")
print(profile_df["gap"].describe())
profile_df.to_csv("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/profile_df.csv")
print("DONE STEP12")
