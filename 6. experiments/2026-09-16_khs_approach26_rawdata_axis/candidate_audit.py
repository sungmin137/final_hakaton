import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
COMMON = ROOT / "4. src/common"
sys.path.insert(0, str(COMMON))
from main import load_train, TARGET
from features.features import gene_columns

train = load_train()
genes = gene_columns(train)
y = train[TARGET].values
values = train[genes].to_numpy()
n_rows, n_cols = values.shape
gene_idx = {g:i for i,g in enumerate(genes)}

# token counts matrix
tok_counts = np.zeros((n_rows, n_cols), dtype=np.int16)
flat = values.ravel()
for idx in np.flatnonzero(flat != "WT"):
    r, c = divmod(idx, n_cols)
    tok_counts[r, c] = len(flat[idx].split(" "))

presence = (tok_counts >= 1)
total_presence_count = presence.sum(axis=0)  # per gene
# overall frequency rank (1 = most frequent)
order = np.argsort(-total_presence_count)
rank_of_gene = {genes[g]: (rank+1) for rank, g in enumerate(order)}
n_genes = len(genes)

positive = ["PIM1","SOCS1","MAP3K1","CHEK2","LRIG1","BRAF","PIK3CA","PTEN","EGFR","ATRX","APC","TP53"]
negative = ["PCLO","SYNE1","RYR2","DST","PLEC","AHNAK","SPTA1","RYR1","DMD","COL6A3","LRP1"]

def summarize(mask):
    n = mask.sum()
    if n == 0:
        return n, None, 0.0
    classes = y[mask]
    vc = pd.Series(classes).value_counts()
    top_class = vc.index[0]
    top_frac = vc.iloc[0] / n
    return n, top_class, top_frac

rows = []
for label, glist in [("positive", positive), ("negative", negative)]:
    for g in glist:
        if g not in gene_idx:
            rows.append(dict(group=label, gene=g, status="NOT_IN_TRAIN"))
            continue
        ci = gene_idx[g]
        rank = rank_of_gene[g]
        total_n = int(total_presence_count[ci])
        plain_mask = presence[:, ci]
        multi_mask = tok_counts[:, ci] >= 2
        n_plain, top_plain, frac_plain = summarize(plain_mask)
        n_multi, top_multi, frac_multi = summarize(multi_mask)
        rows.append(dict(
            group=label, gene=g, status="OK",
            overall_freq_count=total_n, overall_freq_rank=f"{rank}/{n_genes}",
            n_plain=n_plain, top_plain=top_plain, frac_plain=round(frac_plain,3) if n_plain else None,
            n_multi=n_multi, top_multi=top_multi, frac_multi=round(frac_multi,3) if n_multi else None,
        ))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(out.to_string(index=False))
out.to_csv("candidate_audit_result.csv", index=False)
