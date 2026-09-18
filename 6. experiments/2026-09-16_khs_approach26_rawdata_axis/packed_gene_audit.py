"""Task 1/2: identify the 'packed gene' (argmax per-gene token count) per row,
reusing the tok_counts logic from eg_maxtok_stage1.py. TRAIN-ONLY, no model training.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
COMMON = ROOT / "4. src/common"
sys.path.insert(0, str(COMMON))
SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")

from main import load_train, TARGET
from features.features import gene_columns

train = load_train()
genes = gene_columns(train)
values = train[genes].to_numpy()
n_rows, n_cols = values.shape

tok_counts = np.zeros((n_rows, n_cols), dtype=np.int16)
flat = values.ravel()
for idx in np.flatnonzero(flat != "WT"):
    r, c = divmod(idx, n_cols)
    tok_counts[r, c] = len(flat[idx].split(" "))

# cross-check against cached max_tok
G = (values != "WT").sum(axis=1).astype(np.int64)
max_tok_recomp = np.zeros(n_rows, dtype=np.float64)
packed_gene_idx = np.full(n_rows, -1, dtype=np.int64)
for r in range(n_rows):
    row = tok_counts[r]
    if row.max() > 0:
        max_tok_recomp[r] = row.max()
        # if ties, take first (order arbitrary but consistent)
        packed_gene_idx[r] = np.argmax(row)

G_cache = np.load(SC / "s1_G.npy")
mt_cache = np.load(SC / "s1_maxtok_raw.npy")
y_cache = np.load(SC / "s1_y.npy")
classes = json.load(open(SC / "s1_classes.json"))

assert np.array_equal(G, G_cache), "G mismatch vs cache -- row order differs!"
assert np.allclose(max_tok_recomp, mt_cache), "max_tok mismatch vs cache!"
print("Sanity check passed: recomputed G and max_tok match cached arrays exactly.")

packed_gene = np.array([genes[i] if i >= 0 else None for i in packed_gene_idx])
class_labels = np.array(classes)[y_cache]

df = pd.DataFrame({
    "class": class_labels,
    "G": G_cache,
    "max_tok": mt_cache,
    "packed_gene": packed_gene,
})

df.to_csv(SC / "packed_gene_table.csv", index=False)
print(f"n_rows={n_rows}, rows with max_tok>=2: {(df.max_tok>=2).sum()}")
print(df[df.max_tok>=2].max_tok.value_counts().sort_index())
