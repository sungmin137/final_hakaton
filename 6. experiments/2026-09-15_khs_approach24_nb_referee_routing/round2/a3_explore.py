import sys, re
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from main import load_train
from features.features import gene_columns

train = load_train()
genes = gene_columns(train)
G = train[genes].to_numpy()
_POS = re.compile(r"^[A-Z]?(\d+)")

def positions(gene):
    j = genes.index(gene)
    pos = []
    for i in range(len(train)):
        v = G[i, j]
        if v == "WT": continue
        for t in v.split(" "):
            m = _POS.match(t)
            if m: pos.append(int(m.group(1)))
    return pos

cnt = (train[genes] != "WT").sum(0).sort_values(ascending=False)
print("top20 genes by mutated-sample count:")
print(cnt.head(20))

print("\ngene: n_mut_tokens, min_pos, max_pos, p50, p90, n_unique_pos")
for g in cnt.head(20).index:
    p = positions(g)
    if not p: continue
    p = np.array(p)
    print(f"{g:10s} n={len(p):5d} min={p.min():5d} max={p.max():6d} p50={np.median(p):7.1f} p90={np.percentile(p,90):7.1f} nuniq={len(set(p.tolist()))}")
