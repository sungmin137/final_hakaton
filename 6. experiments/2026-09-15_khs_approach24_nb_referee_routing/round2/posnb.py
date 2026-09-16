"""Candidate A: gene-relative position-bucket NB. Honest per-fold fit (no leakage)."""
import sys, re, json, time
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from main import load_train, TARGET, twin_groups
from features.features import gene_columns

_POS = re.compile(r"^[A-Z]?(\d+)")
_MIS = re.compile(r"^([A-Z])(\d+)([A-Z*])$")
MIN_CARRIERS = 200   # gene inclusion threshold (fit-fold only) -> ~85 genes x 4 buckets ~= 340 dims, comparable to spec-NB's 383
N_BUCKET = 4

def _kind(tok):
    if tok.endswith("*") or "fs" in tok: return "lof"
    m = _MIS.match(tok)
    if m: return "syn" if m.group(1) == m.group(3) else "mis"
    return "other"

def _pos(tok):
    m = _POS.match(tok)
    return int(m.group(1)) if m else None

def fit_gene_buckets(df, genes):
    """fit-part only: select genes (carrier count >= MIN_CARRIERS), compute per-gene quartile edges
    from that gene's own OBSERVED functional-mutation positions in the fit fold."""
    G = df[genes].to_numpy()
    carrier_cnt = (df[genes] != "WT").sum(axis=0)
    sel_genes = [g for g in genes if carrier_cnt[g] >= MIN_CARRIERS]
    gi_all = {g: k for k, g in enumerate(genes)}
    pos_by_gene = {g: [] for g in sel_genes}
    sel_set = set(sel_genes)
    for i, j in zip(*np.nonzero(G != "WT")):
        g = genes[j]
        if g not in sel_set: continue
        for t in G[i, j].split(" "):
            if _kind(t) == "syn": continue
            p = _pos(t)
            if p is not None: pos_by_gene[g].append(p)
    edges = {}
    for g in sel_genes:
        p = np.array(pos_by_gene[g])
        if len(p) < N_BUCKET:  # degenerate: too few points for quartiles, drop gene
            continue
        qs = np.quantile(p, [0.25, 0.5, 0.75])
        qs = np.unique(qs)  # guard against ties collapsing bucket edges
        edges[g] = qs
    final_genes = list(edges.keys())
    return final_genes, edges

def build_matrix(df, genes, sel_genes, edges):
    G = df[genes].to_numpy()
    col_idx = {}
    ncol = 0
    gene_bucket_offset = {}
    for g in sel_genes:
        nb = len(edges[g]) + 1  # buckets = len(unique edges)+1
        gene_bucket_offset[g] = (ncol, nb)
        ncol += nb
    X = np.zeros((len(df), ncol), dtype=np.float32)
    sel_set = set(sel_genes)
    for i, j in zip(*np.nonzero(G != "WT")):
        g = genes[j]
        if g not in sel_set: continue
        off, nb = gene_bucket_offset[g]
        for t in G[i, j].split(" "):
            if _kind(t) == "syn": continue
            p = _pos(t)
            if p is None: continue
            b = np.searchsorted(edges[g], p, side="right")  # 0..nb-1
            X[i, off + b] += 1
    return X


def main():
    train = load_train()
    genes = gene_columns(train)
    le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
    groups = twin_groups(train)
    n = len(train)
    oof = np.zeros((n, len(classes)))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    t0 = time.time()
    dims = []
    for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
        tr_part = train.iloc[tri]
        sel_genes, edges = fit_gene_buckets(tr_part, genes)
        dims.append(sum(len(edges[g])+1 for g in sel_genes))
        Xtr = build_matrix(tr_part, genes, sel_genes, edges)
        Xva = build_matrix(train.iloc[vai], genes, sel_genes, edges)
        m = MultinomialNB(alpha=0.5).fit(Xtr, y[tri])
        oof[vai] = m.predict_proba(Xva)
        print(f"fold {k}: n_genes={len(sel_genes)} dim={Xtr.shape[1]} fold_f1={f1_score(y[vai], oof[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)

    f1 = f1_score(y, oof.argmax(1), average="macro")
    print(f"\nstandalone Position-bucket NB macro F1 = {f1:.4f}")
    print(f"avg dims per fold = {np.mean(dims):.1f}")

    OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/"
    np.save(OUT+"posnb_oof.npy", oof)
    json.dump({"standalone_macro_f1": f1, "avg_dims": float(np.mean(dims)), "min_carriers": MIN_CARRIERS, "n_bucket_target": N_BUCKET}, open(OUT+"posnb_result.json","w"))

if __name__ == "__main__":
    main()
