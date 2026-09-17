"""B-2 audit: dom_<GENE>_<DOMAIN> vs hs_<GENE>:<variant> and lit_<CLASS>_score overlap."""
import sys
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from main import load_train, TARGET
from features.features import gene_columns, InsightFeatures
from approach7_preprocess.preprocess_features import PreprocessFeatures

train = load_train()
genes = gene_columns(train)

pp = PreprocessFeatures().fit(train)
Xdom = pp.transform(train)
dom_cols = [c for c in Xdom.columns if c.startswith("dom_")]
print(f"total dom_ columns: {len(dom_cols)}")

# per-column hit count, sorted
dom_hits = Xdom[dom_cols].sum(0).sort_values(ascending=False)
print("\ntop 15 domains by hit count:")
print(dom_hits.head(15))

ins = InsightFeatures().fit(train)
Xhs = ins.transform(train)
hs_cols = [c for c in Xhs.columns if c.startswith("hs_")]

# per-gene: total domain hits (any domain for that gene) vs total hotspot hits (any hotspot for that gene) vs total mutated-sample count for gene
genes_with_dom = sorted(set(c.split("_")[1] for c in dom_cols))
print(f"\ngenes with domain annotation: {len(genes_with_dom)}")
print(genes_with_dom)

rows = []
for g in genes_with_dom:
    gdom_cols = [c for c in dom_cols if c.startswith(f"dom_{g}_")]
    dom_total = Xdom[gdom_cols].sum(1)  # per-patient count across this gene's domains (could be >1 if multi-domain hit... no, usually 0/1 per domain, sum could be >1 if variant falls in overlapping domains, rare)
    dom_any = (dom_total > 0).astype(int)
    ghs_cols = [c for c in hs_cols if c.startswith(f"hs_{g}:")]
    if ghs_cols:
        hs_any = (Xhs[ghs_cols].sum(1) > 0).astype(int)
    else:
        hs_any = pd.Series(np.zeros(len(train), dtype=int), index=train.index)
    mut_any = (train[g] != "WT").astype(int)
    n_dom_carriers = int(dom_any.sum())
    n_hs_carriers = int(hs_any.sum())
    n_mut_carriers = int(mut_any.sum())
    # among domain carriers, how many are ALSO hotspot carriers for same gene
    overlap = int(((dom_any==1) & (hs_any==1)).sum())
    corr = np.corrcoef(dom_any, mut_any)[0,1] if dom_any.std()>0 and mut_any.std()>0 else np.nan
    rows.append(dict(gene=g, n_dom_carriers=n_dom_carriers, n_hs_carriers=n_hs_carriers,
                      n_mut_carriers=n_mut_carriers, overlap_dom_and_hs=overlap,
                      frac_dom_that_is_hs=round(overlap/max(n_dom_carriers,1),3),
                      corr_dom_vs_mut=round(corr,3) if corr==corr else None))

df = pd.DataFrame(rows).sort_values("n_dom_carriers", ascending=False)
pd.set_option("display.width", 160)
print("\nper-gene domain vs hotspot vs mutation-count overlap (sorted by domain carrier count):")
print(df.head(20).to_string(index=False))

OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/"
df.to_csv(OUT+"domain_hs_overlap.csv", index=False)

# also: does a gene have multiple domains where mutations cluster differently vs. spread?
print("\n--- distinct-structure check: for genes w/ >=2 domains, how many samples hit >1 domain (spread) vs exactly1 (clustered)? ---")
for g in genes_with_dom:
    gdom_cols = [c for c in dom_cols if c.startswith(f"dom_{g}_")]
    if len(gdom_cols) < 2: continue
    cnt_domains_hit = (Xdom[gdom_cols] > 0).sum(1)
    carriers = cnt_domains_hit[cnt_domains_hit>0]
    if len(carriers)==0: continue
    print(f"{g:10s} ndomains={len(gdom_cols)} carriers={len(carriers):4d} multi-domain-hit={int((carriers>1).sum()):4d} ({(carriers>1).mean():.1%})")
