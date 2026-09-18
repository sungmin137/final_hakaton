import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
from features.features import gene_columns
from approach4_literature.literature_map import GENE_INFO
from sklearn.model_selection import StratifiedGroupKFold
from postprocess.twin_rule import _hash_rows

TRAIN = "/Users/admin/Desktop/해커톤_암종분류/branch_khs/1. info/data/train.csv"
FEATURES_PY = "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common/features/features.py"

train = pd.read_csv(TRAIN)
genes = gene_columns(train)
print("n_rows", len(train), "n_genes", len(genes))

M = (train[genes].to_numpy() != "WT")  # bool matrix n x g
classes = sorted(train["SUBCLASS"].unique())
print("n_classes", len(classes))

# ---- Step1: frequency filter ----
carrier_counts = M.sum(axis=0)
freq_df = pd.Series(carrier_counts, index=genes)
print("\nFrequency distribution of gene carrier counts:")
for q in [0,1,2,5,10,15,20,30,50,100]:
    print(f"  >= {q}: {int((freq_df>=q).sum())} genes")
print(freq_df.describe())

CUTOFF = 15
sel_genes = [g for g,c in zip(genes, carrier_counts) if c >= CUTOFF]
print(f"\nSelected genes with carrier_count >= {CUTOFF}: {len(sel_genes)}")

gi_all = {g:i for i,g in enumerate(genes)}
sel_idx = [gi_all[g] for g in sel_genes]
Msel = M[:, sel_idx]  # n x |sel_genes|

y = train["SUBCLASS"].values

def rate_matrix(mask_rows, M_, y_, classes_):
    """Return DataFrame genes x classes of P(gene mut | class) using rows where mask_rows True."""
    sub_M = M_[mask_rows]
    sub_y = y_[mask_rows]
    out = np.zeros((M_.shape[1], len(classes_)))
    for ci, c in enumerate(classes_):
        idx = sub_y == c
        n = idx.sum()
        if n == 0:
            out[:, ci] = np.nan
            continue
        out[:, ci] = sub_M[idx].sum(axis=0) / n
    return out

full_mask = np.ones(len(train), dtype=bool)
rate_full = rate_matrix(full_mask, Msel, y, classes)  # genes x 26
rate_df = pd.DataFrame(rate_full, index=sel_genes, columns=classes)
print("\nFull gene x class rate matrix shape:", rate_df.shape)

rate_df.to_csv("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/rate_matrix.csv")

# ---- Step2: per-gene profile ----
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

# classify
def classify(row):
    if row["n_classes_above_3xmean"] >= 8:
        return "범용"
    if row["gap"] < 0.02 and row["n_classes_above_3xmean"] <= 3:
        return "애매(2class)"
    if row["n_classes_above_3xmean"] <= 3:
        return "특이"
    return "기타"

profile_df["type"] = profile_df.apply(classify, axis=1)
print("\nQ1/Q2/Q4 profile type counts:")
print(profile_df["type"].value_counts())

profile_df.to_csv("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/profile_df.csv")

# ---- Step3: boundary direction analysis ----
group_ids = pd.factorize(_hash_rows(train, genes))[0]
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
y_le = pd.factorize(y)[0]
fold_train_masks = []
for tri, vai in skf.split(train, y_le, group_ids):
    mask = np.zeros(len(train), dtype=bool)
    mask[tri] = True
    fold_train_masks.append(mask)
print("\nfold train sizes:", [m.sum() for m in fold_train_masks])

def boundary_analysis(A, B, topn=50, min_dominant_count=10):
    nA = (y == A).sum(); nB = (y == B).sum()
    countA = Msel[y == A].sum(axis=0)
    countB = Msel[y == B].sum(axis=0)
    rA = rate_df[A].values
    rB = rate_df[B].values
    # continuity-corrected log-odds-ratio (Haldane-Anscombe), count-based not rate-based
    cc = 0.5
    dir_score = (np.log((countA + cc) / (nA - countA + cc))
                 - np.log((countB + cc) / (nB - countB + cc)))
    df = pd.DataFrame({"gene": sel_genes, "rate_A": rA, "rate_B": rB,
                        "count_A": countA, "count_B": countB, "dir_score": dir_score})
    df["abs_score"] = df["dir_score"].abs()
    df["dominant_count"] = np.where(df["dir_score"] >= 0, df["count_A"], df["count_B"])
    # require real evidence in the class the gene is claimed to favor -- excludes
    # singleton/near-zero-carrier flukes that produce spuriously huge raw rate-ratios
    df_evidenced = df[df["dominant_count"] >= min_dominant_count].copy()
    n_dropped = len(df) - len(df_evidenced)
    print(f"[{A} vs {B}] dropped {n_dropped}/{len(df)} genes with dominant_count < {min_dominant_count} "
          f"(insufficient carriers in the class they'd favor -> unstable rate-ratio artifacts)")
    df_evidenced = df_evidenced.sort_values("abs_score", ascending=False)
    top = df_evidenced.head(topn).copy()

    # fold recompute
    fold_scores = np.zeros((len(top), 5))
    for fi, mask in enumerate(fold_train_masks):
        r_fold = rate_matrix(mask, Msel, y, classes)
        r_fold_df = pd.DataFrame(r_fold, index=sel_genes, columns=classes)
        sub_y_fold = y[mask]
        cA_f = pd.Series((Msel[mask][sub_y_fold == A].sum(axis=0)), index=sel_genes).reindex(top["gene"]).values
        cB_f = pd.Series((Msel[mask][sub_y_fold == B].sum(axis=0)), index=sel_genes).reindex(top["gene"]).values
        nA_f = (sub_y_fold == A).sum(); nB_f = (sub_y_fold == B).sum()
        fold_scores[:, fi] = (np.log((cA_f + cc) / (nA_f - cA_f + cc))
                               - np.log((cB_f + cc) / (nB_f - cB_f + cc)))

    sign_overall = np.sign(top["dir_score"].values)
    sign_folds = np.sign(fold_scores)
    consistent = (sign_folds == sign_overall[:, None]).all(axis=1)
    n_consistent_folds = (sign_folds == sign_overall[:, None]).sum(axis=1)
    top["fold_sign_consistent_5of5"] = consistent
    top["n_folds_same_sign"] = n_consistent_folds
    top["min_fold_score"] = fold_scores.min(axis=1)
    top["max_fold_score"] = fold_scores.max(axis=1)
    top["is_known_driver"] = top["gene"].isin(GENE_INFO.keys())
    return top, fold_scores

top_gbm, fs_gbm = boundary_analysis("GBMLGG", "LGG", topn=50)
top_kip, fs_kip = boundary_analysis("KIPAN", "KIRC", topn=50)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 60)

cols_show = ["gene","count_A","count_B","rate_A","rate_B","dir_score","n_folds_same_sign","fold_sign_consistent_5of5","is_known_driver"]
print("\n\n=== GBMLGG vs LGG top boundary genes (evidence-filtered, count-based log-OR) ===")
print(top_gbm[cols_show].to_string())

print("\n\n=== KIPAN vs KIRC top boundary genes (evidence-filtered, count-based log-OR) ===")
print(top_kip[cols_show].to_string())

# Rank/score of the known "expected axis" genes for sanity-check, regardless of whether
# they made the evidence-filtered top-50 (they should be strong, just not necessarily extreme)
def known_gene_lookup(A, B, gene_list):
    nA = (y == A).sum(); nB = (y == B).sum()
    cc = 0.5
    rows = []
    for g in gene_list:
        if g not in gi_all:
            continue
        j = gi_all[g]
        a = M[y == A, j].sum(); b = M[y == B, j].sum()
        score = (np.log((a + cc) / (nA - a + cc)) - np.log((b + cc) / (nB - b + cc)))
        rows.append(dict(gene=g, count_A=a, count_B=b, rate_A=a/nA, rate_B=b/nB, dir_score=score))
    return pd.DataFrame(rows)

print("\n--- known-driver sanity scores, GBMLGG(A) vs LGG(B) ---")
print(known_gene_lookup("GBMLGG","LGG",["IDH1","ATRX","TP53","EGFR","PTEN"]).to_string(index=False))
print("\n--- known-driver sanity scores, KIPAN(A) vs KIRC(B) ---")
print(known_gene_lookup("KIPAN","KIRC",["VHL","MET"]).to_string(index=False))

top_gbm.to_csv("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/top_gbm_lgg.csv", index=False)
top_kip.to_csv("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/top_kipan_kirc.csv", index=False)

# ---- Step4: duplication check ----
KNOWN_GBM = {"IDH1","ATRX","TP53","EGFR","PTEN"}
KNOWN_KIP = {"VHL","MET"}

# combo genes actually referenced in features.py
COMBO_GENES = set()
for a,b in [("IDH1","TP53"),("IDH1","ATRX"),("ATRX","TP53"),("APC","TP53"),("TP53","CDKN2A"),
            ("TP53","NOTCH1"),("PTEN","PIK3CA"),("PTEN","CTNNB1"),("BRAF","TP53"),("VHL","TP53"),
            ("TP53","NFE2L2"),("CDKN2A","NOTCH1"),("ERCC2","FGFR3"),("KIT","NCOR2")]:
    COMBO_GENES.add(a); COMBO_GENES.add(b)

def new_genes_report(top, known_pair_drivers, pair_name):
    stable = top[top["fold_sign_consistent_5of5"]]
    new_stable = stable[~stable["gene"].isin(known_pair_drivers) & ~stable["gene"].isin(GENE_INFO.keys())]
    print(f"\n\n=== {pair_name}: stable(5/5) genes = {len(stable)} / top50 ===")
    print(f"Of these, NOT in known-pair-driver set AND NOT in GENE_INFO (189 literature genes): {len(new_stable)}")
    if len(new_stable):
        cols = ["gene","rate_A","rate_B","dir_score","carrier_count" if "carrier_count" in new_stable else "gene"]
        merged = new_stable.merge(profile_df[["carrier_count"]], left_on="gene", right_index=True, how="left")
        print(merged[["gene","rate_A","rate_B","dir_score","carrier_count"]].to_string())
        print("\nAlso in COMBO_GENES (features.py)?", new_stable["gene"].isin(COMBO_GENES).tolist())
    return new_stable

new_gbm = new_genes_report(top_gbm, KNOWN_GBM, "GBMLGG/LGG")
new_kip = new_genes_report(top_kip, KNOWN_KIP, "KIPAN/KIRC")

# ---- Step5: coverage check ----
def coverage_check(new_genes_df, class_list, n_sample=100):
    if len(new_genes_df) == 0:
        print("No new genes to check coverage for.")
        return
    idx_rows = train[train["SUBCLASS"].isin(class_list)].index
    sample_idx = np.random.RandomState(42).choice(idx_rows, size=min(n_sample, len(idx_rows)), replace=False)
    ng = new_genes_df["gene"].tolist()
    sub = train.loc[sample_idx, ng]
    has_any = (sub != "WT").any(axis=1)
    n_any = has_any.sum()
    per_gene_counts = (sub != "WT").sum(axis=0)
    print(f"Sample size={len(sample_idx)} from classes {class_list}")
    print(f"Rows carrying >=1 of the {len(ng)} new genes: {n_any} ({n_any/len(sample_idx)*100:.1f}%)")
    print("Per-gene carrier counts in this sample:")
    print(per_gene_counts.sort_values(ascending=False).to_string())

print("\n\n=== Coverage check: GBMLGG/LGG new genes ===")
coverage_check(new_gbm, ["GBMLGG","LGG"], n_sample=100)

print("\n\n=== Coverage check: KIPAN/KIRC new genes ===")
coverage_check(new_kip, ["KIPAN","KIRC"], n_sample=100)

print("\nDONE")
