import sys
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from features.features import gene_columns, TARGET
from main import twin_groups, load_train

pd.set_option("display.width", 140)
train = load_train()
genes = gene_columns(train)
sub = train[genes]
is_mut = (sub != "WT").to_numpy()
flat = sub.to_numpy().ravel()
n_rows, n_cols = sub.shape

G = is_mut.sum(axis=1)  # n_mut_genes
E = np.zeros(n_rows, dtype=np.int64)  # n_tokens
M = np.zeros(n_rows, dtype=np.int64)  # genes with >=2 tokens per sample
max_tok_per_gene = np.zeros(n_rows, dtype=np.int64)
var_tok_per_gene = np.zeros(n_rows, dtype=np.float64)

# per-row per-gene token counts
tok_counts = np.zeros((n_rows, n_cols), dtype=np.int16)
for idx in np.flatnonzero(flat != "WT"):
    r = idx // n_cols
    c = idx % n_cols
    tok_counts[r, c] = len(flat[idx].split(" "))

E = tok_counts.sum(axis=1)
M = (tok_counts >= 2).sum(axis=1)
# stats among mutated genes only (tok_counts>0)
for r in range(n_rows):
    vals = tok_counts[r][tok_counts[r] > 0]
    if len(vals) > 0:
        max_tok_per_gene[r] = vals.max()
        var_tok_per_gene[r] = vals.var()

assert np.array_equal(G, (sub != "WT").sum(axis=1).to_numpy())

df = pd.DataFrame({
    "SUBCLASS": train[TARGET].values,
    "G": G, "E": E, "M": M,
    "max_tok": max_tok_per_gene, "var_tok": var_tok_per_gene,
})
df["EG"] = np.where(df.G > 0, df.E / df.G.replace(0, np.nan), np.nan)
df["MG"] = np.where(df.G > 0, df.M / df.G.replace(0, np.nan), np.nan)

print("=== STEP1: 원시 통계 ===")
print(f"n_rows={n_rows}, G=0 rows: {(df.G==0).sum()} ({(df.G==0).mean()*100:.2f}%)")
print(df[["G","E","M","EG","MG","max_tok","var_tok"]].describe())
print()

print("=== STEP2: raw class association (uncontrolled) ===")
cls_stats = df.groupby("SUBCLASS").agg(n=("G","size"), G_mean=("G","mean"), EG_mean=("EG","mean"), MG_mean=("MG","mean")).sort_values("EG_mean")
print(cls_stats)
print()

print("=== STEP3a: burden-matched stratified comparison ===")
bins = [(10,20),(20,30),(30,50),(50,100)]
for lo, hi in bins:
    mask = (df.G >= lo) & (df.G < hi)
    sub_df = df[mask]
    counts = sub_df.groupby("SUBCLASS").size().sort_values(ascending=False)
    adequate = counts[counts >= 15].index.tolist()
    print(f"--- burden bin G in [{lo},{hi}) : n={mask.sum()}, classes with >=15 samples: {len(adequate)} ---")
    if len(adequate) >= 2:
        bin_cls = sub_df[sub_df.SUBCLASS.isin(adequate)].groupby("SUBCLASS").agg(
            n=("G","size"), EG_mean=("EG","mean"), EG_std=("EG","std"), MG_mean=("MG","mean")
        ).sort_values("EG_mean")
        print(bin_cls)
    else:
        print("  (too few classes with adequate n, skip)")
    print()

print("=== STEP3b: residualization (E/G and M/G on log(G)) ===")
mask_g = df.G > 0
logG = np.log(df.loc[mask_g, "G"])
def residualize(y, x):
    # simple linear regression residual
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return y - pred, coef

eg_resid, eg_coef = residualize(df.loc[mask_g, "EG"].values, logG.values)
mg_resid, mg_coef = residualize(df.loc[mask_g, "MG"].values, logG.values)
df.loc[mask_g, "EG_resid"] = eg_resid
df.loc[mask_g, "MG_resid"] = mg_resid
print(f"EG ~ logG coef: slope={eg_coef[0]:.4f}, intercept={eg_coef[1]:.4f}")
print(f"MG ~ logG coef: slope={mg_coef[0]:.4f}, intercept={mg_coef[1]:.4f}")

resid_cls = df[mask_g].groupby("SUBCLASS").agg(
    n=("EG_resid","size"), EG_resid_mean=("EG_resid","mean"), MG_resid_mean=("MG_resid","mean")
).sort_values("EG_resid_mean")
print(resid_cls)

pooled_std_eg = df.loc[mask_g, "EG_resid"].std()
pooled_std_mg = df.loc[mask_g, "MG_resid"].std()
between_range_eg = resid_cls.EG_resid_mean.max() - resid_cls.EG_resid_mean.min()
between_range_mg = resid_cls.MG_resid_mean.max() - resid_cls.MG_resid_mean.min()
print(f"\nEG_resid: pooled_std={pooled_std_eg:.4f}, between-class range={between_range_eg:.4f}, ratio={between_range_eg/pooled_std_eg:.3f}")
print(f"MG_resid: pooled_std={pooled_std_mg:.4f}, between-class range={between_range_mg:.4f}, ratio={between_range_mg/pooled_std_mg:.3f}")

# ANOVA-style eta^2 (between var / total var)
from scipy import stats as sstats
groups_eg = [g["EG_resid"].values for _, g in df[mask_g].groupby("SUBCLASS")]
groups_mg = [g["MG_resid"].values for _, g in df[mask_g].groupby("SUBCLASS")]
f_eg, p_eg = sstats.f_oneway(*groups_eg)
f_mg, p_mg = sstats.f_oneway(*groups_mg)
print(f"\nANOVA EG_resid ~ SUBCLASS: F={f_eg:.3f}, p={p_eg:.3e}")
print(f"ANOVA MG_resid ~ SUBCLASS: F={f_mg:.3f}, p={p_mg:.3e}")

# eta^2
grand_mean_eg = df.loc[mask_g,"EG_resid"].mean()
ss_between_eg = sum(len(g)*(g.mean()-grand_mean_eg)**2 for g in groups_eg)
ss_total_eg = sum((df.loc[mask_g,"EG_resid"]-grand_mean_eg)**2)
eta2_eg = ss_between_eg/ss_total_eg
grand_mean_mg = df.loc[mask_g,"MG_resid"].mean()
ss_between_mg = sum(len(g)*(g.mean()-grand_mean_mg)**2 for g in groups_mg)
ss_total_mg = sum((df.loc[mask_g,"MG_resid"]-grand_mean_mg)**2)
eta2_mg = ss_between_mg/ss_total_mg
print(f"eta^2 EG_resid = {eta2_eg:.4f}")
print(f"eta^2 MG_resid = {eta2_mg:.4f}")
print()

print("=== STEP4: fold stability of EG_resid class ranking ===")
y_all = train[TARGET].values
groups_twin = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rankings = []
for k, (tri, vai) in enumerate(skf.split(train, y_all, groups_twin)):
    tr_mask = np.zeros(n_rows, dtype=bool); tr_mask[tri] = True
    fold_mask = tr_mask & mask_g.values
    x = logG.values[mask_g.values[tri[np.isin(np.arange(n_rows), tri)]]] if False else None
    # redo regression fit on this fold's train rows only
    sel = fold_mask
    xg = np.log(df.loc[sel, "G"]).values
    yg = df.loc[sel, "EG"].values
    A = np.vstack([xg, np.ones_like(xg)]).T
    coef, *_ = np.linalg.lstsq(A, yg, rcond=None)
    resid = yg - A @ coef
    tmp = pd.DataFrame({"SUBCLASS": df.loc[sel, "SUBCLASS"].values, "resid": resid})
    cls_mean = tmp.groupby("SUBCLASS")["resid"].mean()
    cls_n = tmp.groupby("SUBCLASS")["resid"].size()
    adequate_cls = cls_n[cls_n >= 10].index
    rank = cls_mean.loc[adequate_cls].sort_values(ascending=False)
    rankings.append(rank)
    print(f"fold{k}: top5 high-EG_resid classes: {list(rank.index[:5])}")
    print(f"        bottom5 low-EG_resid classes: {list(rank.index[-5:])}")

# compute rank correlation across folds for common classes
common_classes = set(rankings[0].index)
for r in rankings[1:]:
    common_classes &= set(r.index)
common_classes = sorted(common_classes)
print(f"\ncommon classes across all folds (n>=10 each fold): {len(common_classes)}")
rank_df = pd.DataFrame({f"fold{k}": r.reindex(common_classes).rank(ascending=False) for k, r in enumerate(rankings)})
print(rank_df)
from itertools import combinations
corrs = []
for i, j in combinations(range(5), 2):
    c = rank_df[f"fold{i}"].corr(rank_df[f"fold{j}"], method="spearman")
    corrs.append(c)
print(f"\npairwise spearman rank correlation across folds: mean={np.mean(corrs):.3f}, min={np.min(corrs):.3f}, max={np.max(corrs):.3f}")
print()

print("=== STEP5: distribution shape (var_tok, max_tok) after burden control ===")
maxtok_resid, _ = residualize(df.loc[mask_g, "max_tok"].values, logG.values)
vartok_resid, _ = residualize(df.loc[mask_g, "var_tok"].values, logG.values)
df.loc[mask_g, "maxtok_resid"] = maxtok_resid
df.loc[mask_g, "vartok_resid"] = vartok_resid
groups_maxtok = [g["maxtok_resid"].values for _, g in df[mask_g].groupby("SUBCLASS")]
groups_vartok = [g["vartok_resid"].values for _, g in df[mask_g].groupby("SUBCLASS")]
f_max, p_max = sstats.f_oneway(*groups_maxtok)
f_var, p_var = sstats.f_oneway(*groups_vartok)
print(f"ANOVA max_tok_resid ~ SUBCLASS: F={f_max:.3f}, p={p_max:.3e}")
print(f"ANOVA var_tok_resid ~ SUBCLASS: F={f_var:.3f}, p={p_var:.3e}")
grand_mean_max = df.loc[mask_g,"maxtok_resid"].mean()
eta2_max = sum(len(g)*(g.mean()-grand_mean_max)**2 for g in groups_maxtok) / sum((df.loc[mask_g,"maxtok_resid"]-grand_mean_max)**2)
grand_mean_var = df.loc[mask_g,"vartok_resid"].mean()
eta2_var = sum(len(g)*(g.mean()-grand_mean_var)**2 for g in groups_vartok) / sum((df.loc[mask_g,"vartok_resid"]-grand_mean_var)**2)
print(f"eta^2 max_tok_resid = {eta2_max:.4f}")
print(f"eta^2 var_tok_resid = {eta2_var:.4f}")
