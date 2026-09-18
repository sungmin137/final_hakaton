"""Stage 1 (seed=42): 4-condition test of EG_resid / max_tok_resid on top of
접근16 (v4s.45/v2.25/v4sp.15/specNB.15). Fold-fit residualization (fit on
80% train portion of each fold, apply to both train+val portions of that fold).
Only v4s component is modified; v2/v4sp/specNB reused from cache (unchanged).
train.csv only.
"""
import sys, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
COMMON = ROOT / "4. src/common"
sys.path.insert(0, str(COMMON))
SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad")
OUT = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")

from main import ROOT as MROOT, SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker, TARGET
from features.features import gene_columns
from approach23_synthetic_aug.spectrum_ref import spectrum_features

EPS = 1e-12

train = load_train()
genes = gene_columns(train)
values = train[genes].to_numpy()
n_rows, n_cols = values.shape

# ---------- raw G, E, max_tok ----------
tok_counts = np.zeros((n_rows, n_cols), dtype=np.int16)
flat = values.ravel()
for idx in np.flatnonzero(flat != "WT"):
    r, c = divmod(idx, n_cols)
    tok_counts[r, c] = len(flat[idx].split(" "))

G = (values != "WT").sum(axis=1).astype(np.int64)
E = tok_counts.sum(axis=1).astype(np.int64)
max_tok = np.zeros(n_rows, dtype=np.float64)
for r in range(n_rows):
    vals = tok_counts[r][tok_counts[r] > 0]
    if len(vals) > 0:
        max_tok[r] = vals.max()
EG = np.where(G > 0, E / np.where(G == 0, 1, G), 0.0)

print(f"G=0 rows: {(G==0).sum()} ({(G==0).mean()*100:.2f}%)")

burden = G.copy()  # burden used elsewhere == n_mut_genes(=G) consistent w/ stage1/2 scripts (values!=WT sum)
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]

groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y, groups))
print(f"seed={SEED} fold sizes: {[len(v) for _, v in folds]}", flush=True)

# ---------- fold-fit residualization ----------
def fit_resid(train_idx, val_idx, raw):
    """Fit raw ~ log(G) on train_idx (G>0 only), apply to train_idx+val_idx.
    G==0 rows get sentinel resid=0."""
    resid = np.zeros(n_rows, dtype=np.float64)
    fit_mask = (G[train_idx] > 0)
    fit_rows = train_idx[fit_mask]
    logG_fit = np.log(G[fit_rows].astype(np.float64))
    y_fit = raw[fit_rows]
    slope, intercept = np.polyfit(logG_fit, y_fit, deg=1)
    for idx_set in (train_idx, val_idx):
        mask = G[idx_set] > 0
        rows = idx_set[mask]
        pred = slope * np.log(G[rows].astype(np.float64)) + intercept
        resid[rows] = raw[rows] - pred
        # G==0 rows in idx_set keep sentinel 0 (already initialized)
    return resid, slope, intercept

oof_EG_resid = np.zeros(n_rows, dtype=np.float64)
oof_maxtok_resid = np.zeros(n_rows, dtype=np.float64)

sp_cache = {}
example_rows = []

v4s_B_oof = np.zeros((n, len(classes)))  # +EG_resid
v4s_C_oof = np.zeros((n, len(classes)))  # +max_tok_resid
v4s_D_oof = np.zeros((n, len(classes)))  # +both

t0 = time.time()
for k, (tri, vai) in enumerate(folds):
    tri = np.asarray(tri); vai = np.asarray(vai)
    eg_resid_fold, eg_slope, eg_int = fit_resid(tri, vai, EG)
    mt_resid_fold, mt_slope, mt_int = fit_resid(tri, vai, max_tok)
    oof_EG_resid[vai] = eg_resid_fold[vai]
    oof_maxtok_resid[vai] = mt_resid_fold[vai]
    print(f"fold{k}: EG~logG slope={eg_slope:.4f} icpt={eg_int:.4f} | "
          f"maxtok~logG slope={mt_slope:.4f} icpt={mt_int:.4f}", flush=True)

    if k == 0:
        for i in list(tri[:3]) + list(vai[:3]):
            example_rows.append({
                "row": int(i), "fold": k, "split": "train" if i in tri else "val",
                "G": int(G[i]), "EG_raw": round(float(EG[i]), 4),
                "EG_resid": round(float(eg_resid_fold[i]), 4),
                "max_tok_raw": float(max_tok[i]),
                "max_tok_resid": round(float(mt_resid_fold[i]), 4),
            })

    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm4 = FeatureMaker("v4").fit(tr_part)
    Xtr_v4, Xva_v4 = fm4.transform(tr_part), fm4.transform(va_part)
    sp_tr = spectrum_features(tr_part, genes).reset_index(drop=True)
    sp_va = spectrum_features(va_part, genes).reset_index(drop=True)
    Xtr_v4s = pd.concat([Xtr_v4.reset_index(drop=True), sp_tr], axis=1)
    Xva_v4s = pd.concat([Xva_v4.reset_index(drop=True), sp_va], axis=1)

    eg_tr = pd.Series(eg_resid_fold[tri], name="EG_resid").reset_index(drop=True)
    eg_va = pd.Series(eg_resid_fold[vai], name="EG_resid").reset_index(drop=True)
    mt_tr = pd.Series(mt_resid_fold[tri], name="max_tok_resid").reset_index(drop=True)
    mt_va = pd.Series(mt_resid_fold[vai], name="max_tok_resid").reset_index(drop=True)

    # B: + EG_resid only
    XtrB = pd.concat([Xtr_v4s, eg_tr], axis=1)
    XvaB = pd.concat([Xva_v4s, eg_va], axis=1)
    mB = xgb.XGBClassifier(**PARAMS).fit(XtrB, y[tri])
    v4s_B_oof[vai] = mB.predict_proba(XvaB)
    print(f"fold{k} v4s+EG_resid done ({time.time()-t0:.0f}s)", flush=True)

    # C: + max_tok_resid only
    XtrC = pd.concat([Xtr_v4s, mt_tr], axis=1)
    XvaC = pd.concat([Xva_v4s, mt_va], axis=1)
    mC = xgb.XGBClassifier(**PARAMS).fit(XtrC, y[tri])
    v4s_C_oof[vai] = mC.predict_proba(XvaC)
    print(f"fold{k} v4s+max_tok_resid done ({time.time()-t0:.0f}s)", flush=True)

    # D: + both
    XtrD = pd.concat([Xtr_v4s, eg_tr, mt_tr], axis=1)
    XvaD = pd.concat([Xva_v4s, eg_va, mt_va], axis=1)
    mD = xgb.XGBClassifier(**PARAMS).fit(XtrD, y[tri])
    v4s_D_oof[vai] = mD.predict_proba(XvaD)
    print(f"fold{k} v4s+both done ({time.time()-t0:.0f}s)", flush=True)

np.save(OUT / "s1_v4s_B_oof.npy", v4s_B_oof)
np.save(OUT / "s1_v4s_C_oof.npy", v4s_C_oof)
np.save(OUT / "s1_v4s_D_oof.npy", v4s_D_oof)
np.save(OUT / "s1_EG_resid_oof.npy", oof_EG_resid)
np.save(OUT / "s1_maxtok_resid_oof.npy", oof_maxtok_resid)
np.save(OUT / "s1_EG_raw.npy", EG)
np.save(OUT / "s1_maxtok_raw.npy", max_tok)
np.save(OUT / "s1_G.npy", G)
np.save(OUT / "s1_y.npy", y)
with open(OUT / "s1_classes.json", "w") as f:
    json.dump(classes, f, ensure_ascii=False)
with open(OUT / "s1_examples.json", "w") as f:
    json.dump(example_rows, f, ensure_ascii=False, indent=2)

print(f"\ntotal elapsed: {time.time()-t0:.0f}s")
print("saved to", OUT)
