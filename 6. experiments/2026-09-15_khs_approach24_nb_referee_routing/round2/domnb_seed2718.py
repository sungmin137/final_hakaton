"""Candidate B independent-seed check: domain-hit NB at seed=2718, reusing stage2 v4s/v2/v4sp/specNB OOF
from the prior pathway-audit agent's full independent retrain (5ebeae57 scratchpad)."""
import sys, json, time
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from main import load_train, TARGET, twin_groups
from features.features import gene_columns
from approach7_preprocess.preprocess_features import PreprocessFeatures

SEED2 = 2718
S2DIR = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/"
OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/"

train = load_train()
genes = gene_columns(train)
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
groups = twin_groups(train)
n = len(train)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED2)
folds = list(skf.split(train, y, groups))
print(f"seed={SEED2} fold sizes: {[len(v) for _,v in folds]}", flush=True)

oof = np.zeros((n, len(classes)))
t0 = time.time()
for k, (tri, vai) in enumerate(folds):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    pp = PreprocessFeatures().fit(tr_part)
    Xtr_full = pp.transform(tr_part); Xva_full = pp.transform(va_part)
    dom_cols = [c for c in Xtr_full.columns if c.startswith("dom_")]
    Xtr = Xtr_full[dom_cols].to_numpy(); Xva = Xva_full[dom_cols].to_numpy()
    m = MultinomialNB(alpha=0.5).fit(Xtr, y[tri])
    oof[vai] = m.predict_proba(Xva)
    print(f"fold {k} domNB done, n_dom_cols={len(dom_cols)} ({time.time()-t0:.0f}s)", flush=True)

standalone_f1 = f1_score(y, oof.argmax(1), average="macro")
print(f"\nstandalone Domain-hit NB (seed=2718) macro F1 = {standalone_f1:.4f}")
np.save(OUT+"domnb_oof_seed2718.npy", oof)

# reuse stage2 seed=2718 cached component OOFs
v4s = np.load(S2DIR+"stage2_v4s_oof.npy")
v2  = np.load(S2DIR+"stage2_v2_oof.npy")
v4sp = np.load(S2DIR+"stage2_v4sp_oof.npy")
spec = np.load(S2DIR+"stage2_nb_oof.npy")
assert v4s.shape == v2.shape == v4sp.shape == spec.shape == oof.shape, (v4s.shape, oof.shape)

scale_map = json.load(open("/Users/admin/Desktop/해커톤_암종분류/branch_khs/6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

EPS=1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p,EPS))*w for p,w in parts)
    return softmax(z, axis=1)

base = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15))
base_pred = (base*scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")
print(f"\n[seed=2718] BASE macro F1 = {base_f1:.4f}")

swap = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(oof,.15))
swap_pred = (swap*scales).argmax(1)
swap_f1 = f1_score(y, swap_pred, average="macro")
print(f"[seed=2718] SWAP(domNB) macro F1 = {swap_f1:.4f}  delta = {swap_f1-base_f1:+.4f}")

addw = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.10),(oof,.10))
add_pred = (addw*scales).argmax(1)
add_f1 = f1_score(y, add_pred, average="macro")
print(f"[seed=2718] ADD(spec.10+dom.10) macro F1 = {add_f1:.4f}  delta = {add_f1-base_f1:+.4f}")

spec_pred_s = spec.argmax(1); dom_pred_s = oof.argmax(1)
sc = spec_pred_s==y; dc = dom_pred_s==y
print(f"\n[seed=2718] 4-quadrant: spec_correct&dom_correct={(sc&dc).sum()} spec_correct&dom_wrong={(sc&~dc).sum()} spec_wrong&dom_correct={(~sc&dc).sum()} spec_wrong&dom_wrong={(~sc&~dc).sum()}")

json.dump({
  "seed": SEED2, "standalone_domnb_f1": standalone_f1,
  "base_f1": base_f1, "swap_f1": swap_f1, "swap_delta": swap_f1-base_f1,
  "add_f1": add_f1, "add_delta": add_f1-base_f1,
}, open(OUT+"domnb_seed2718_result.json","w"), indent=2)
