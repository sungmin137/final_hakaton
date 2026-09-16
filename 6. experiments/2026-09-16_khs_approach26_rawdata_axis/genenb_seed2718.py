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

SEED2 = 2718
S2DIR = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/"
OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/"

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
    tr_df, va_df = train.iloc[tri], train.iloc[vai]
    Mtr = (tr_df[genes] != "WT").astype(np.uint8)
    keep = Mtr.columns[Mtr.sum(axis=0) > 0]
    Xtr = Mtr[keep].to_numpy()
    Xva = (va_df[keep] != "WT").astype(np.uint8).to_numpy()
    m = MultinomialNB(alpha=0.5).fit(Xtr, y[tri])
    oof[vai] = m.predict_proba(Xva)
    print(f"fold {k} geneNB done, n_cols={len(keep)} ({time.time()-t0:.0f}s)", flush=True)

standalone_f1 = f1_score(y, oof.argmax(1), average="macro")
print(f"\nstandalone Gene-identity MultinomialNB (seed={SEED2}) macro F1 = {standalone_f1:.4f}")
np.save(OUT+"genenb_multi_oof_seed2718.npy", oof)

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
print(f"\n[seed={SEED2}] BASE macro F1 = {base_f1:.4f}")

swap = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(oof,.15))
swap_pred = (swap*scales).argmax(1)
swap_f1 = f1_score(y, swap_pred, average="macro")
print(f"[seed={SEED2}] SWAP(geneNB .15) macro F1 = {swap_f1:.4f}  delta = {swap_f1-base_f1:+.4f}")

add1 = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.10),(oof,.05))
add1_pred = (add1*scales).argmax(1)
add1_f1 = f1_score(y, add1_pred, average="macro")
print(f"[seed={SEED2}] ADD(spec.10+gene.05) macro F1 = {add1_f1:.4f}  delta = {add1_f1-base_f1:+.4f}")

spec_pred_s = spec.argmax(1); gene_pred_s = oof.argmax(1)
sc = spec_pred_s==y; gc = gene_pred_s==y
print(f"\n[seed={SEED2}] 4-quadrant: spec_c&gene_c={(sc&gc).sum()} spec_c&gene_w={(sc&~gc).sum()} spec_w&gene_c={(~sc&gc).sum()} spec_w&gene_w={(~sc&~gc).sum()}")

bw = base_pred != y; sw = swap_pred != y
rescued = bw & ~sw; hurt = ~bw & sw
print(f"[seed={SEED2}] ensemble-level SWAP: rescue={rescued.sum()} hurt={hurt.sum()} net={rescued.sum()-hurt.sum()}")

bwa = base_pred != y; awa = add1_pred != y
rescued_a = bwa & ~awa; hurt_a = ~bwa & awa
print(f"[seed={SEED2}] ensemble-level ADD1: rescue={rescued_a.sum()} hurt={hurt_a.sum()} net={rescued_a.sum()-hurt_a.sum()}")

json.dump({
  "seed": SEED2, "standalone_genenb_f1": standalone_f1,
  "base_f1": base_f1, "swap_f1": swap_f1, "swap_delta": swap_f1-base_f1,
  "add1_f1": add1_f1, "add1_delta": add1_f1-base_f1,
  "rescue_swap_net": int(rescued.sum()-hurt.sum()),
  "rescue_add1_net": int(rescued_a.sum()-hurt_a.sum()),
}, open(OUT+"genenb_seed2718_result.json","w"), indent=2)
