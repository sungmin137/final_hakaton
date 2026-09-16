import sys, json
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from main import load_train, TARGET

SC = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof/"
OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/"

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)

v4s = np.load(SC+"v4s_oof_proba.npy")
v2  = np.load(SC+"v4p_oof_proba.npy")
v4sp = np.load(SC+"v4sp_oof_proba.npy")
spec = np.load(SC+"oof_spectrum_profile_score.npy")
genenb = np.load(OUT+"genenb_multi_oof.npy")
scale_map = json.load(open(SC+"class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

assert v4s.shape == genenb.shape, (v4s.shape, genenb.shape)
y2 = np.load(OUT+"y.npy")
assert np.array_equal(y, y2)

EPS=1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p,EPS))*w for p,w in parts)
    return softmax(z, axis=1)

base = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15))
base_pred = (base*scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")
print(f"BASE macro F1 = {base_f1:.4f}")

swap = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(genenb,.15))
swap_pred = (swap*scales).argmax(1)
swap_f1 = f1_score(y, swap_pred, average="macro")
print(f"SWAP(geneNB .15) macro F1 = {swap_f1:.4f}  delta = {swap_f1-base_f1:+.4f}")

add1 = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.10),(genenb,.05))
add1_pred = (add1*scales).argmax(1)
add1_f1 = f1_score(y, add1_pred, average="macro")
print(f"ADD(spec.10+gene.05) macro F1 = {add1_f1:.4f}  delta = {add1_f1-base_f1:+.4f}")

add2 = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.075),(genenb,.075))
add2_pred = (add2*scales).argmax(1)
add2_f1 = f1_score(y, add2_pred, average="macro")
print(f"ADD(spec.075+gene.075) macro F1 = {add2_f1:.4f}  delta = {add2_f1-base_f1:+.4f}")

# 4-quadrant standalone spec-NB vs gene-NB
spec_pred_s = spec.argmax(1)
gene_pred_s = genenb.argmax(1)
sc = spec_pred_s == y
gc = gene_pred_s == y
print("\n4-quadrant (standalone spec-NB vs standalone gene-NB):")
print(f"  spec correct & gene correct: {(sc&gc).sum()}")
print(f"  spec correct & gene wrong:   {(sc&~gc).sum()}")
print(f"  spec wrong & gene correct:   {(~sc&gc).sum()}  <- gene rescues where spec fails")
print(f"  spec wrong & gene wrong:     {(~sc&~gc).sum()}")

# ensemble-level rescue vs BASE
bw = base_pred != y
sw = swap_pred != y
rescued = bw & ~sw
hurt = ~bw & sw
print(f"\nensemble-level (SWAP vs BASE): BASE wrong -> SWAP correct (rescue) = {rescued.sum()}")
print(f"ensemble-level: BASE correct -> SWAP wrong (hurt)    = {hurt.sum()}")
print(f"net = {rescued.sum()-hurt.sum()}")

# how many of BASE's wrong rows does gene-NB standalone get right (rescue candidates)
cand = bw & gc
print(f"\nBASE wrong rows where gene-NB standalone is correct (rescue candidates) = {cand.sum()} / {bw.sum()} base-wrong")

json.dump({
  "base_f1": base_f1, "swap_f1": swap_f1, "swap_delta": swap_f1-base_f1,
  "add1_f1": add1_f1, "add1_delta": add1_f1-base_f1,
  "add2_f1": add2_f1, "add2_delta": add2_f1-base_f1,
  "rescue_ensemble_net": int(rescued.sum()-hurt.sum()),
  "rescue_candidates_standalone": int(cand.sum()),
  "base_wrong_n": int(bw.sum()),
  "quadrant": {"spec_c_gene_c": int((sc&gc).sum()), "spec_c_gene_w": int((sc&~gc).sum()),
               "spec_w_gene_c": int((~sc&gc).sum()), "spec_w_gene_w": int((~sc&~gc).sum())}
}, open(OUT+"ensemble_result.json","w"), indent=2)
