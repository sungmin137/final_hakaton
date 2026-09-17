import sys, json
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from main import load_train, TARGET

SC = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof/"
OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/"

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)

v4s = np.load(SC+"v4s_oof_proba.npy")
v2  = np.load(SC+"v4p_oof_proba.npy")
v4sp = np.load(SC+"v4sp_oof_proba.npy")
spec = np.load(SC+"oof_spectrum_profile_score.npy")
domnb = np.load(OUT+"domnb_oof.npy")
scale_map = json.load(open(SC+"class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

EPS=1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p,EPS))*w for p,w in parts)
    return softmax(z, axis=1)

base = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15))
base_pred = (base*scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")

swap = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(domnb,.15))
swap_pred = (swap*scales).argmax(1)
swap_f1 = f1_score(y, swap_pred, average="macro")

print(f"BASE   macro F1 = {base_f1:.4f}")
print(f"SWAP(domNB) macro F1 = {swap_f1:.4f}  delta = {swap_f1-base_f1:+.4f}")

add = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.10),(domnb,.10))
add_pred = (add*scales).argmax(1)
add_f1 = f1_score(y, add_pred, average="macro")
print(f"ADD(spec.10+dom.10) macro F1 = {add_f1:.4f}  delta = {add_f1-base_f1:+.4f}")

for w in (0.05, 0.08):
    a2 = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15-w),(domnb,w))
    a2p = (a2*scales).argmax(1)
    a2f1 = f1_score(y,a2p,average="macro")
    print(f"ADD(spec.{0.15-w:.2f}+dom.{w:.2f}) macro F1 = {a2f1:.4f}  delta={a2f1-base_f1:+.4f}")

spec_pred_s = spec.argmax(1)
dom_pred_s = domnb.argmax(1)
sc = spec_pred_s == y
dc = dom_pred_s == y
print("\n4-quadrant (standalone spec-NB vs standalone domain-NB):")
print(f"  spec correct & dom correct: {(sc&dc).sum()}")
print(f"  spec correct & dom wrong:   {(sc&~dc).sum()}")
print(f"  spec wrong & dom correct:   {(~sc&dc).sum()}  <- dom rescues where spec fails")
print(f"  spec wrong & dom wrong:     {(~sc&~dc).sum()}")

bw = base_pred != y
sw = swap_pred != y
rescued = bw & ~sw
hurt = ~bw & sw
print(f"\nensemble-level: BASE wrong -> SWAP correct (rescue) = {rescued.sum()}")
print(f"ensemble-level: BASE correct -> SWAP wrong (hurt)    = {hurt.sum()}")
print(f"net = {rescued.sum()-hurt.sum()}")

json.dump({
  "base_f1": base_f1, "swap_f1": swap_f1, "swap_delta": swap_f1-base_f1,
  "add_f1": add_f1, "add_delta": add_f1-base_f1,
  "rescue_ensemble_net": int(rescued.sum()-hurt.sum())
}, open(OUT+"ensemble_b_result.json","w"), indent=2)
