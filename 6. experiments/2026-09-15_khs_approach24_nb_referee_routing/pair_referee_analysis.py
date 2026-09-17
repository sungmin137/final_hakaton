import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

REPO = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad")
OOF = SC / "oof"
PREV = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad")

train = pd.read_csv(REPO / "1. info/data/train.csv")
le = LabelEncoder().fit(train["SUBCLASS"])
classes = list(le.classes_)
y = le.transform(train["SUBCLASS"])
n = len(train)
EPS = 1e-12

def log_blend(*parts):
    z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
    return softmax(z, axis=1)

v4s = np.load(OOF / "2026-09-14_v4s_xgb_mild_col_grp_oof_proba.npy")
v4p = np.load(OOF / "2026-09-13_v4p_xgb_mild_col_grp_oof_proba.npy")
v4sp = np.load(OOF / "2026-09-14_v4sp_xgb_mild_col_grp_oof_proba.npy")
specnb = np.load(PREV / "oof_spec_nb.npy")
y_prev = np.load(PREV / "y.npy")
assert np.array_equal(y, y_prev)

scale_map = json.loads((OOF / "2026-09-09_v4_xgb_cs_class_scales_recovered.json").read_text())
scales = np.array([scale_map[c] for c in classes])

# 접근16 reconstruction
ens = log_blend((v4s, 0.45), (v4p, 0.25), (v4sp, 0.15), (specnb, 0.15))
ens_scaled = ens * scales
ens_pred = ens_scaled.argmax(1)
ens_f1 = f1_score(y, ens_pred, average="macro")
print(f"[sanity] 접근16 reconstructed OOF macro F1 = {ens_f1:.4f} (expect ~0.5306-0.5309)")

cls_idx = {c: i for i, c in enumerate(classes)}

def analyze_pair(A, B):
    iA, iB = cls_idx[A], cls_idx[B]
    gate = (ens_pred == iA) | (ens_pred == iB)
    n_gate = gate.sum()

    # within gate: is truth in {A,B}? (some gate rows might have truth outside pair - unusual but check)
    truth_in_pair = (y == iA) | (y == iB)
    n_gate_truth_outside = int((gate & ~truth_in_pair).sum())

    ens_correct = gate & (ens_pred == y)
    ens_wrong_ab = gate & (ens_pred != y)  # ensemble wrong on gated rows (predicted A or B)

    # NB pairwise preference within {A,B}, renormalized
    nb_A = specnb[:, iA]
    nb_B = specnb[:, iB]
    nb_pref_A = nb_A > nb_B  # True => NB prefers A
    nb_pick = np.where(nb_pref_A, iA, iB)

    # --- errors within gate ---
    err_idx = np.where(ens_wrong_ab)[0]
    n_err = len(err_idx)
    # among these, truth must be in {A,B} for a pairwise swap to have a chance of fixing it
    err_truth_in_pair = truth_in_pair[err_idx]
    n_err_truth_in_pair = int(err_truth_in_pair.sum())
    n_err_truth_outside = n_err - n_err_truth_in_pair

    nb_would_fix = 0
    for i in err_idx:
        if not truth_in_pair[i]:
            continue  # NB pairwise pick is always in {A,B}; can't fix a truth outside the pair
        if nb_pick[i] == y[i]:
            nb_would_fix += 1

    # --- cost side: correct rows in gate ---
    corr_idx = np.where(ens_correct)[0]
    n_corr = len(corr_idx)
    nb_would_break = 0
    for i in corr_idx:
        if nb_pick[i] != y[i]:
            nb_would_break += 1

    net_gain = nb_would_fix - nb_would_break

    # macro F1 delta estimate: apply NB-pairwise swap only within gate set, recompute macro F1
    new_pred = ens_pred.copy()
    new_pred[gate] = nb_pick[gate]
    new_f1 = f1_score(y, new_pred, average="macro")
    delta_f1 = new_f1 - ens_f1

    print(f"\n=== Pair {A}/{B} ===")
    print(f"gate size (ens pred in {{{A},{B}}}): {n_gate}")
    print(f"  of which truth outside pair: {n_gate_truth_outside}")
    print(f"ens correct in gate: {n_corr}")
    print(f"ens wrong in gate (errors): {n_err}  (truth-in-pair: {n_err_truth_in_pair}, truth-outside-pair: {n_err_truth_outside})")
    print(f"NB-pairwise would FIX (of the {n_err_truth_in_pair} truth-in-pair errors): {nb_would_fix}  "
          f"({nb_would_fix/n_err_truth_in_pair*100:.1f}% of fixable errors)" if n_err_truth_in_pair else "  n/a")
    print(f"NB-pairwise would BREAK (of the {n_corr} currently-correct): {nb_would_break} "
          f"({nb_would_break/n_corr*100:.1f}% of correct)" if n_corr else "  n/a")
    print(f"net gain (fixes - breaks) = {nb_would_fix} - {nb_would_break} = {net_gain}")
    print(f"macro F1: baseline={ens_f1:.5f}  after-pairwise-swap-in-gate={new_f1:.5f}  delta={delta_f1:+.5f}")

    return dict(pair=(A, B), n_gate=n_gate, n_err=n_err, n_err_truth_in_pair=n_err_truth_in_pair,
                nb_fix=nb_would_fix, n_corr=n_corr, nb_break=nb_would_break, net_gain=net_gain,
                f1_base=ens_f1, f1_after=new_f1, delta_f1=delta_f1)

r1 = analyze_pair("GBMLGG", "LGG")
r2 = analyze_pair("KIPAN", "KIRC")

print("\n\n=== SUMMARY ===")
for r in (r1, r2):
    print(r)
