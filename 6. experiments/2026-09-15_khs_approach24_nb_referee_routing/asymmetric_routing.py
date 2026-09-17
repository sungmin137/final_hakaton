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

# 접근16 reconstruction (BASE)
ens = log_blend((v4s, 0.45), (v4p, 0.25), (v4sp, 0.15), (specnb, 0.15))
ens_scaled = ens * scales
ens_pred = ens_scaled.argmax(1)
BASE_F1 = f1_score(y, ens_pred, average="macro")
print(f"[sanity] BASE (접근16) reconstructed OOF macro F1 = {BASE_F1:.5f} (expect ~0.5306-0.5309)")

cls_idx = {c: i for i, c in enumerate(classes)}
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

def analyze_pair_asymmetric(A, B):
    """A = majority side NB is biased toward (helps this direction: BASE said B, NB says A confidently, true=A).
       B = minority side (never touch BASE-predicted-A rows)."""
    iA, iB = cls_idx[A], cls_idx[B]
    gate = (ens_pred == iA) | (ens_pred == iB)
    base_A_mask = gate & (ens_pred == iA)
    base_B_mask = gate & (ens_pred == iB)

    nb_A = specnb[:, iA]
    nb_B = specnb[:, iB]
    conf_A = nb_A / np.maximum(nb_A + nb_B, EPS)  # NB pairwise confidence toward A

    print(f"\n{'='*70}\n=== Pair {A}(A,majority)/{B}(B,minority) ===")
    print(f"gate size: {gate.sum()}  | BASE-pred-A: {base_A_mask.sum()}  | BASE-pred-B: {base_B_mask.sum()}")

    rows = []
    best = None
    for thr in THRESHOLDS:
        swap_mask = base_B_mask & (conf_A > thr)
        n_swap = int(swap_mask.sum())

        fixed = int((swap_mask & (y == iA)).sum())   # was wrong (true=A), now corrected
        broken = int((swap_mask & (y == iB)).sum())  # was correct (true=B), now broken
        other = n_swap - fixed - broken  # truth outside {A,B} entirely (shouldn't really happen but check)
        net = fixed - broken

        new_pred = ens_pred.copy()
        new_pred[swap_mask] = iA
        new_f1 = f1_score(y, new_pred, average="macro")
        delta_f1 = new_f1 - BASE_F1

        rows.append(dict(threshold=thr, n_swap=n_swap, fixed=fixed, broken=broken,
                          other_truth=other, net=net, f1_after=new_f1, delta_f1=delta_f1))
        print(f"thr={thr:.1f}  n_swap={n_swap:4d}  fixed={fixed:4d}  broken={broken:4d}  "
              f"other_truth={other:3d}  net={net:+4d}  macroF1={new_f1:.5f}  ΔF1={delta_f1:+.5f}")

        if delta_f1 > 0 and (best is None or delta_f1 > best['delta_f1']):
            best = rows[-1]

    # structural check: BASE-predicted-A rows never touched by construction (swap_mask always subset of base_B_mask)
    # verify explicitly for the last threshold's swap_mask logic (true for all, since swap_mask &= base_B_mask always)
    touched_A_rows = False
    for thr in THRESHOLDS:
        swap_mask = base_B_mask & (conf_A > thr)
        if (swap_mask & base_A_mask).any():
            touched_A_rows = True
    print(f"\n[구조 확인] BASE-predicted-A 행이 한 번이라도 swap_mask에 포함된 적 있는가? -> {touched_A_rows} (False여야 정상)")

    return dict(pair=(A, B), base_f1=BASE_F1, rows=rows, best=best, touched_A_rows=touched_A_rows)

r1 = analyze_pair_asymmetric("GBMLGG", "LGG")
r2 = analyze_pair_asymmetric("KIPAN", "KIRC")

print("\n\n" + "="*70)
print("=== FINAL SUMMARY ===")
for r in (r1, r2):
    A, B = r['pair']
    print(f"\n{A}/{B}: BASE F1={r['base_f1']:.5f}")
    if r['best']:
        b = r['best']
        print(f"  BEST net-positive threshold: {b['threshold']} -> net={b['net']:+d}, ΔF1={b['delta_f1']:+.5f}, macroF1={b['f1_after']:.5f}")
    else:
        print(f"  NO net-positive (ΔF1>0) threshold found in grid {THRESHOLDS}")
    print(f"  structural check (never touches BASE-A rows): {'PASS' if not r['touched_A_rows'] else 'FAIL'}")
