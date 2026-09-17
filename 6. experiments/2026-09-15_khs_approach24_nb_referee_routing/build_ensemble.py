import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

REPO = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad")
OOF = SC / "oof"

train = pd.read_csv(REPO / "1. info/data/train.csv")
le = LabelEncoder().fit(train["SUBCLASS"])
y = le.transform(train["SUBCLASS"])
n, classes = len(train), len(le.classes_)
EPS = 1e-12

def log_blend(*parts):
    z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
    return softmax(z, axis=1)

v4s = np.load(OOF / "2026-09-14_v4s_xgb_mild_col_grp_oof_proba.npy")
v4p = np.load(OOF / "2026-09-13_v4p_xgb_mild_col_grp_oof_proba.npy")   # = v2
v4sp = np.load(OOF / "2026-09-14_v4sp_xgb_mild_col_grp_oof_proba.npy")
spec = np.load(OOF / "2026-09-15_spec_xgb_mild_col_grp_oof_proba.npy")

# spec-NB from prior session scratchpad (reuse)
PREV = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad")
specnb = np.load(PREV / "oof_spec_nb.npy")
y_prev = np.load(PREV / "y.npy")
assert np.array_equal(y, y_prev), "label order mismatch between sessions!"

assert v4s.shape == v4p.shape == v4sp.shape == specnb.shape == (n, classes)

scale_map = json.loads((OOF / "2026-09-09_v4_xgb_cs_class_scales_recovered.json").read_text())
scales = np.array([scale_map[c] for c in le.classes_])

# 접근16 reconstruction: v4s(.45) / v2=v4p(.25) / v4sp(.15) / spec-NB(.15) log-blend + class-scale recovery
ens = log_blend((v4s, 0.45), (v4p, 0.25), (v4sp, 0.15), (specnb, 0.15))
ens_pred = (ens * scales).argmax(1)
ens_f1 = f1_score(y, ens_pred, average="macro")
print(f"[sanity] 접근16 reconstructed ensemble OOF macro F1 = {ens_f1:.4f} (expect ~0.5306-0.5309)")
ens_wrong = (ens_pred != y).astype(int)

def standalone(name, proba, use_scale=True):
    pred = (proba * scales).argmax(1) if use_scale else proba.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    wrong = (pred != y).astype(int)
    return f1, wrong

layers = {}
f1_spec, wrong_spec = standalone("spectrum(v4s)", v4s)
f1_nb, wrong_nb = standalone("NB(spec-NB)", specnb, use_scale=False)  # NB probs not scale-fit; try both
f1_nb_raw, wrong_nb_raw = standalone("NB(spec-NB) no-scale-check", specnb, use_scale=False)

layers["spectrum(v4s)"] = (f1_spec, wrong_spec)
layers["NB(spec-NB)"] = (f1_nb, wrong_nb)

print(f"\nspectrum(v4s) standalone OOF macroF1 = {f1_spec:.4f}")
print(f"NB(spec-NB) standalone OOF macroF1 (no scale) = {f1_nb:.4f}")
f1_nb_sc, wrong_nb_sc = standalone("NB scaled", specnb, use_scale=True)
print(f"NB(spec-NB) standalone OOF macroF1 (with scale) = {f1_nb_sc:.4f}")

def phi(wrong_a, wrong_b):
    return matthews_corrcoef(wrong_a, wrong_b)  # MCC of binary vectors == phi coefficient

print("\nphi(spectrum, ensemble) =", round(phi(wrong_spec, ens_wrong), 4))
print("phi(NB no-scale, ensemble) =", round(phi(wrong_nb, ens_wrong), 4))
print("phi(NB scaled, ensemble) =", round(phi(wrong_nb_sc, ens_wrong), 4))

np.save(SC / "ens_wrong.npy", ens_wrong)
np.save(SC / "y_check.npy", y)
np.save(SC / "wrong_spectrum.npy", wrong_spec)
np.save(SC / "wrong_nb.npy", wrong_nb_sc)
print("\nsaved ens_wrong.npy, wrong_spectrum.npy, wrong_nb.npy, y_check.npy")
