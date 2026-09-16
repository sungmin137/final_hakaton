import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

REPO = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad")

train = pd.read_csv(REPO / "1. info/data/train.csv")
le = LabelEncoder().fit(train["SUBCLASS"])
y = le.transform(train["SUBCLASS"])

ens_wrong = np.load(SC / "ens_wrong.npy")
wrong_spec = np.load(SC / "wrong_spectrum.npy")
wrong_nb = np.load(SC / "wrong_nb.npy")

layer_dirs = {"gene_identity": SC / "layer_oof/gene_identity", "driver_lit": SC / "layer_oof/driver_lit"}
results = {}
for name, d in layer_dirs.items():
    if not (d / "oof_proba.npy").exists():
        print(f"[skip] {name}: not ready")
        continue
    oof = np.load(d / "oof_proba.npy")
    pred = oof.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    wrong = (pred != y).astype(int)
    results[name] = (f1, wrong)
    print(f"{name}: standalone OOF macroF1={f1:.4f}")

results["spectrum(v4s)"] = (0.5035, wrong_spec)
results["NB(spec-NB)"] = (0.2393, wrong_nb)

def phi(a, b):
    return matthews_corrcoef(a, b)

print("\n--- phi vs ensemble ---")
for name, (f1, wrong) in results.items():
    print(f"{name}: phi vs ens = {phi(wrong, ens_wrong):.4f}")

print("\n--- pairwise phi among 4 layers ---")
names = list(results.keys())
for i, a in enumerate(names):
    for b in names[i+1:]:
        print(f"phi({a}, {b}) = {phi(results[a][1], results[b][1]):.4f}")
