"""Candidate B: domain-hit-count NB. Reuses PreprocessFeatures' dom_ logic as-is (no reimplementation)."""
import sys, json, time
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from main import load_train, TARGET, twin_groups
from features.features import gene_columns
from approach7_preprocess.preprocess_features import PreprocessFeatures

train = load_train()
genes = gene_columns(train)
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
groups = twin_groups(train)
n = len(train)
oof = np.zeros((n, len(classes)))
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
t0 = time.time()
ndims = []
for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    pp = PreprocessFeatures().fit(tr_part)   # fold-fit only, existing code as-is
    Xtr_full = pp.transform(tr_part)
    Xva_full = pp.transform(va_part)
    dom_cols = [c for c in Xtr_full.columns if c.startswith("dom_")]
    ndims.append(len(dom_cols))
    Xtr = Xtr_full[dom_cols].to_numpy()
    Xva = Xva_full[dom_cols].to_numpy()
    m = MultinomialNB(alpha=0.5).fit(Xtr, y[tri])
    oof[vai] = m.predict_proba(Xva)
    print(f"fold {k}: n_dom_cols={len(dom_cols)} fold_f1={f1_score(y[vai], oof[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)

f1 = f1_score(y, oof.argmax(1), average="macro")
print(f"\nstandalone Domain-hit NB macro F1 = {f1:.4f}")
print(f"avg dims per fold = {np.mean(ndims):.1f}")

OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/"
np.save(OUT+"domnb_oof.npy", oof)
json.dump({"standalone_macro_f1": f1, "avg_dims": float(np.mean(ndims))}, open(OUT+"domnb_result.json","w"))
