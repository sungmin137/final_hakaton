import sys, json, time
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from main import load_train, TARGET, twin_groups
from features.features import gene_columns

train = load_train()
genes = gene_columns(train)
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
groups = twin_groups(train)
n = len(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_bern = np.zeros((n, len(classes)))
oof_multi = np.zeros((n, len(classes)))  # quick comparison variant using same binary matrix as counts (0/1)
t0 = time.time()
ndims = []
for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
    tr_df, va_df = train.iloc[tri], train.iloc[vai]
    Mtr = (tr_df[genes] != "WT").astype(np.uint8)
    keep = Mtr.columns[Mtr.sum(0) > 0]   # fold-fit: drop genes all-WT in training fold (zero-variance, uninformative)
    ndims.append(len(keep))
    Xtr = Mtr[keep].to_numpy()
    Xva = (va_df[keep] != "WT").astype(np.uint8).to_numpy()

    mb = BernoulliNB(alpha=0.5).fit(Xtr, y[tri])
    oof_bern[vai] = mb.predict_proba(Xva)

    mm = MultinomialNB(alpha=0.5).fit(Xtr, y[tri])
    oof_multi[vai] = mm.predict_proba(Xva)

    f1b = f1_score(y[vai], oof_bern[vai].argmax(1), average="macro")
    f1m = f1_score(y[vai], oof_multi[vai].argmax(1), average="macro")
    print(f"fold {k}: n_gene_cols={len(keep)} bernoulli_f1={f1b:.4f} multinomial_f1={f1m:.4f} ({time.time()-t0:.0f}s)", flush=True)

f1_bern = f1_score(y, oof_bern.argmax(1), average="macro")
f1_multi = f1_score(y, oof_multi.argmax(1), average="macro")
print(f"\nstandalone Gene-identity BernoulliNB macro F1 = {f1_bern:.4f}")
print(f"standalone Gene-identity MultinomialNB(binary-as-count) macro F1 = {f1_multi:.4f}")
print(f"avg gene cols kept per fold = {np.mean(ndims):.1f} / {len(genes)} total")

OUT = "/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/"
np.save(OUT+"genenb_bern_oof.npy", oof_bern)
np.save(OUT+"genenb_multi_oof.npy", oof_multi)
np.save(OUT+"y.npy", y)
json.dump({"bernoulli_f1": f1_bern, "multinomial_f1": f1_multi, "avg_gene_cols": float(np.mean(ndims)), "total_genes": len(genes)},
          open(OUT+"genenb_result.json","w"), indent=2)

# burden bin breakdown (bernoulli, the chosen model)
burden = (train[genes] != "WT").sum(1).to_numpy()
bins = [(1, 30), (31, 100), (101, 396), (397, 10**6)]
bin_labels = ["1-30", "31-100", "101-396", "397+"]
pred_bern = oof_bern.argmax(1)
print("\nburden-bin breakdown (BernoulliNB):")
for (lo, hi), lab in zip(bins, bin_labels):
    m = (burden >= lo) & (burden <= hi)
    acc = accuracy_score(y[m], pred_bern[m]) if m.sum() else float('nan')
    print(f"  {lab:<10} n={m.sum():>5} acc={acc:.4f}")

# class-level notable results
from collections import Counter
print("\nper-class F1 (top and bottom 5 by support):")
rows = []
for ci, c in enumerate(classes):
    m = y == ci
    if m.sum() == 0: continue
    f1c = f1_score(y[m], pred_bern[m], average=None, labels=[ci])[0] if m.sum() else float('nan')
    # Use binary f1 for this class across whole set instead:
    f1c = f1_score((y==ci).astype(int), (pred_bern==ci).astype(int))
    rows.append((c, m.sum(), f1c))
rows.sort(key=lambda r: -r[2])
for c, sup, f1c in rows[:5]:
    print(f"  BEST  {c:<10} n={sup:>4} f1={f1c:.3f}")
for c, sup, f1c in rows[-5:]:
    print(f"  WORST {c:<10} n={sup:>4} f1={f1c:.3f}")
