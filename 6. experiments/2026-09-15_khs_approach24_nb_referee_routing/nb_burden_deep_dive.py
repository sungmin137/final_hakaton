"""접근16 NB burden-dependent weighting + 397+ 메커니즘 딥다이브.
train.csv ONLY. test.csv는 어떤 형태로도 열지 않는다. 새 feature/model 구현 없음 (분석 전용).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

REPO = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
SC5 = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad")
SC1 = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad")
SRC = SC5 / "src_sungmin" / "4. src" / "common"
sys.path.insert(0, str(SRC))
from main import twin_groups  # noqa: E402
from features.features import TARGET, gene_columns  # noqa: E402

train = pd.read_csv(REPO / "1. info/data/train.csv")  # train ONLY
genes = gene_columns(train)
values = train[genes].to_numpy()

le = LabelEncoder().fit(train[TARGET])
classes = list(le.classes_)
y = le.transform(train[TARGET])
n = len(y)
y_cached = np.load(SC5 / "y.npy")
assert (y == y_cached).all()
burden = np.load(SC5 / "burden.npy")
assert (burden == (values != "WT").sum(axis=1)).all()

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

def spectrum_counts(values: np.ndarray) -> np.ndarray:
    out = np.zeros((len(values), 383), dtype=np.float32)
    for patient, row in enumerate(values):
        for raw in row[row != "WT"]:
            for token in raw.split():
                if token.endswith("*") or "fs" in token:
                    out[patient, 380] += 1; continue
                m = MISSENSE.match(token)
                if m is None:
                    out[patient, 382] += 1; continue
                before, after = m.group(1), m.group(3)
                if before == after:
                    out[patient, 381] += 1
                elif before in AA_INDEX and after in AA_INDEX:
                    offset = AA_INDEX[before] * 19 + (AA_INDEX[after] - (AA_INDEX[after] > AA_INDEX[before]))
                    out[patient, offset] += 1
                else:
                    out[patient, 382] += 1
    return out

X_nb = spectrum_counts(values)
n_missense_tokens = X_nb[:, :380].sum(axis=1)

groups = twin_groups(train)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(train, y, groups=groups))

EPS = 1e-9
def log_blend(parts):
    z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
    return softmax(z, axis=1)

scale_map = json.loads((SC5 / "exp/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
scales = np.array([scale_map[c] for c in classes])
v4s = np.load(SC5 / "exp/2026-09-14_v4s_xgb_mild_col_grp/oof_proba.npy")
v2 = np.load(SC5 / "exp/2026-09-13_v4p_xgb_mild_col_grp/oof_proba.npy")   # v4p
v4sp = np.load(SC5 / "exp/2026-09-14_v4sp_xgb_mild_col_grp/oof_proba.npy")
specnb = np.load(SC5 / "oof_spec_nb.npy")

# ---- Sanity: reconstruct 접근16 ----
ens16 = log_blend([(v4s, .45), (v2, .25), (v4sp, .15), (specnb, .15)])
pred16 = (ens16 * scales).argmax(1)
f1_16 = f1_score(y, pred16, average="macro")
print(f"[sanity] 접근16 reconstructed OOF macro F1 = {f1_16:.4f} (expect ~0.5306-0.5309)")

nb_pred = specnb.argmax(1)
nb_acc_all = accuracy_score(y, nb_pred)
print(f"[sanity] NB standalone OOF acc = {nb_acc_all:.4f}, macroF1 = {f1_score(y, nb_pred, average='macro'):.4f}")

# =========================================================================
# PART A: burden-dependent NB weighting, 2-region only (<=30 vs >30)
# =========================================================================
print("\n" + "=" * 100)
print("PART A: 2-region NB weight candidates")
print("=" * 100)

le_mask = burden <= 30
gt_mask = burden > 30

def two_region_blend(w_lo, w_hi):
    """apply NB weight w_lo for burden<=30, w_hi for burden>30; rescale other 3 weights proportionally
    keeping their RELATIVE ratio (.45:.25:.15 -> sum .85) so total weight = 1 in each region."""
    pred = np.zeros(n, dtype=int)
    f1_region = {}
    for mask, w_nb, lab in [(le_mask, w_lo, "<=30"), (gt_mask, w_hi, ">30")]:
        rem = 1 - w_nb
        base_sum = 0.45 + 0.25 + 0.15  # = 0.85
        w_v4s = 0.45 / base_sum * rem
        w_v2 = 0.25 / base_sum * rem
        w_v4sp = 0.15 / base_sum * rem
        z = (np.log(np.maximum(v4s[mask], EPS)) * w_v4s +
             np.log(np.maximum(v2[mask], EPS)) * w_v2 +
             np.log(np.maximum(v4sp[mask], EPS)) * w_v4sp +
             np.log(np.maximum(specnb[mask], EPS)) * w_nb)
        prob = softmax(z, axis=1) * scales
        pred[mask] = prob.argmax(1)
        f1_region[lab] = f1_score(y[mask], pred[mask], average="macro")
    overall = f1_score(y, pred, average="macro")
    return overall, f1_region, pred

candidates = [
    ("BASE(.15/.15)", 0.15, 0.15),
    ("<=30:.10 / >30:.20", 0.10, 0.20),
    ("<=30:.10 / >30:.25", 0.10, 0.25),
    ("<=30:.05 / >30:.25", 0.05, 0.25),
    ("<=30:.15 / >30:.25", 0.15, 0.25),
]

f1_16_le = f1_score(y[le_mask], pred16[le_mask], average="macro")
f1_16_gt = f1_score(y[gt_mask], pred16[gt_mask], average="macro")

rows = []
for name, wlo, whi in candidates:
    overall, f1_region, pred = two_region_blend(wlo, whi)
    d_overall = overall - f1_16
    d_le = f1_region["<=30"] - f1_16_le
    d_gt = f1_region[">30"] - f1_16_gt
    rows.append((name, overall, d_overall, d_le, d_gt))
    print(f"{name:24s} overall={overall:.4f} (Δ{d_overall:+.4f})  <=30 Δ{d_le:+.4f}  >30 Δ{d_gt:+.4f}")

print("\n| weight rule | Overall OOF | Δ vs 접근16 | ≤30 구간 Δ | >30 구간 Δ | 판정 |")
print("|---|---|---|---|---|---|")
for name, overall, d_overall, d_le, d_gt in rows:
    verdict = "NO-GO" if d_overall <= 0.0005 else ("주의-검토" if d_overall <= 0.002 else "후보")
    print(f"| {name} | {overall:.4f} | {d_overall:+.4f} | {d_le:+.4f} | {d_gt:+.4f} | {verdict} |")

np.save(SC1 / "partA_pred16.npy", pred16)

# =========================================================================
# PART B: 397+ mechanism deep dive
# =========================================================================
print("\n" + "=" * 100)
print("PART B: 397+ mechanism deep dive")
print("=" * 100)

# ---- B-1: census ----
b397 = burden >= 397
print(f"\n[B-1] 397+ N = {b397.sum()} / {n} ({b397.sum()/n*100:.2f}%)")

subclass = train[TARGET].values
b397_classes = pd.Series(subclass[b397]).value_counts()
print("\nPer-class breakdown (397+):")
print(f"{'class':<10}{'n':>6}{'NB acc':>10}{'v4s acc':>10}{'ens16 acc':>12}{'NB margin(med)':>16}")

margin = np.zeros(n, dtype=np.float64)
for k, (tri, vai) in enumerate(splits):
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_nb[tri], y[tri])
    logpost = nb.predict_log_proba(X_nb[vai])
    sorted_lp = np.sort(logpost, axis=1)
    margin[vai] = sorted_lp[:, -1] - sorted_lp[:, -2]
assert (specnb.argmax(1) == nb_pred).all()

v4s_pred = v4s.argmax(1)
class_rows = []
for cls in classes:
    ci = le.transform([cls])[0]
    m = b397 & (y == ci)
    if m.sum() == 0:
        continue
    nb_acc = accuracy_score(y[m], nb_pred[m])
    v4s_acc = accuracy_score(y[m], v4s_pred[m])
    ens_acc = accuracy_score(y[m], pred16[m])
    med_margin = np.median(margin[m])
    class_rows.append((cls, m.sum(), nb_acc, v4s_acc, ens_acc, med_margin))
    print(f"{cls:<10}{m.sum():>6}{nb_acc:>10.3f}{v4s_acc:>10.3f}{ens_acc:>12.3f}{med_margin:>16.2f}")

class_rows.sort(key=lambda r: -r[1])
top_share = sum(r[1] for r in class_rows[:2]) / b397.sum()
print(f"\nTop-2 classes by N account for {top_share*100:.1f}% of 397+ samples "
      f"(concentration check: {'CONCENTRATED' if top_share > 0.5 else 'BROAD'})")

overall_nb_acc_397 = accuracy_score(y[b397], nb_pred[b397])
print(f"\noverall NB acc in 397+ = {overall_nb_acc_397:.4f} (context claims 63.9%)")

# ---- burden bin table for B-3 comparisons ----
bins = [(1, 30), (31, 100), (101, 396), (397, 10**6)]
bin_labels = ["1-30", "31-100", "101-396", "397+"]

print(f"\n{'bin':<10}{'n':>8}{'NB acc':>10}{'NB margin(mean)':>18}{'correct margin':>16}{'wrong margin':>14}")
bin_table = []
for (lo, hi), lab in zip(bins, bin_labels):
    m = (burden >= lo) & (burden <= hi)
    nb_acc = accuracy_score(y[m], nb_pred[m])
    mean_margin = margin[m].mean()
    corr_m = m & (nb_pred == y)
    wrong_m = m & (nb_pred != y)
    corr_margin = margin[corr_m].mean() if corr_m.sum() else float('nan')
    wrong_margin = margin[wrong_m].mean() if wrong_m.sum() else float('nan')
    bin_table.append((lab, m.sum(), nb_acc, mean_margin, corr_margin, wrong_margin, corr_m.sum(), wrong_m.sum()))
    print(f"{lab:<10}{m.sum():>8}{nb_acc:>10.3f}{mean_margin:>18.3f}{corr_margin:>16.3f}{wrong_margin:>14.3f}")

# ---- B-4: margin direction check, 397+ correct vs incorrect ----
print("\n[B-4] 397+ margin: correct vs incorrect NB predictions")
corr_397 = b397 & (nb_pred == y)
wrong_397 = b397 & (nb_pred != y)
print(f"correct n={corr_397.sum()}, mean margin={margin[corr_397].mean():.3f}, median={np.median(margin[corr_397]):.3f}")
print(f"wrong   n={wrong_397.sum()}, mean margin={margin[wrong_397].mean():.3f}, median={np.median(margin[wrong_397]):.3f}")
if wrong_397.sum() >= 3 and corr_397.sum() >= 3:
    u, p = mannwhitneyu(margin[corr_397], margin[wrong_397], alternative='greater')
    print(f"MWU(correct > wrong) p={p:.4g}")
frac_wrong_high_margin = None
if wrong_397.sum() > 0:
    thresh = np.median(margin[corr_397])
    frac_wrong_high_margin = (margin[wrong_397] >= thresh).mean()
    print(f"fraction of WRONG 397+ preds with margin >= correct-median({thresh:.2f}): {frac_wrong_high_margin*100:.1f}%")

# =========================================================================
# B-2/B-3: evidence quantity vs signature clarity -- random subsampling of
# missense tokens for 397+ rows, refit-free (use SAME fold-fit NB models)
# =========================================================================
print("\n" + "=" * 100)
print("[B-2/B-3] Evidence-quantity vs signature-clarity: token subsampling probe")
print("=" * 100)

rng = np.random.default_rng(42)
fold_of = np.zeros(n, dtype=int)
nb_models = {}
for k, (tri, vai) in enumerate(splits):
    fold_of[vai] = k
    nbk = MultinomialNB(alpha=0.5)
    nbk.fit(X_nb[tri], y[tri])
    nb_models[k] = nbk

idx_397 = np.where(b397)[0]
sub_idx = rng.choice(idx_397, size=min(300, len(idx_397)), replace=False)

subsample_sizes = [20, 50, 100, 200, "full"]
results_sub = {sz: {"acc": [], "margin": []} for sz in subsample_sizes}

for i in sub_idx:
    k = fold_of[i]
    nbk = nb_models[k]
    full_counts = X_nb[i, :380].astype(np.int64)  # missense-only counts (380 dims)
    total_tok = int(full_counts.sum())
    tokens = np.repeat(np.arange(380), full_counts)  # expand to token list
    for sz in subsample_sizes:
        if sz == "full":
            vec_missense = full_counts.astype(np.float32)
        else:
            if total_tok <= sz:
                vec_missense = full_counts.astype(np.float32)  # can't subsample below available
            else:
                chosen = rng.choice(tokens, size=sz, replace=False)
                vec_missense = np.bincount(chosen, minlength=380).astype(np.float32)
        vec = np.zeros(383, dtype=np.float32)
        vec[:380] = vec_missense
        vec[380:] = X_nb[i, 380:]  # keep LoF/syn/other unchanged (not subsampled -- missense-only probe)
        logpost = nbk.predict_log_proba(vec.reshape(1, -1))[0]
        pred_i = logpost.argmax()
        sorted_lp = np.sort(logpost)
        mgn = sorted_lp[-1] - sorted_lp[-2]
        results_sub[sz]["acc"].append(int(pred_i == y[i]))
        results_sub[sz]["margin"].append(mgn)

print(f"\n{'subsample size':<18}{'n':>6}{'accuracy':>12}{'mean margin':>14}{'median margin':>16}")
for sz in subsample_sizes:
    accs = results_sub[sz]["acc"]
    mgns = results_sub[sz]["margin"]
    print(f"{str(sz):<18}{len(accs):>6}{np.mean(accs):>12.3f}{np.mean(mgns):>14.3f}{np.median(mgns):>16.3f}")

# =========================================================================
# B-5/B-6: contribution concentration across burden bins
# =========================================================================
print("\n" + "=" * 100)
print("[B-5/B-6] Per-feature contribution concentration to NB margin")
print("=" * 100)

def contribution_concentration(idx_list):
    """For each sample i (with its fold-fit NB model), compute per-feature contribution to the
    top1-vs-top2 log margin: contrib_j = count_j * (log p(feature_j|top1) - log p(feature_j|top2)).
    Returns arrays of top1/top5/top10/top20 share of total |margin| and a concentration measure."""
    shares_top1, shares_top5, shares_top10, shares_top20, gini_list = [], [], [], [], []
    for i in idx_list:
        k = fold_of[i]
        nbk = nb_models[k]
        logpost = nbk.predict_log_proba(X_nb[i].reshape(1, -1))[0]
        order = np.argsort(logpost)
        top1_c, top2_c = order[-1], order[-2]
        # feature_log_prob_ shape (n_classes, n_features)
        flp = nbk.feature_log_prob_
        diff = flp[top1_c] - flp[top2_c]  # per-feature log-prob diff
        contrib = X_nb[i] * diff  # per-feature contribution to margin (sums to exact margin)
        total_margin = contrib.sum()
        abs_contrib = np.abs(contrib)
        order_c = np.argsort(-abs_contrib)
        cum = np.cumsum(abs_contrib[order_c])
        tot_abs = abs_contrib.sum()
        if tot_abs <= 1e-9:
            continue
        shares_top1.append(cum[0] / tot_abs if len(cum) > 0 else np.nan)
        shares_top5.append(cum[min(4, len(cum)-1)] / tot_abs)
        shares_top10.append(cum[min(9, len(cum)-1)] / tot_abs)
        shares_top20.append(cum[min(19, len(cum)-1)] / tot_abs)
        # Gini coefficient of abs contributions as concentration measure
        sorted_abs = np.sort(abs_contrib)
        nfeat = len(sorted_abs)
        cum_abs = np.cumsum(sorted_abs)
        gini = (nfeat + 1 - 2 * np.sum(cum_abs) / cum_abs[-1]) / nfeat if cum_abs[-1] > 0 else np.nan
        gini_list.append(gini)
    return (np.array(shares_top1), np.array(shares_top5), np.array(shares_top10),
            np.array(shares_top20), np.array(gini_list))

print(f"\n{'bin':<10}{'n':>6}{'top1 share':>12}{'top5 share':>12}{'top10 share':>13}{'top20 share':>13}{'gini':>8}")
concentration_by_bin = {}
for (lo, hi), lab in zip(bins, bin_labels):
    m = (burden >= lo) & (burden <= hi)
    idx = np.where(m)[0]
    if len(idx) > 400:
        idx = rng.choice(idx, size=400, replace=False)
    s1, s5, s10, s20, gini = contribution_concentration(idx)
    concentration_by_bin[lab] = (s1.mean(), s5.mean(), s10.mean(), s20.mean(), np.nanmean(gini))
    print(f"{lab:<10}{len(idx):>6}{s1.mean():>12.3f}{s5.mean():>12.3f}{s10.mean():>13.3f}{s20.mean():>13.3f}{np.nanmean(gini):>8.3f}")

# ---- B-7: is 397 a real boundary? finer bins ----
print("\n[B-7] Finer subdivision of 397+ to test boundary vs continuous trend")
finer_bins = [(101, 200), (201, 396), (397, 500), (501, 700), (701, 10**6)]
finer_labels = ["101-200", "201-396", "397-500", "501-700", "701+"]
for (lo, hi), lab in zip(finer_bins, finer_labels):
    m = (burden >= lo) & (burden <= hi)
    if m.sum() == 0:
        print(f"{lab:<10} n=0 (empty)")
        continue
    nb_acc = accuracy_score(y[m], nb_pred[m]) if m.sum() else float('nan')
    mean_margin = margin[m].mean()
    print(f"{lab:<10} n={m.sum():<6} NB acc={nb_acc:.3f}  mean margin={mean_margin:.2f}")

# =========================================================================
# B-8: NB vs v4s rescue structure, overall and within 397+
# =========================================================================
print("\n" + "=" * 100)
print("[B-8] NB-vs-v4s(XGB) rescue structure")
print("=" * 100)

def rescue_table(mask, label):
    nb_c = (nb_pred == y) & mask
    xgb_c = (v4s_pred == y) & mask
    both_c = nb_c & xgb_c
    nb_only = nb_c & ~xgb_c
    xgb_only = xgb_c & ~nb_c
    both_w = mask & ~nb_c & ~xgb_c
    print(f"\n-- {label} (n={mask.sum()}) --")
    print(f"{'group':<20}{'n':>6}{'mean burden':>13}{'NB margin':>12}")
    for gname, gm in [("both correct", both_c), ("NB-only correct", nb_only),
                       ("XGB-only correct", xgb_only), ("both wrong", both_w)]:
        if gm.sum() == 0:
            print(f"{gname:<20}{0:>6}")
            continue
        print(f"{gname:<20}{gm.sum():>6}{burden[gm].mean():>13.1f}{margin[gm].mean():>12.2f}")
    return dict(both_c=both_c.sum(), nb_only=nb_only.sum(), xgb_only=xgb_only.sum(), both_w=both_w.sum())

all_mask = np.ones(n, dtype=bool)
rescue_table(all_mask, "전체 (train)")
rescue_table(b397, "397+ 전용")

print("\n[DONE] all sections complete.")
