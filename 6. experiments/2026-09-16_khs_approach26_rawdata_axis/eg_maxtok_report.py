import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy import stats
from sklearn.metrics import f1_score

SC5 = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad")
SC9 = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")
ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
EPS = 1e-12

classes = json.loads((SC9 / "s1_classes.json").read_text())
y = np.load(SC9 / "s1_y.npy")
G = np.load(SC9 / "s1_G.npy")
EG_raw = np.load(SC9 / "s1_EG_raw.npy")
maxtok_raw = np.load(SC9 / "s1_maxtok_raw.npy")
EG_resid = np.load(SC9 / "s1_EG_resid_oof.npy")
maxtok_resid = np.load(SC9 / "s1_maxtok_resid_oof.npy")
n = len(y)

# ---------------- base (unchanged) components, seed=42, cached ----------------
v4s_A = np.load(SC5 / "exp/v4s/oof_proba.npy")
v2 = np.load(SC5 / "exp/v4p/oof_proba.npy")
v4sp = np.load(SC5 / "exp/v4sp/oof_proba.npy")
nb = np.load(SC5 / "nb_oof_seed42.npy")
assert v4s_A.shape == v2.shape == v4sp.shape == nb.shape == (n, len(classes))

v4s_B = np.load(SC9 / "s1_v4s_B_oof.npy")
v4s_C = np.load(SC9 / "s1_v4s_C_oof.npy")
v4s_D = np.load(SC9 / "s1_v4s_D_oof.npy")

scale_map = json.loads((ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
scales = np.array([scale_map[c] for c in classes])


def log_blend(*parts):
    z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
    return softmax(z, axis=1)


def build(v4s_variant):
    proba = log_blend((v4s_variant, .45), (v2, .25), (v4sp, .15), (nb, .15))
    pred = (proba * scales).argmax(1)
    return pred


preds = {
    "A_BASE": build(v4s_A),
    "B_EGresid": build(v4s_B),
    "C_maxtokresid": build(v4s_C),
    "D_both": build(v4s_D),
}

f1s = {k: f1_score(y, p, average="macro") for k, p in preds.items()}
print("=== Step0 sanity: BASE macro F1 (expect ~0.5306-0.5309) ===")
print(f1s["A_BASE"])

# ---------------- Step0: fold-fit residualization example rows ----------------
print("\n=== Step0: fold-fit residualization example rows (fold0, first 3 train + 3 val) ===")
examples = json.loads((SC9 / "s1_examples.json").read_text())
print(pd.DataFrame(examples).to_string(index=False))

# ---------------- Step1: class-level raw vs resid summary table ----------------
df = pd.DataFrame({
    "SUBCLASS": [classes[c] for c in y],
    "G": G, "EG_raw": EG_raw, "EG_resid": EG_resid,
    "max_tok_raw": maxtok_raw, "max_tok_resid": maxtok_resid,
})
cls_tbl = df.groupby("SUBCLASS").agg(
    n=("G", "size"),
    EG_raw_mean=("EG_raw", "mean"),
    EG_resid_mean=("EG_resid", "mean"),
    max_tok_raw_mean=("max_tok_raw", "mean"),
    max_tok_resid_mean=("max_tok_resid", "mean"),
).sort_values("EG_resid_mean")
print("\n=== Step1: 클래스별 raw vs residual 요약표 ===")
print(cls_tbl.round(4).to_string())
cls_tbl.round(6).to_csv(SC9 / "step1_class_summary.csv")

# ANOVA-style eta^2 check for resid (sanity vs prior agent's claim)
def eta_squared(values, groups):
    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    ss_between = 0.0
    for g in np.unique(groups):
        vals = values[groups == g]
        ss_between += len(vals) * (vals.mean() - grand_mean) ** 2
    return ss_between / ss_total

eta_eg = eta_squared(EG_resid, y)
eta_mt = eta_squared(maxtok_resid, y)
f_eg, p_eg = stats.f_oneway(*[EG_resid[y == c] for c in range(len(classes))])
f_mt, p_mt = stats.f_oneway(*[maxtok_resid[y == c] for c in range(len(classes))])
print(f"\nEG_resid: eta^2={eta_eg:.4f}, ANOVA p={p_eg:.3e}")
print(f"max_tok_resid: eta^2={eta_mt:.4f}, ANOVA p={p_mt:.3e}")

# ---------------- Step2: A/B/C/D overall + burden-bin + class table ----------------
BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi < 10**9 else '+'}" for lo, hi in BINS]


def macro_by_bin(pred):
    row = {}
    for (lo, hi), nm in zip(BINS, bin_names):
        mask = (G >= lo) & (G <= hi)
        row[nm] = round(f1_score(y[mask], pred[mask], average="macro", labels=sorted(set(y[mask].tolist()))), 4)
        row[nm + "_n"] = int(mask.sum())
    return row


print("\n=== Step2: A/B/C/D overall macro F1 + delta vs A ===")
overall_rows = []
for k, p in preds.items():
    f1 = f1_score(y, p, average="macro")
    overall_rows.append({"condition": k, "macro_f1": round(f1, 6), "delta_vs_A": round(f1 - f1s["A_BASE"], 6)})
overall_df = pd.DataFrame(overall_rows)
print(overall_df.to_string(index=False))
overall_df.to_csv(SC9 / "step2_overall.csv", index=False)

print("\n=== Step2: burden-bin macro F1 (7 bins) ===")
bin_tbl = pd.DataFrame({k: macro_by_bin(p) for k, p in preds.items()}).T
print(bin_tbl.to_string())
bin_tbl.to_csv(SC9 / "step2_burden_bins.csv")

print("\n=== Step2: class-level F1 diffs vs A (worst/best 5) for B, C, D ===")
f1_A_cls = f1_score(y, preds["A_BASE"], average=None, labels=range(len(classes)))
for cond in ["B_EGresid", "C_maxtokresid", "D_both"]:
    f1_cls = f1_score(y, preds[cond], average=None, labels=range(len(classes)))
    diff = pd.Series(f1_cls - f1_A_cls, index=classes).sort_values()
    print(f"\n--- {cond} vs A: worst 5 ---")
    print(diff.head(5))
    print(f"--- {cond} vs A: best 5 ---")
    print(diff.tail(5))
    diff.to_csv(SC9 / f"step2_classdiff_{cond}.csv")

print("\n=== Step2: prediction-change concentration check (rescue/hurt by bin) ===")
for cond in ["B_EGresid", "C_maxtokresid", "D_both"]:
    pred = preds[cond]
    base_pred = preds["A_BASE"]
    changed = pred != base_pred
    rescued = changed & (pred == y) & (base_pred != y)
    hurt = changed & (pred != y) & (base_pred == y)
    print(f"\n--- {cond}: changed={changed.sum()} rescued={rescued.sum()} hurt={hurt.sum()} net={rescued.sum()-hurt.sum()} ---")
    for (lo, hi), nm in zip(BINS, bin_names):
        mask = (G >= lo) & (G <= hi)
        print(f"  bin {nm}: n={mask.sum()}, rescued={(rescued & mask).sum()}, hurt={(hurt & mask).sum()}")

print("\nsaved csvs to", SC9)
