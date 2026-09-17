"""4개 packed-gene indicator(PTEN_multi/LRIG1_multi/APC_multi/MAP3K1_multi)를
v4s에 추가해 refit -> 접근16 BASE 재구성 후 결합 처치 + 개별 ablation.
train.csv only. StratifiedGroupKFold(5, shuffle=True, random_state=SEED) on twin_groups(train).
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
COMMON = ROOT / "4. src/common"
sys.path.insert(0, str(COMMON))
OUT = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")
SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof")

from main import SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker, TARGET
from features.features import gene_columns
from approach23_synthetic_aug.spectrum_ref import spectrum_features

GENES4 = ["PTEN", "LRIG1", "APC", "MAP3K1"]

train = load_train()
genes = gene_columns(train)
values = train[genes].to_numpy()
gene_idx = {g: i for i, g in enumerate(genes)}
burden = (values != "WT").sum(axis=1)

# build the 4 indicator columns once (deterministic row-level transform, no fold-fitting needed)
ind_df = pd.DataFrame(index=train.index)
for g in GENES4:
    col = values[:, gene_idx[g]]
    multi = np.array([1 if (str(cell) != "WT" and len(str(cell).split()) >= 2) else 0 for cell in col], dtype=np.int8)
    ind_df[f"{g}_multi"] = multi
    print(f"sanity: {g}_multi positive = {int(multi.sum())}")

le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]

groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y, groups))
print(f"seed={SEED} fold sizes: {[len(v) for _, v in folds]}", flush=True)

RUNS = ["combined"] + [f"only_{g}" for g in GENES4]
oofs = {name: np.zeros((n, len(classes))) for name in RUNS}

t0 = time.time()
for k, (tri, vai) in enumerate(folds):
    tri = np.asarray(tri); vai = np.asarray(vai)
    tr_part, va_part = train.iloc[tri], train.iloc[vai]

    fm4 = FeatureMaker("v4").fit(tr_part)
    Xtr_v4, Xva_v4 = fm4.transform(tr_part), fm4.transform(va_part)
    sp_tr = spectrum_features(tr_part, genes).reset_index(drop=True)
    sp_va = spectrum_features(va_part, genes).reset_index(drop=True)
    Xtr_v4s = pd.concat([Xtr_v4.reset_index(drop=True), sp_tr], axis=1)
    Xva_v4s = pd.concat([Xva_v4.reset_index(drop=True), sp_va], axis=1)

    ind_tr = ind_df.iloc[tri].reset_index(drop=True)
    ind_va = ind_df.iloc[vai].reset_index(drop=True)

    for name in RUNS:
        if name == "combined":
            cols = [f"{g}_multi" for g in GENES4]
        else:
            g = name.replace("only_", "")
            cols = [f"{g}_multi"]
        Xtr = pd.concat([Xtr_v4s, ind_tr[cols]], axis=1)
        Xva = pd.concat([Xva_v4s, ind_va[cols]], axis=1)
        model = xgb.XGBClassifier(**PARAMS).fit(Xtr, y[tri])
        oofs[name][vai] = model.predict_proba(Xva)

    f1s = {name: f1_score(y[vai], oofs[name][vai].argmax(1), average="macro") for name in RUNS}
    print(f"fold{k}: " + " ".join(f"{n_}={v:.4f}" for n_, v in f1s.items()) + f" ({time.time()-t0:.0f}s)", flush=True)

for name in RUNS:
    np.save(OUT / f"v4s_{name}_oof.npy", oofs[name])

# ---- reconstruct 접근16 with each v4s-variant replacing plain v4s ----
v2 = np.load(SC / "v4p_oof_proba.npy")
v4sp = np.load(SC / "v4sp_oof_proba.npy")
spec = np.load(SC / "oof_spectrum_profile_score.npy")
scale_map = json.loads((SC / "class_scales_recovered.json").read_text())
scales = np.array([scale_map[c] for c in classes])

EPS = 1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
    return softmax(z, axis=1)

def burden_report(y_, pred_, burden_):
    bands = {"1_3": (burden_ >= 1) & (burden_ <= 3), "4_7": (burden_ >= 4) & (burden_ <= 7),
             "8_15": (burden_ >= 8) & (burden_ <= 15), "16_30": (burden_ >= 16) & (burden_ <= 30),
             "31_100": (burden_ >= 31) & (burden_ <= 100), "101_396": (burden_ >= 101) & (burden_ <= 396),
             "397_plus": burden_ >= 397}
    out = {}
    for name, mask in bands.items():
        if mask.sum() == 0:
            continue
        out[name] = {"n": int(mask.sum()), "f1": round(float(f1_score(y_[mask], pred_[mask], average="macro")), 5)}
    return out

base_blend = log_blend((oofs.get("_placeholder_", None) if False else np.load(SC / "v4s_oof_proba.npy"), .45), (v2, .25), (v4sp, .15), (spec, .15))
base_pred = (base_blend * scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")
print(f"\nBASE (plain v4s) macro F1 = {base_f1:.6f}")

report = {"seed": SEED, "base_f1": float(base_f1), "runs": {}}
for name in RUNS:
    blend = log_blend((oofs[name], .45), (v2, .25), (v4sp, .15), (spec, .15))
    pred = (blend * scales).argmax(1)
    f1 = f1_score(y, pred, average="macro")
    changed = pred != base_pred
    rescue = int((changed & (pred == y) & (base_pred != y)).sum())
    hurt = int((changed & (pred != y) & (base_pred == y)).sum())
    entry = {
        "macro_f1": round(float(f1), 6),
        "delta_vs_base": round(float(f1 - base_f1), 6),
        "changed": int(changed.sum()),
        "rescue": rescue,
        "hurt": hurt,
        "net": rescue - hurt,
        "burden": burden_report(y, pred, burden),
    }
    per_class = f1_score(y, pred, average=None, labels=range(len(classes)))
    per_class_base = f1_score(y, base_pred, average=None, labels=range(len(classes)))
    cdf = pd.DataFrame({"class": classes, "base_f1": per_class_base, "treat_f1": per_class, }).assign(delta=lambda d: d.treat_f1 - d.base_f1)
    entry["class_focus"] = cdf[cdf["class"].isin(["UCEC", "ACC", "COAD", "BRCA"])].to_dict("records")
    cdf_sorted = cdf.sort_values("delta")
    entry["worst5"] = cdf_sorted.head(5).to_dict("records")
    entry["best5"] = cdf_sorted.tail(5).to_dict("records")
    report["runs"][name] = entry
    print(f"{name:12s} f1={f1:.6f} delta={f1-base_f1:+.6f} changed={changed.sum()} rescue={rescue} hurt={hurt} net={rescue-hurt}")

(OUT / "packed4_v4s_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
print("\nsaved to", OUT)
print(f"total elapsed: {time.time()-t0:.0f}s")
