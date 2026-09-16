"""v4s alone vs v4s+domain-hit as direct XGB features.
StratifiedGroupKFold(5, shuffle=True, random_state=42) on twin_groups(train), PARAM_SETS['mild_col'].
Reuses PreprocessFeatures dom_ logic and spectrum_features as-is. train.csv only.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
COMMON = ROOT / "4. src/common"
sys.path.insert(0, str(COMMON))
OUT = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")

from main import SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker, TARGET
from features.features import gene_columns
from approach23_synthetic_aug.spectrum_ref import spectrum_features
from approach7_preprocess.preprocess_features import PreprocessFeatures

train = load_train()
genes = gene_columns(train)
values = train[genes].to_numpy()
burden = (values != "WT").sum(axis=1)  # n_mut_genes, used for burden-bin breakdown

le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]

groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y, groups))
print(f"seed={SEED} fold sizes: {[len(v) for _, v in folds]}", flush=True)

oof_base = np.zeros((n, len(classes)))
oof_dom = np.zeros((n, len(classes)))
n_dom_cols = []

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

    pp = PreprocessFeatures().fit(tr_part)   # existing dom_ code as-is, fold-fit only
    Xtr_pp = pp.transform(tr_part).reset_index(drop=True)
    Xva_pp = pp.transform(va_part).reset_index(drop=True)
    dom_cols = [c for c in Xtr_pp.columns if c.startswith("dom_")]
    n_dom_cols.append(len(dom_cols))

    Xtr_dom = pd.concat([Xtr_v4s, Xtr_pp[dom_cols]], axis=1)
    Xva_dom = pd.concat([Xva_v4s, Xva_pp[dom_cols]], axis=1)

    mB = xgb.XGBClassifier(**PARAMS).fit(Xtr_v4s, y[tri])
    oof_base[vai] = mB.predict_proba(Xva_v4s)
    f1b = f1_score(y[vai], oof_base[vai].argmax(1), average="macro")

    mD = xgb.XGBClassifier(**PARAMS).fit(Xtr_dom, y[tri])
    oof_dom[vai] = mD.predict_proba(Xva_dom)
    f1d = f1_score(y[vai], oof_dom[vai].argmax(1), average="macro")

    print(f"fold{k}: n_dom_cols={len(dom_cols)} BASE_f1={f1b:.4f} DOM_f1={f1d:.4f} ({time.time()-t0:.0f}s)", flush=True)

f1_base = f1_score(y, oof_base.argmax(1), average="macro")
f1_dom = f1_score(y, oof_dom.argmax(1), average="macro")
print(f"\nOverall BASE (v4s) macro F1 = {f1_base:.4f}")
print(f"Overall v4s+domain-hit macro F1 = {f1_dom:.4f}")
print(f"Delta = {f1_dom-f1_base:+.4f}")
print(f"avg dom cols per fold = {np.mean(n_dom_cols):.1f}")

# burden-bin breakdown
bands = {"0_10": burden <= 10, "11_30": (burden >= 11) & (burden <= 30),
         "31_100": (burden >= 31) & (burden <= 100), "101_plus": burden >= 101}
burden_report = {}
for name, mask in bands.items():
    fb = f1_score(y[mask], oof_base[mask].argmax(1), average="macro")
    fd = f1_score(y[mask], oof_dom[mask].argmax(1), average="macro")
    burden_report[name] = {"n": int(mask.sum()), "base_f1": round(float(fb), 5), "dom_f1": round(float(fd), 5), "delta": round(float(fd - fb), 5)}
    print(f"  burden {name}: n={mask.sum():4d} BASE={fb:.4f} DOM={fd:.4f} delta={fd-fb:+.4f}")

# class-level breakdown
per_class_base = f1_score(y, oof_base.argmax(1), average=None, labels=range(len(classes)))
per_class_dom = f1_score(y, oof_dom.argmax(1), average=None, labels=range(len(classes)))
class_df = pd.DataFrame({"class": classes, "base_f1": per_class_base, "dom_f1": per_class_dom})
class_df["delta"] = class_df["dom_f1"] - class_df["base_f1"]
class_df = class_df.sort_values("delta")
print("\nworst 5 (biggest negative delta):")
print(class_df.head(5).to_string(index=False))
print("\nbest 5 (biggest positive delta):")
print(class_df.tail(5).to_string(index=False))

np.save(OUT / "domxgb_oof_base.npy", oof_base)
np.save(OUT / "domxgb_oof_dom.npy", oof_dom)
class_df.to_csv(OUT / "domxgb_class_delta.csv", index=False)
json.dump({
    "f1_base": float(f1_base), "f1_dom": float(f1_dom), "delta": float(f1_dom - f1_base),
    "avg_dom_cols": float(np.mean(n_dom_cols)), "burden": burden_report,
}, open(OUT / "domxgb_result.json", "w"), indent=2, ensure_ascii=False)
print(f"\ntotal elapsed: {time.time()-t0:.0f}s")
print("saved to", OUT)
