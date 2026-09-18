"""approach23 v17: (1) 26클래스 전체 가벼운 augmentation(+10%/+20%), (2) TP53 burden-normalized feature.
기존 최고 파이프라인의 핵심 구성요소인 "9차"(features v4, params mild_col)를 대표로 사용 —
21차 전체 파이프라인(9차+v2 블렌드+배율+twin_rule+hypermut_rule)을 통째로 재현하지 않음(명시).
새 모델/서드모델/블렌드/라우팅/후처리 규칙은 만들지 않음. 결과는 새 실험 디렉터리에만 저장.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from main import ROOT, DATA, SEED, PARAM_SETS, twin_groups, load_train, FeatureMaker
from features.features import gene_columns, TARGET
from approach23_synthetic_aug.synthetic_gen import generate_synthetic

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)
PARAMS = PARAM_SETS["mild_col"]
FEATURES = "v4"

OUT = ROOT / "6. experiments" / "2026-09-14_approach23_v17_light_aug_burden_norm"
OUT.mkdir(parents=True, exist_ok=True)

class_size = pd.Series(y).value_counts().sort_index().to_numpy()  # per encoded class

BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi<10**9 else '+'}" for lo, hi in BINS]


def macro_by_bin(pred_full):
    row = {}
    for (lo, hi), nm in zip(BINS, bin_names):
        m = (burden >= lo) & (burden <= hi)
        row[nm] = round(f1_score(y[m], pred_full[m], average="macro", labels=sorted(set(y[m].tolist()))), 4)
    row["전체"] = round(f1_score(y, pred_full, average="macro", labels=sorted(set(y.tolist()))), 4)
    return row


groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y, groups))

# ================================================================ 실험 1: 가벼운 전체 augmentation
print("=" * 78)
print("실험 1: 26클래스 전체 +10% / +20% augmentation (fold-honest, 도너는 그 fold tr_part만)")
results = {}
synthetic_records = {"0.10": [], "0.20": []}

for frac_name, frac in [("0.10", 0.10), ("0.20", 0.20)]:
    oof_pred = np.full(n, -1)
    for k, (tri, vai) in enumerate(folds):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        fm = FeatureMaker(FEATURES).fit(tr_part)
        Xtr_real = fm.transform(tr_part)
        Xva = fm.transform(va_part)
        y_tr_real = y[tri]

        syn_frames = []
        for ci, cname in enumerate(classes):
            pool = tr_part[tr_part[TARGET] == cname]
            n_pool = len(pool)
            n_syn = max(1, round(n_pool * frac))
            if n_pool < 2:
                continue
            syn_seed = SEED * 1000 + k * 10 + ci
            syn_frames.append(generate_synthetic(pool, cname, n_syn, seed=syn_seed))
        syn_all = pd.concat(syn_frames, ignore_index=True)
        Xsyn = fm.transform(syn_all)
        Xtr = pd.concat([Xtr_real, Xsyn], ignore_index=True)
        y_tr = np.concatenate([y_tr_real, le.transform(syn_all[TARGET])])

        model = xgb.XGBClassifier(**PARAMS).fit(Xtr, y_tr)
        oof_pred[vai] = model.predict(Xva)
        if k == 0:
            synthetic_records[frac_name].append(syn_all.assign(fold=k))
        print(f"  frac={frac_name} fold{k}: real train {len(tr_part)} + synthetic {len(syn_all)} = {len(Xtr)}", flush=True)

    results[f"aug_{frac_name}"] = macro_by_bin(oof_pred)
    np.save(OUT / f"oof_pred_aug{frac_name}.npy", oof_pred)
    per_class_f1 = f1_score(y, oof_pred, average=None, labels=range(len(classes)))
    pd.Series(per_class_f1, index=classes).to_csv(OUT / f"per_class_f1_aug{frac_name}.csv")

# fold0의 synthetic 샘플 저장 (생성 방식 기록/sanity check용)
for frac_name in ["0.10", "0.20"]:
    pd.concat(synthetic_records[frac_name], ignore_index=True).to_csv(OUT / f"synthetic_sample_fold0_aug{frac_name}.csv", index=False)

# ================================================================ baseline (9차, 기존 저장 OOF 재사용)
oof9_saved = np.load(ROOT / "6. experiments/2026-09-11_v4_xgb_mild_col_grp/oof_proba.npy")
pred9_saved = oof9_saved.argmax(1)
results["baseline_9cha"] = macro_by_bin(pred9_saved)

print("\n" + "=" * 78)
print("실험 1 결과: baseline vs +10% vs +20%")
tbl1 = pd.DataFrame([results["baseline_9cha"], results["aug_0.10"], results["aug_0.20"]],
                    index=["baseline_9cha", "aug_+10%", "aug_+20%"])
print(tbl1.to_string())

# class별 변화(baseline 대비)
base_per_class = pd.Series(f1_score(y, pred9_saved, average=None, labels=range(len(classes))), index=classes)
for frac_name in ["0.10", "0.20"]:
    pc = pd.read_csv(OUT / f"per_class_f1_aug{frac_name}.csv", index_col=0).iloc[:, 0]
    diff = (pc - base_per_class).sort_values()
    print(f"\naug_{frac_name} class별 변화 최악 5개 / 최선 5개:")
    print("  최악:", ", ".join(f"{c}({diff[c]:+.3f})" for c in diff.head(5).index))
    print("  최선:", ", ".join(f"{c}({diff[c]:+.3f})" for c in diff.tail(5).index))

# synthetic burden 분포 확인 (fold0)
print("\nsynthetic sample burden 분포 (fold0, +10%/+20%) vs 원본 train burden 분포:")
for frac_name in ["0.10", "0.20"]:
    syn_df = pd.read_csv(OUT / f"synthetic_sample_fold0_aug{frac_name}.csv")
    syn_burden = (syn_df[genes] != "WT").sum(axis=1)
    print(f"  aug_{frac_name}: n={len(syn_df)}, burden mean={syn_burden.mean():.1f} median={syn_burden.median():.1f} "
          f"(원본 train mean={burden.mean():.1f} median={np.median(burden):.1f})")

out_dir = OUT
tbl1.to_csv(out_dir / "exp1_summary.csv")
print(f"\nsaved: {out_dir}")
