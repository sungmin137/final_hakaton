"""approach23 v1: GBMLGG/LGG synthetic augmentation (+50) 통제 실험.

목적: 현재 train만으로 만든 GBMLGG/LGG synthetic 샘플(2-도너 crossover 교란, synthetic_gen.py)이
26클래스 메인 파이프라인의 일반화 성능(정직 CV, group-twins)을 개선하는지 확인한다.

4개 조건을 동일 fold·동일 seed에서 비교: baseline / GBMLGG+50 / LGG+50 / GBMLGG+50+LGG+50.
synthetic은 그 fold의 학습 파티션(tr_part)의 GBMLGG/LGG 행에서만 생성하고, 검증 파티션(va_part)에는
전혀 관여하지 않는다. FeatureMaker는 tr_part(실제 행만)로 fit해서 4개 조건이 같은 피처 공간을
공유하게 하고, synthetic 행 추가로 인한 학습 데이터 차이만 비교되도록 통제했다.

메인 모델: features v4, params mild_col (approach11 v11 통합테스트와 동일 — 팀 기존 비교 기준과 맞춤)

설명 문서: 2. team/approaches/approach23.md
실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach23_synthetic_aug/approach23_v1_run_experiment.py"
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, PARAM_SETS, TARGET, twin_groups, load_train, FeatureMaker
from approach23_synthetic_aug.synthetic_gen import generate_synthetic

PAIR = ["GBMLGG", "LGG"]
N_SYN = 50
SEEDS = [42, 7, 123]
N_SPLITS = 5
FEATURES = "v4"
PARAMS = PARAM_SETS["mild_col"]

CONDITIONS = {
    "baseline": {},
    "gbmlgg_synth50": {"GBMLGG": N_SYN},
    "lgg_synth50": {"LGG": N_SYN},
    "both_synth50": {"GBMLGG": N_SYN, "LGG": N_SYN},
}
CLASS_SEED_OFFSET = {"GBMLGG": 1, "LGG": 2}

train = load_train()
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
classes = list(le.classes_)

per_seed_rows = []
for sd in SEEDS:
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=sd)
    oof = {cond: np.full(len(train), -1) for cond in CONDITIONS}

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]

        fm = FeatureMaker(FEATURES).fit(tr_part)          # 실제 행만으로 fit (조건 간 공정 비교)
        Xtr_real = fm.transform(tr_part)
        Xva = fm.transform(va_part)

        for cond, spec in CONDITIONS.items():
            Xtr, y_tr = Xtr_real, y_all[tri]
            if spec:
                syn_frames = []
                for cls, n in spec.items():
                    pool = tr_part[tr_part[TARGET] == cls]     # 이 fold의 학습 파티션만, val/test 미포함
                    syn_seed = sd * 1000 + k * 10 + CLASS_SEED_OFFSET[cls]
                    syn_frames.append(generate_synthetic(pool, cls, n, seed=syn_seed))
                syn_all = pd.concat(syn_frames, ignore_index=True)
                Xsyn = fm.transform(syn_all)
                Xtr = pd.concat([Xtr_real, Xsyn], ignore_index=True)
                y_tr = np.concatenate([y_tr, le.transform(syn_all[TARGET])])

            model = xgb.XGBClassifier(**PARAMS).fit(Xtr, y_tr)
            oof[cond][vai] = model.predict_proba(Xva).argmax(1)

        print(f"  seed{sd} fold{k} 완료", flush=True)

    row = {"seed": sd}
    for cond in CONDITIONS:
        pred = oof[cond]
        row[f"{cond}_macro_f1"] = round(f1_score(y_all, pred, average="macro"), 4)
        per_class = dict(zip(classes, f1_score(y_all, pred, average=None, labels=range(len(classes)))))
        row[f"{cond}_GBMLGG_f1"] = round(per_class["GBMLGG"], 4)
        row[f"{cond}_LGG_f1"] = round(per_class["LGG"], 4)
    per_seed_rows.append(row)
    print(row, flush=True)

df = pd.DataFrame(per_seed_rows)
print("\n=== 시드별 결과 ===")
print(df.to_string(index=False))

print("\n=== 평균 ± 표준편차 (seed 3개) ===")
summary = []
for cond in CONDITIONS:
    r = {"condition": cond}
    for metric in ("macro_f1", "GBMLGG_f1", "LGG_f1"):
        col = f"{cond}_{metric}"
        r[f"{metric}_mean"] = round(df[col].mean(), 4)
        r[f"{metric}_std"] = round(df[col].std(), 4)
    summary.append(r)
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_synthetic_gbmlgg_lgg"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "per_seed_results.csv", index=False)
summary_df.to_csv(out_dir / "summary.csv", index=False)
print(f"\nsaved: {out_dir}")
