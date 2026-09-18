"""approach23 v2-B: burden 8~15로 도너 풀을 좁힌 GBMLGG/LGG synthetic +50 통제 실험.

approach23_v1(클래스 전체에서 도너를 뽑음)과 approach14 v3 오답 분석(approach23_v2_error_analysis.py)의
결과를 보면, GBMLGG/LGG 오답은 "전형적인" 표본이 아니라 burden이 8~15로 겹치는 경계 영역에 몰려 있다
(GBMLGG 정답 median 14 vs 오답 median 9, LGG 정답 median 6 vs 오답 median 10). v1의 한쪽만 augment하면
반대 클래스가 무너지던 현상도, 클래스 "전체"(쉬운 표본 포함) 분포를 강화한 부작용으로 설명된다.

이 실험은 synthetic 도너 풀을 **그 fold의 학습 파티션(tr_part) 안에서, 해당 클래스이면서 burden이
8~15인 행**으로만 제한한다. burden은 모델 예측·OOF 정답 여부와 무관한 원본 피처(변이 개수)이므로
"어려운 행"을 정답/오답으로 판별할 필요가 없고, validation의 정답 정보를 전혀 보지 않는다 — v1과
동일한 누수 방지 구조(synthetic은 tr_part에서만, FeatureMaker도 tr_part만으로 fit)를 그대로 유지한다.

burden 범위(8~15)는 팀이 미리 정한 값이 아니라 approach23_v2_error_analysis.py의 실측 정답/오답
median에서 그대로 가져온 것 — 사람이 임의로 정한 규칙이 아니라 데이터가 보여준 경계.

이후 단계(v2-C: marker 조건 추가, v2-D: 실제 hard-profile만)는 이 결과가 명확한 신호를 보일 때만 진행.

설명 문서: 2. team/approaches/approach23.md
실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach23_synthetic_aug/approach23_v2b_run_experiment_burden.py"
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, PARAM_SETS, TARGET, twin_groups, load_train, FeatureMaker
from features.features import gene_columns
from approach23_synthetic_aug.synthetic_gen import generate_synthetic

PAIR = ["GBMLGG", "LGG"]
N_SYN = 50
SEEDS = [42, 7, 123]
N_SPLITS = 5
FEATURES = "v4"
PARAMS = PARAM_SETS["mild_col"]
BURDEN_LO, BURDEN_HI = 8, 15   # approach23_v2_error_analysis.py 실측 정답/오답 median 경계

CONDITIONS = {
    "baseline": {},
    "gbmlgg_synth50": {"GBMLGG": N_SYN},
    "lgg_synth50": {"LGG": N_SYN},
    "both_synth50": {"GBMLGG": N_SYN, "LGG": N_SYN},
}
CLASS_SEED_OFFSET = {"GBMLGG": 1, "LGG": 2}

train = load_train()
genes = gene_columns(train)
burden_all = (train[genes] != "WT").sum(axis=1)
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
classes = list(le.classes_)


def hard_burden_pool(tr_part: pd.DataFrame, cls: str) -> pd.DataFrame:
    b = burden_all.loc[tr_part.index]
    mask = (tr_part[TARGET] == cls) & (b >= BURDEN_LO) & (b <= BURDEN_HI)
    return tr_part[mask]


per_seed_rows = []
for sd in SEEDS:
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=sd)
    oof = {cond: np.full(len(train), -1) for cond in CONDITIONS}

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]

        fm = FeatureMaker(FEATURES).fit(tr_part)
        Xtr_real = fm.transform(tr_part)
        Xva = fm.transform(va_part)

        pool_sizes = {c: len(hard_burden_pool(tr_part, c)) for c in PAIR}

        for cond, spec in CONDITIONS.items():
            Xtr, y_tr = Xtr_real, y_all[tri]
            if spec:
                syn_frames = []
                for cls, n in spec.items():
                    pool = hard_burden_pool(tr_part, cls)   # burden 8~15로 좁힌 도너 풀, tr_part 안에서만
                    syn_seed = sd * 1000 + k * 10 + CLASS_SEED_OFFSET[cls]
                    syn_frames.append(generate_synthetic(pool, cls, n, seed=syn_seed))
                syn_all = pd.concat(syn_frames, ignore_index=True)
                Xsyn = fm.transform(syn_all)
                Xtr = pd.concat([Xtr_real, Xsyn], ignore_index=True)
                y_tr = np.concatenate([y_tr, le.transform(syn_all[TARGET])])

            model = xgb.XGBClassifier(**PARAMS).fit(Xtr, y_tr)
            oof[cond][vai] = model.predict_proba(Xva).argmax(1)

        print(f"  seed{sd} fold{k} 완료 (도너 풀 크기 {pool_sizes})", flush=True)

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
df.to_csv(out_dir / "v2b_burden_per_seed_results.csv", index=False)
summary_df.to_csv(out_dir / "v2b_burden_summary.csv", index=False)
print(f"\nsaved: {out_dir}")
