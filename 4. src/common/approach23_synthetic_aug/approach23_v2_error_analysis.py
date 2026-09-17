"""approach23 v2: approach14 v3(팀 최고 LB 0.45641)의 GBMLGG/LGG 오답 구조 분석.

approach14 v3 = 9차 모델(features v4, mild_col) + v2 모델(features v4p, mild_col)의
로그 확률 평균(w=0.5) + 3차 복원 배율(class_scales_recovered.json) — `approach14_v3_20260913_2132.py`,
`approach14_cw_plus/blend.py`와 동일한 레시피를 정직 CV(group-twins, seed=42)로 재현해서
OOF 예측을 만든다. approach14는 이 블렌드의 test 확률만 저장해뒀고(test 라벨 없음) 클래스별
confusion matrix를 만든 적이 없어서, "실제로 어떤 GBMLGG/LGG가 맞고 틀리는가"는 여기서 새로 계산한다.

목적: +50을 클래스 전체에 뿌리는 대신, 실제 오답이 몰린 mutation/burden 영역을 찾아
다음 synthetic 실험(v3)의 타겟을 좁히기 위한 진단. 이 스크립트는 synthetic을 만들지 않는다.

설명 문서: 2. team/approaches/approach23.md
실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach23_synthetic_aug/approach23_v2_error_analysis.py"
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, PARAM_SETS, TARGET, SEED, twin_groups, load_train, FeatureMaker
from features.features import gene_columns

PAIR = ["GBMLGG", "LGG"]
PARAMS = PARAM_SETS["mild_col"]
SCALE_PATH = ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"
EPS = 1e-9

train = load_train()
genes = gene_columns(train)
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
classes = list(le.classes_)
n = len(train)

groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_p7 = np.zeros((n, len(classes)))
oof_p2 = np.zeros((n, len(classes)))

for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm7 = FeatureMaker("v4").fit(tr_part)
    m7 = xgb.XGBClassifier(**PARAMS).fit(fm7.transform(tr_part), y_all[tri])
    oof_p7[vai] = m7.predict_proba(fm7.transform(va_part))

    fm2 = FeatureMaker("v4p").fit(tr_part)
    m2 = xgb.XGBClassifier(**PARAMS).fit(fm2.transform(tr_part), y_all[tri])
    oof_p2[vai] = m2.predict_proba(fm2.transform(va_part))
    print(f"fold{k} 완료", flush=True)

s3 = np.array([json.load(open(SCALE_PATH))[c] for c in classes])
z = np.log(oof_p7 + EPS) * 0.5 + np.log(oof_p2 + EPS) * 0.5
blend = np.exp(z - z.max(1, keepdims=True)); blend /= blend.sum(1, keepdims=True)
pred_id = (blend * s3).argmax(1)
pred = np.array(classes)[pred_id]
true = train[TARGET].to_numpy()

macro_f1 = f1_score(y_all, pred_id, average="macro")
print(f"\n=== 재현된 approach14 v3 정직 CV macro F1 = {macro_f1:.4f} (문서 기록 0.4941과 비교) ===")

print("\n=== GBMLGG/LGG confusion (행=실제, 열=예측) ===")
cm = confusion_matrix(true, pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
for c in PAIR:
    row = cm_df.loc[c]
    nonzero = row[row > 0].sort_values(ascending=False)
    print(f"실제 {c} (n={row.sum()}): " + ", ".join(f"{k}={v}" for k, v in nonzero.items()))

for c in PAIR:
    yb = (train[TARGET] == c).to_numpy().astype(int)
    pb = (pred == c).astype(int)
    print(f"{c} F1={f1_score(yb, pb):.4f}")

# ---- burden 기준 오답 분석 ----
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
print("\n=== mutation burden: 맞은 행 vs 틀린 행 (GBMLGG/LGG 실제 라벨 기준) ===")
for c in PAIR:
    mask = (true == c)
    correct = mask & (pred == true)
    wrong = mask & (pred != true)
    for name, m in (("정답", correct), ("오답", wrong)):
        b = burden[m]
        if len(b) == 0:
            print(f"{c} {name}: 0행"); continue
        d = pd.Series(b).describe(percentiles=[.25, .5, .75])
        print(f"{c} {name} (n={len(b)}): mean={d['mean']:.1f} median={d['50%']:.1f} "
              f"std={d['std']:.1f} min={d['min']:.0f} max={d['max']:.0f}")

# ---- 데이터 기반(사전 유전자 목록 없이): 정답군 vs 오답군 유전자 빈도 차이 ----
print("\n=== 유전자 빈도 차이: 정답군 vs 오답군 (사전 목록 없이 데이터로만 계산, 상위 10개) ===")
is_mut = (train[genes] != "WT")
for c in PAIR:
    mask = (true == c)
    correct = mask & (pred == true)
    wrong = mask & (pred != true)
    if wrong.sum() < 3:
        print(f"{c}: 오답 표본이 너무 적어({wrong.sum()}) 생략"); continue
    fc = is_mut[correct].mean()
    fw = is_mut[wrong].mean()
    diff = (fw - fc).sort_values(key=np.abs, ascending=False)
    print(f"-- {c} (정답 n={correct.sum()}, 오답 n={wrong.sum()}) --")
    for g in diff.head(10).index:
        print(f"  {g}: 정답군={fc[g]:.3f} 오답군={fw[g]:.3f} diff={diff[g]:+.3f}")

# ---- 참고: 팀이 이미 확인한 문헌 마커(approach11 v7) 기준 오답군 프로필 ----
MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
print("\n=== 참고: 기존 approach11 문헌 마커 기준 정답/오답군 변이율 ===")
for c in PAIR:
    mask = (true == c)
    correct = mask & (pred == true)
    wrong = mask & (pred != true)
    print(f"-- {c} --")
    for g in MARKERS:
        fc = is_mut.loc[correct, g].mean() if correct.sum() else float("nan")
        fw = is_mut.loc[wrong, g].mean() if wrong.sum() else float("nan")
        print(f"  {g}: 정답군={fc:.3f} 오답군={fw:.3f}")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_synthetic_gbmlgg_lgg"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "approach14v3_repro_oof_p7.npy", oof_p7)
np.save(out_dir / "approach14v3_repro_oof_p2.npy", oof_p2)
cm_df.to_csv(out_dir / "approach14v3_repro_confusion_matrix.csv")
print(f"\nsaved: {out_dir}")
