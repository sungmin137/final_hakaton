"""접근 14 v3: 9차 모델(v4 피처, colsample 0.7)과 접근14 v2 모델(v4p = v4 + 유형분리 점수, colsample 0.7)의 로그 확률 평균(w=0.5)
→ 3차 복원 배율 → 쌍둥이 규칙. 정직 CV: 9차 0.4847 → 평균 0.4941 (변이 31~100개 구간 0.403→0.457).
9차 test 확률은 저장본(6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy, 배율 곱해진 상태 → 나눠서 원시화)을 재사용하고,
v2 모델은 train 전체로 학습해 test에 적용한다. test 통계는 쓰지 않는다.

재현: python3 "4. src/submissions_source/approach14_v3_20260913_2132.py"  →  5. submissions/approach14_v3_20260913_2132.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from main import ROOT, DATA, ID, TARGET, FeatureMaker, PARAM_SETS, load_train, load_test
from postprocess.twin_rule import TwinRule

NAME = "approach14_v3_20260913_2132"; W = 0.5
train = load_train(); le = LabelEncoder(); y = le.fit_transform(train[TARGET]); classes = list(le.classes_)
s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
fm = FeatureMaker("v4p").fit(train); model = xgb.XGBClassifier(**PARAM_SETS["mild_col"]).fit(fm.transform(train), y)
test = load_test(); p2 = model.predict_proba(fm.transform(test))
p7 = np.load(ROOT / "6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy") / s3; p7 = p7 / p7.sum(1, keepdims=True)
z = np.log(p7 + 1e-9) * (1 - W) + np.log(p2 + 1e-9) * W; proba = np.exp(z - z.max(1, keepdims=True)); proba /= proba.sum(1, keepdims=True)
pred = le.inverse_transform((proba * s3).argmax(1))
sub = pd.read_csv(DATA / "sample_submission.csv"); assert (sub[ID] == test[ID]).all()
sub[TARGET] = pred; sub.to_csv(ROOT / "5. submissions" / f"{NAME}.csv", index=False, encoding="UTF-8-sig")
pred_tw, n_hit = TwinRule().fit(train).apply(test, pred); sub[TARGET] = pred_tw
sub.to_csv(ROOT / "6. experiments/submissions/twin_rule" / f"{NAME}_twin_rule.csv", index=False, encoding="UTF-8-sig")
out = ROOT / "6. experiments/submissions" / NAME; out.mkdir(parents=True, exist_ok=True); np.save(out / "test_proba_raw_blend.npy", proba)
ref = pd.read_csv(ROOT / "5. submissions/approach12_v4_20260910_1651.csv")[TARGET].to_numpy()
print(f"saved {NAME} | 9차와 다른 행 {(pred != ref).sum()} | STES {(pred=='STES').mean():.1%} | 규칙 변경 {(pred_tw != pred).sum()}행")
