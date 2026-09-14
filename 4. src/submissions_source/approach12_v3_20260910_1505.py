"""접근 12 (구 접근 2 v6) (후처리 전용): v5의 16모델 배깅 원시 확률에 **3차(LB 0.4369)의 복원 배율**을 적용하고 쌍둥이 규칙을 얹은 최종 극대화 후보.
입력: 6. experiments/submissions/approach12_v2_20260910_1453/test_proba.npy (v5가 저장한 '원시 × 평균 배율' 확률; 평균 배율을 되돌려 원시 확률 복원)
배율: 6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json (최소제곱 복원, 잔차 1e-7)

재현: python3 "4. src/submissions_source/approach12_v3_20260910_1505.py"  →  5. submissions/approach12_v3_20260910_1505.csv (+ _twin_rule)
설명 문서: 2. team/approaches/approach12.md
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from main import ROOT, DATA, load_train, load_test, TARGET, ID
from postprocess.class_scale import fit_class_scales
from postprocess.twin_rule import TwinRule

NAME = "approach12_v3_20260910_1505"
train = load_train(); classes = sorted(train[TARGET].unique()); yi = train[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
P = np.load(ROOT / "6. experiments/submissions/approach12_v2_20260910_1453/test_proba.npy")
# v5가 곱한 '절반 분할 6회 기하평균 배율'을 동일 절차로 재계산해 되돌린다
oof = np.load(ROOT / "6. experiments/2026-09-09_v4_xgb_grp/oof_proba.npy"); rng = np.random.RandomState(0); logs = []
for _ in range(6):
    idx = rng.permutation(len(yi)); half = idx[: len(yi) // 2]; logs.append(np.log(fit_class_scales(oof[half], yi[half])))
s_avg = np.exp(np.mean(logs, 0)); raw = P / s_avg
assert np.allclose(raw.sum(1), 1, atol=1e-4), "원시 확률 복원 실패"
s3 = json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json")); s3 = np.array([s3[c] for c in classes])
pred = np.array(classes)[(raw * s3).argmax(1)]
test = load_test(); sample = pd.read_csv(DATA / "sample_submission.csv"); assert (sample[ID] == test[ID]).all()
out = ROOT / "5. submissions"; sub = sample.copy(); sub[TARGET] = pred; sub.to_csv(out / f"{NAME}.csv", index=False, encoding="UTF-8-sig")
pred_tw, n_hit = TwinRule().fit(train).apply(test, pred); sub_tw = sample.copy(); sub_tw[TARGET] = pred_tw
sub_tw.to_csv(ROOT / "6. experiments/submissions/twin_rule" / f"{NAME}_twin_rule.csv", index=False, encoding="UTF-8-sig")
ref = pd.read_csv(ROOT / "6. experiments/submissions/twin_rule/approach2_v2_20260909_1653_twin_rule.csv")[TARGET].to_numpy()
print(f"saved: {NAME}.csv / _twin_rule.csv | 3차(규칙본)와 다른 행 {(pred_tw != ref).sum()} | STES {(pred_tw == 'STES').mean():.1%} | 규칙 변경 {(pred != pred_tw).sum()}행")
