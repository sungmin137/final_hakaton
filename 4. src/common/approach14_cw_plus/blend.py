"""접근 14 v3 이후 공용: 9차 모델 확률 + 접근14 v2 모델 확률의 로그평균 블렌드 → 3차 복원 배율 → (선택) HNSC/STES 라우터 → 쌍둥이 규칙.

두 모델의 test 확률은 16차(approach14_v3_20260913_2132) 시점의 저장본을 그대로 쓴다(재학습 편차 없이 '한 번에 한 요소'만 바꾸기 위해).
- 9차: 6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy (배율 곱해진 상태 → 나눠서 원시화)
- v2 : 6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy (w=0.5 블렌드 저장본에서 p2 ∝ blend²/p7 로 복원, 재현 오차 1e-8, 예측 일치 확인)
라우터는 혜림 님 approach19(hyerim 브랜치) 그대로: HNSC/STES 환자만으로 학습한 LR(C=0.1, 빈도차 상위 40 유전자), 예측이 두 암종 중 하나이고 확신 ≥0.7일 때만 교체.
test 통계는 어디에도 쓰지 않는다.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from main import ROOT, DATA, ID, TARGET, SEED, load_train, load_test
from postprocess.twin_rule import TwinRule

P7_PATH = ROOT / "6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy"
P2_PATH = ROOT / "6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy"
SCALE_PATH = ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"
EPS = 1e-9


def log_blend(p7: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    z = np.log(p7 + EPS) * (1 - w) + np.log(p2 + EPS) * w
    p = np.exp(z - z.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)


def hnsc_stes_router(train, test, le, pred_id, pair=("HNSC", "STES"), C=0.1, confidence=0.7, top_k=40):
    """approach19 소프트 재판기. 반환: (수정된 pred_id, 후보 수, 실제 바뀐 수)."""
    genes = [c for c in train.columns if c not in (ID, TARGET)]
    y = le.transform(train[TARGET]); a, b = le.transform(list(pair))
    Xtr = (train[genes].to_numpy() != "WT").astype(np.int8); ptr = np.flatnonzero(np.isin(y, (a, b)))
    top = np.argsort(np.abs(Xtr[ptr][y[ptr] == a].mean(0) - Xtr[ptr][y[ptr] == b].mean(0)))[-top_k:]
    ref = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED).fit(Xtr[ptr][:, top], y[ptr])
    Xte = (test[genes].to_numpy() != "WT").astype(np.int8); p_ref = ref.predict_proba(Xte[:, top])
    use = np.isin(pred_id, (a, b)) & (p_ref.max(1) >= confidence)
    out = pred_id.copy(); out[use] = ref.classes_[p_ref.argmax(1)][use]
    return out, int(use.sum()), int((out != pred_id).sum())


def run(name: str, w: float, router: bool = False, ref_name: str = "approach14_v3_20260913_2132") -> None:
    train = load_train(); le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_)
    s3 = np.array([json.load(open(SCALE_PATH))[c] for c in classes])
    p7 = np.load(P7_PATH) / s3; p7 = p7 / p7.sum(1, keepdims=True); p2 = np.load(P2_PATH)
    test = load_test(); proba = log_blend(p7, p2, w); pred_id = (proba * s3).argmax(1)
    msg = ""
    if router:
        pred_id, n_cand, n_chg = hnsc_stes_router(train, test, le, pred_id); msg = f" | 라우터 후보 {n_cand}, 변경 {n_chg}행"
    pred = le.inverse_transform(pred_id)
    sub = pd.read_csv(DATA / "sample_submission.csv"); assert (sub[ID] == test[ID]).all()
    sub[TARGET] = pred; sub.to_csv(ROOT / "5. submissions" / f"{name}.csv", index=False, encoding="UTF-8-sig")
    pred_tw, n_hit = TwinRule().fit(train).apply(test, pred); sub[TARGET] = pred_tw
    sub.to_csv(ROOT / "6. experiments/submissions/twin_rule" / f"{name}_twin_rule.csv", index=False, encoding="UTF-8-sig")
    out = ROOT / "6. experiments/submissions" / name; out.mkdir(parents=True, exist_ok=True); np.save(out / "test_proba_raw_blend.npy", proba)
    ref = pd.read_csv(ROOT / "5. submissions" / f"{ref_name}.csv")[TARGET].to_numpy()
    ref_tw = pd.read_csv(ROOT / "6. experiments/submissions/twin_rule" / f"{ref_name}_twin_rule.csv")[TARGET].to_numpy()
    print(f"saved {name} (w={w}, router={router}){msg} | 16차와 다른 행 {(pred != ref).sum()} (규칙본끼리 {(pred_tw != ref_tw).sum()}) "
          f"| STES {(pred == 'STES').mean():.1%} | 규칙 변경 {(pred_tw != pred).sum()}행")
