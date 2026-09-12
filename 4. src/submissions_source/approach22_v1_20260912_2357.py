"""접근22 v1: v7 + class_scale(HNSC/STES만 옛날 고정값, 나머지 24개는 v7 전용 재계산)
+ 혜림님 HNSC/STES 라우팅(approach19) + 쌍둥이 규칙.

배경: 옛날 고정 class_scale 파일은 v7 이전 모델로 한 번 계산된 것이라 v7 자신의 OOF로 재계산하면
CV가 크게(+0.0143) 오름. 하지만 재계산본은 STES 배율이 1.0(억제 없음)이 되면서 실제 test 추론 시
STES 예측비율이 23.3%까지 치솟는 위험이 발견됨(과거 LB 하락과 상관된 패턴, approach11.md v21 참고).
혜림님의 HNSC/STES 라우터는 "HNSC=0.7, STES=0.5"라는 옛날 값을 전제로 검증(CV+LB 0.44235)됐는데,
재계산본은 HNSC도 0.7->1.5로 바뀌어서 라우터 개입 조건 자체가 달라짐 — 검증 안 된 조합이 됨.
그래서 HNSC/STES 둘 다 옛날 값으로 고정하고, 나머지 24개 클래스만 v7 재계산 값을 씀.

쌍둥이 규칙 미적용본(approach22_v1_20260912_2357.csv)은 이미 LB 제출됨(0.434171, v7보다 낮음).
이 스크립트는 쌍둥이 규칙까지 적용한 버전을 approach22로 새로 이름 붙여 이어감(approach11.md 참고).

실행: python3 "4. src/submissions_source/approach22_v1_20260912_2357.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from main import DATA, ID, ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, gene_columns, load_test, load_train
from postprocess.twin_rule import TwinRule

TAG = "approach22_v1_20260912_2357"
PAIR = ("HNSC", "STES")
C, CONFIDENCE, TOP_K = 0.1, 0.70, 40
# 24개 클래스는 v7 자신의 정직 CV OOF(6. experiments/2026-09-11_v4_xgb_mild_col_grp/oof_proba.npy)로
# fit_class_scales() 재계산. HNSC/STES 둘만 옛날 고정값으로 되돌림(혜림님 라우터 전제 조건 보존).
SCALES = {"ACC": 0.5, "BLCA": 1.2, "BRCA": 0.5, "CESC": 1.5, "COAD": 0.5, "DLBC": 2.5, "GBMLGG": 0.5,
          "HNSC": 0.7, "KIPAN": 0.85, "KIRC": 2.5, "LAML": 0.5, "LGG": 1.5, "LIHC": 0.85, "LUAD": 1.0,
          "LUSC": 1.0, "OV": 1.2, "PAAD": 2.0, "PCPG": 1.2, "PRAD": 1.5, "SARC": 1.5, "SKCM": 0.5,
          "STES": 0.5, "TGCT": 0.85, "THCA": 0.5, "THYM": 1.0, "UCEC": 0.85}


def validate(sub: pd.DataFrame, sample: pd.DataFrame, labels: set[str]) -> None:
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and (sub[ID] == sample[ID]).all()
    assert sub[TARGET].notna().all() and set(sub[TARGET]) <= labels


def main() -> None:
    train = load_train(); genes = gene_columns(train)
    le = LabelEncoder(); y = le.fit_transform(train[TARGET])
    ids = {name: i for i, name in enumerate(le.classes_)}
    a, b = ids[PAIR[0]], ids[PAIR[1]]
    fm = FeatureMaker("v4").fit(train)
    main_model = xgb.XGBClassifier(**{**XGB_PARAMS, "colsample_bytree": .7, "random_state": SEED})
    main_model.fit(fm.transform(train), y)

    Xtr = (train[genes].to_numpy() != "WT").astype(np.int8)
    ptr = np.flatnonzero(np.isin(y, (a, b)))
    top = np.argsort(np.abs(Xtr[ptr][y[ptr] == a].mean(0) - Xtr[ptr][y[ptr] == b].mean(0)))[-TOP_K:]
    referee = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED)
    referee.fit(Xtr[ptr][:, top], y[ptr])

    test = load_test()
    main_proba = main_model.predict_proba(fm.transform(test))
    scale = np.array([SCALES[n] for n in le.classes_])
    pred_id = (main_proba * scale).argmax(1)
    Xte = (test[genes].to_numpy() != "WT").astype(np.int8)
    p_ref = referee.predict_proba(Xte[:, top])
    ref_id = referee.classes_[p_ref.argmax(1)]
    use = np.isin(pred_id, (a, b)) & (p_ref.max(1) >= CONFIDENCE)
    before = pred_id.copy(); pred_id[use] = ref_id[use]
    pred = le.inverse_transform(pred_id)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    out = ROOT / "5. submissions"; out.mkdir(exist_ok=True)
    sub = sample.copy(); sub[TARGET] = pred; validate(sub, sample, set(train[TARGET]))
    base_path = out / f"{TAG}.csv"; sub.to_csv(base_path, index=False, encoding="UTF-8-sig")

    pred_tw, hits = TwinRule().fit(train).apply(test, pred)
    sub_tw = sample.copy(); sub_tw[TARGET] = pred_tw; validate(sub_tw, sample, set(train[TARGET]))
    twin_path = out / f"{TAG}_twin_rule.csv"; sub_tw.to_csv(twin_path, index=False, encoding="UTF-8-sig")
    print(f"base={base_path}\ntwin={twin_path}")
    print(f"router_candidates={int(use.sum())}, router_changed={int((before != pred_id).sum())}, "
          f"twin_matches={hits}, twin_changed={int((pred != pred_tw).sum())}")
    dist = pd.Series(pred_tw).value_counts(normalize=True)
    print(f"[test] STES 예측비율={dist.get('STES',0)*100:.2f}%  HNSC 예측비율={dist.get('HNSC',0)*100:.2f}%")


if __name__ == "__main__":
    main()
