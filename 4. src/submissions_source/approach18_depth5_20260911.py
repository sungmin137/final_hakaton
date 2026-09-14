"""접근18 D 제출 재현 스크립트.

설정은 실제 최고 LB v7과 동일한 V4 + colsample=.7 + train OOF 복원 클래스 배율이며,
트리 깊이만 6에서 5로 낮춘다. test는 이 최종 추론 단계에서만 읽는다.
기본본은 v7과 같이 TwinRule 미적용, 별도 _twin_rule 파일은 선택 비교용이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from main import DATA, ID, ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, load_test, load_train  # noqa: E402
from postprocess.twin_rule import TwinRule  # noqa: E402


TAG = "approach18_depth5_20260911"
# 3차/v7 계열에서 train OOF만으로 찾고 복원해 둔 고정 배율. test 통계는 사용하지 않는다.
SCALES = {
    "ACC": 0.5, "BLCA": 1.2, "BRCA": 0.5, "CESC": 1.5, "COAD": 0.5,
    "DLBC": 2.0, "GBMLGG": 0.7, "HNSC": 0.7, "KIPAN": 1.0, "KIRC": 2.5,
    "LAML": 0.7, "LGG": 1.2, "LIHC": 1.0, "LUAD": 1.2, "LUSC": 2.0,
    "OV": 0.85, "PAAD": 1.2, "PCPG": 1.2, "PRAD": 2.0, "SARC": 3.0,
    "SKCM": 0.5, "STES": 0.5, "TGCT": 0.7, "THCA": 0.5, "THYM": 2.5,
    "UCEC": 1.0,
}


def validate(sub: pd.DataFrame, sample: pd.DataFrame, labels: set[str]) -> None:
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)
    assert (sub[ID] == sample[ID]).all()
    assert sub[TARGET].notna().all()
    assert set(sub[TARGET]) <= labels


def main() -> None:
    # 1. train 전체로만 피처·모델을 확정한다.
    train = load_train()
    le = LabelEncoder()
    y = le.fit_transform(train[TARGET])
    params = {**XGB_PARAMS, "colsample_bytree": 0.7, "max_depth": 5, "random_state": SEED}
    fm = FeatureMaker("v4").fit(train)
    model = xgb.XGBClassifier(**params).fit(fm.transform(train), y)

    # 2. 이 시점에서만 test를 읽어 고정된 변환·모델을 적용한다.
    test = load_test()
    proba = model.predict_proba(fm.transform(test))
    scales = np.array([SCALES[c] for c in le.classes_], dtype=np.float64)
    pred = le.inverse_transform((proba * scales).argmax(1))

    sample = pd.read_csv(DATA / "sample_submission.csv")
    out_dir = ROOT / "5. submissions"
    out_dir.mkdir(exist_ok=True)
    sub = sample.copy()
    sub[TARGET] = pred
    validate(sub, sample, set(train[TARGET]))
    base_path = out_dir / f"{TAG}.csv"
    sub.to_csv(base_path, index=False, encoding="UTF-8-sig")

    # 3. 규칙 미적용 기본본과 별도로만 쌍둥이 규칙본을 만든다.
    pred_tw, n_hit = TwinRule().fit(train).apply(test, pred)
    sub_tw = sample.copy()
    sub_tw[TARGET] = pred_tw
    validate(sub_tw, sample, set(train[TARGET]))
    twin_path = out_dir / f"{TAG}_twin_rule.csv"
    sub_tw.to_csv(twin_path, index=False, encoding="UTF-8-sig")

    print("base:", base_path)
    print("twin:", twin_path)
    print(f"rows={len(sub)}, labels={sub[TARGET].nunique()}, twin_matches={n_hit}, changed={(pred != pred_tw).sum()}")


if __name__ == "__main__":
    main()
