"""접근10 최종 추론 — 여기서만 test.csv를 읽는다.

기존 최고 제출 구성(V4 + train OOF 클래스 배율 + TwinRule)에 접근10 재판기를
더한다. 재판기의 발동 조건은 CV에서 검증한 '보정 전 V4 확률 상위 2개가 혼동 쌍,
차이 0.30 이하'다. test 라벨·분포·통계는 어떤 결정에도 사용하지 않는다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from approach10_pair_referee_cv import PAIR_NAMES, pair_mask
from main import DATA, ID, PARAM_SETS, ROOT, TARGET, FeatureMaker, gene_columns, load_test, load_train
from postprocess.class_scale import fit_class_scales
from postprocess.twin_rule import TwinRule

TAG = "approach10_pair_referee_v1_20260910"
GAP_THRESHOLD = 0.30
V4_OOF_DIR = ROOT / "6. experiments" / "2026-09-09_v4_xgb_grp"


def validate(sub: pd.DataFrame, sample: pd.DataFrame, labels: set[str]) -> None:
    assert list(sub.columns) == list(sample.columns), "컬럼 불일치"
    assert len(sub) == len(sample), "행 수 불일치"
    assert (sub[ID] == sample[ID]).all(), "ID 순서 불일치"
    assert sub[TARGET].notna().all(), "라벨 결측"
    assert set(sub[TARGET]) <= labels, "train에 없는 라벨"


def main() -> None:
    t0 = time.time()
    train = load_train()
    le = LabelEncoder()
    y = le.fit_transform(train[TARGET])
    class_id = {name: i for i, name in enumerate(le.classes_)}
    pairs = [(class_id[a], class_id[b]) for a, b in PAIR_NAMES]

    # 1. train 전체로 기존 V4를 학습한다.
    fm = FeatureMaker("v4").fit(train)
    model = xgb.XGBClassifier(**PARAM_SETS["official"])
    model.fit(fm.transform(train), y)
    print(f"[train] V4 {len(fm.columns)}개 피처 학습 완료 ({time.time() - t0:.0f}s)")

    # 2. 이 시점에서만 test를 연다. 결측은 기존 파이프라인과 같이 WT로만 처리한다.
    test = load_test()
    raw_proba = model.predict_proba(fm.transform(test))

    # 3. 클래스 배율은 과거 V4의 train OOF와 train 라벨로만 맞춘다.
    oof = np.load(V4_OOF_DIR / "oof_proba.npy")
    scales = fit_class_scales(oof, y)
    pred_id = (raw_proba * scales).argmax(1).astype(np.int16)
    print("[post] class scale 적용")

    # 4. 접근10 재판기. 원본 유전자 변이 유무만 사용하며, gate는 CV와 똑같이
    #    클래스 보정 전 V4의 상위 두 확률과 0.30 문턱으로 정한다.
    genes = gene_columns(train)
    X_train_binary = (train[genes].to_numpy() != "WT").astype(np.int8)
    X_test_binary = (test[genes].to_numpy() != "WT").astype(np.int8)
    order = np.argsort(raw_proba, axis=1)
    top1, top2 = order[:, -1], order[:, -2]
    gap = raw_proba[np.arange(len(test)), top1] - raw_proba[np.arange(len(test)), top2]
    before_referee = pred_id.copy()

    for (a, b), (a_name, b_name) in zip(pairs, PAIR_NAMES):
        pair_train = np.flatnonzero((y == a) | (y == b))
        judge = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        judge.fit(X_train_binary[pair_train], y[pair_train])
        eligible = pair_mask(top1, top2, a, b) & (gap <= GAP_THRESHOLD)
        if eligible.any():
            pred_id[eligible] = judge.predict(X_test_binary[eligible]).astype(np.int16)
        print(f"[referee] {a_name} vs {b_name}: {int(eligible.sum())}명")

    pred = le.inverse_transform(pred_id)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = pred
    validate(sub, sample, set(train[TARGET]))

    out_dir = ROOT / "5. submissions"
    out_dir.mkdir(exist_ok=True)
    base_path = out_dir / f"{TAG}.csv"
    sub.to_csv(base_path, index=False, encoding="UTF-8-sig")

    # 5. historical best 구성과 동일하게 쌍둥이 규칙 적용본도 별도로 저장한다.
    twin_pred, n_twin_hit = TwinRule().fit(train).apply(test, pred)
    sub_twin = sample.copy()
    sub_twin[TARGET] = twin_pred
    validate(sub_twin, sample, set(train[TARGET]))
    twin_path = out_dir / f"{TAG}_twin_rule.csv"
    sub_twin.to_csv(twin_path, index=False, encoding="UTF-8-sig")

    print(f"[result] 재판기가 클래스 보정 답을 바꾼 행: {(before_referee != pred_id).sum()}")
    print(f"[result] 쌍둥이 일치 행: {n_twin_hit}, 쌍둥이 규칙으로 바뀐 답: {(pred != twin_pred).sum()}")
    print(f"[result] 확신도 중앙값: {np.median((raw_proba * scales).max(1)):.3f}")
    print("saved:", base_path)
    print("saved:", twin_path)


if __name__ == "__main__":
    main()
