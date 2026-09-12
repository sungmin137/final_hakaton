"""접근19 제출 재현: 최고 LB V7 + 검증 통과한 HNSC↔STES soft 재판.

라우터 설정(C=.1, confidence=.7, 상위 40 유전자)은 train 반복 CV에서 미리 확정했다.
test의 분포·예측 개수로 어떤 값도 조정하지 않는다.
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
from main import DATA, ID, ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, gene_columns, load_test, load_train  # noqa: E402
from postprocess.twin_rule import TwinRule  # noqa: E402

TAG = "approach19_hnsc_stes_soft_20260912"
PAIR = ("HNSC", "STES")
C, CONFIDENCE, TOP_K = 0.1, 0.70, 40
# v7에서 train OOF로만 얻어 둔 고정 class scale
SCALES = {"ACC": .5, "BLCA": 1.2, "BRCA": .5, "CESC": 1.5, "COAD": .5, "DLBC": 2., "GBMLGG": .7,
          "HNSC": .7, "KIPAN": 1., "KIRC": 2.5, "LAML": .7, "LGG": 1.2, "LIHC": 1., "LUAD": 1.2,
          "LUSC": 2., "OV": .85, "PAAD": 1.2, "PCPG": 1.2, "PRAD": 2., "SARC": 3., "SKCM": .5,
          "STES": .5, "TGCT": .7, "THCA": .5, "THYM": 2.5, "UCEC": 1.}


def validate(sub: pd.DataFrame, sample: pd.DataFrame, labels: set[str]) -> None:
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and (sub[ID] == sample[ID]).all()
    assert sub[TARGET].notna().all() and set(sub[TARGET]) <= labels


def main() -> None:
    # 1) 이 시점까지는 train으로만 모든 규칙과 모델을 고정한다.
    train = load_train(); genes = gene_columns(train)
    le = LabelEncoder(); y = le.fit_transform(train[TARGET])
    ids = {name: i for i, name in enumerate(le.classes_)}
    a, b = ids[PAIR[0]], ids[PAIR[1]]
    fm = FeatureMaker("v4").fit(train)
    main_model = xgb.XGBClassifier(**{**XGB_PARAMS, "colsample_bytree": .7, "random_state": SEED})
    main_model.fit(fm.transform(train), y)

    # HNSC/STES만의 간단한 재판기: 전체 train에서 빈도 차가 큰 40개 유전자만 쓴다.
    Xtr = (train[genes].to_numpy() != "WT").astype(np.int8)
    ptr = np.flatnonzero(np.isin(y, (a, b)))
    top = np.argsort(np.abs(Xtr[ptr][y[ptr] == a].mean(0) - Xtr[ptr][y[ptr] == b].mean(0)))[-TOP_K:]
    referee = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED)
    referee.fit(Xtr[ptr][:, top], y[ptr])

    # 2) 고정된 상태에서만 test를 한 번 읽어 추론한다.
    test = load_test()
    main_proba = main_model.predict_proba(fm.transform(test))
    scale = np.array([SCALES[n] for n in le.classes_])
    pred_id = (main_proba * scale).argmax(1)
    Xte = (test[genes].to_numpy() != "WT").astype(np.int8)
    p_ref = referee.predict_proba(Xte[:, top])
    ref_id = referee.classes_[p_ref.argmax(1)]
    # 기존 답이 이 두 암종 중 하나이고, 재판기가 70% 이상 확신할 때만 부드럽게 교체.
    use = np.isin(pred_id, (a, b)) & (p_ref.max(1) >= CONFIDENCE)
    before = pred_id.copy(); pred_id[use] = ref_id[use]
    pred = le.inverse_transform(pred_id)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    out = ROOT / "5. submissions"; out.mkdir(exist_ok=True)
    sub = sample.copy(); sub[TARGET] = pred; validate(sub, sample, set(train[TARGET]))
    base_path = out / f"{TAG}.csv"; sub.to_csv(base_path, index=False, encoding="UTF-8-sig")

    # 쌍둥이 규칙은 별도 파일로 보존한다. 선택 여부는 팀의 공정성 판단에 따른다.
    pred_tw, hits = TwinRule().fit(train).apply(test, pred)
    sub_tw = sample.copy(); sub_tw[TARGET] = pred_tw; validate(sub_tw, sample, set(train[TARGET]))
    twin_path = out / f"{TAG}_twin_rule.csv"; sub_tw.to_csv(twin_path, index=False, encoding="UTF-8-sig")
    print(f"base={base_path}\ntwin={twin_path}")
    print(f"router_candidates={int(use.sum())}, router_changed={int((before != pred_id).sum())}, twin_matches={hits}, twin_changed={int((pred != pred_tw).sum())}")


if __name__ == "__main__":
    main()
