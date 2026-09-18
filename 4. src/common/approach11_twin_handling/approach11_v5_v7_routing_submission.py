"""approach11 v5: v7(colsample_bytree 0.7, 복원 배율, LB 0.4377 최고) 설정에
GBMLGG/LGG 전용 라우팅 앙상블(approach11_v4, CV +0.0012, 3회 재현 확인)을 실제 test 추론에 적용.
메인 모델 예측이 GBMLGG/LGG일 때만 전용 서브모델 판정으로 덮어씀 — approach11_v4와 동일 규칙.
test.csv는 이 파일에서만 읽는다(진입 스크립트가 이 main()을 호출).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from main import DATA, ID, ROOT, TARGET, FeatureMaker, PARAM_SETS, class_weights, load_test, load_train
from postprocess.twin_rule import TwinRule

PAIR = ["GBMLGG", "LGG"]
SCALE_FILE = ROOT / "6. experiments" / "2026-09-09_v4_xgb_cs" / "class_scales_recovered.json"


def validate(sub: pd.DataFrame, sample: pd.DataFrame, train_labels: set) -> None:
    assert list(sub.columns) == list(sample.columns), "컬럼 불일치"
    assert len(sub) == len(sample), "행 수 불일치"
    assert (sub[ID] == sample[ID]).all(), "ID 순서 불일치"
    assert sub[TARGET].notna().all(), "라벨 결측"
    assert set(sub[TARGET]) <= train_labels, "train에 없는 라벨"


def main(tag: str) -> None:
    train = load_train()
    test = load_test()
    le = LabelEncoder()
    y = le.fit_transform(train[TARGET])
    params = PARAM_SETS["mild_col"]

    # 메인 모델: v7과 동일 설정 (features v4, colsample_bytree 0.7)
    fm = FeatureMaker("v4").fit(train)
    model = xgb.XGBClassifier(**params).fit(fm.transform(train), y)
    proba = model.predict_proba(fm.transform(test))

    # v7과 동일한 복원 배율 적용
    sc = json.load(open(SCALE_FILE))
    scales = np.array([float(sc[c]) for c in le.classes_])
    proba = proba * scales
    main_pred = proba.argmax(1)

    # GBMLGG/LGG 전용 서브모델 (approach11_v4와 동일 구조, train 전체로 학습)
    pair_idx = {c: le.transform([c])[0] for c in PAIR}
    tr_pair = train[train[TARGET].isin(PAIR)]
    y_pair = (tr_pair[TARGET] == "LGG").astype(int).to_numpy()  # 0=GBMLGG, 1=LGG
    fm2 = FeatureMaker("v4").fit(tr_pair)
    w = class_weights(y_pair)
    spec_model = xgb.XGBClassifier(**{**params, "objective": "binary:logistic"}
                                    ).fit(fm2.transform(tr_pair), y_pair, sample_weight=w)
    spec_pred = spec_model.predict(fm2.transform(test))  # 0=GBMLGG, 1=LGG

    combined_pred = main_pred.copy()
    route_mask = np.isin(main_pred, [pair_idx["GBMLGG"], pair_idx["LGG"]])
    combined_pred[route_mask] = np.where(spec_pred[route_mask] == 0, pair_idx["GBMLGG"], pair_idx["LGG"])
    print(f"[route] test 중 GBMLGG/LGG로 예측된 행: {route_mask.sum()} / {len(test)}")
    print(f"[route] 서브모델이 메인 예측을 바꾼 행: {(combined_pred[route_mask] != main_pred[route_mask]).sum()}")

    pred = le.inverse_transform(combined_pred)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    labels = set(train[TARGET])

    out_dir = ROOT / "5. submissions"
    sub = sample.copy(); sub[TARGET] = pred
    validate(sub, sample, labels)
    sub.to_csv(out_dir / f"{tag}.csv", index=False, encoding="UTF-8-sig")

    pred_tw, n_hit = TwinRule().fit(train).apply(test, pred)
    sub_tw = sample.copy(); sub_tw[TARGET] = pred_tw
    validate(sub_tw, sample, labels)
    sub_tw.to_csv(out_dir / f"{tag}_twin_rule.csv", index=False, encoding="UTF-8-sig")

    exp = ROOT / "6. experiments" / "submissions" / tag
    exp.mkdir(parents=True, exist_ok=True)
    np.save(exp / "test_proba_main.npy", proba)
    print(f"[test] train과 완전 동일 행 {n_hit} ({n_hit/len(test):.1%}), 규칙으로 바뀐 예측 {(pred != pred_tw).sum()}행")
    print(f"saved: 5. submissions/{tag}.csv, {tag}_twin_rule.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "approach11_v5_manual_run")
