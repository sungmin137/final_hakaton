"""접근 9 v1: v3(팀 코드 그대로) + 유전자 역할 오버라이드(TP53 등 6개, missense만). 후처리 없음.
정직 CV 0.4663~0.4681, LB 0.4031845152 (2026-09-09 18:35:37 제출, 원래 파일명
approach4_gene_role_override_v3_20260909-1827.csv — "approach4"는 잘못 붙은 이름, 실제로는
성민님 approach4(문헌 피처)와 무관. 확인 근거: 2. team/approaches/approach9.md).

이 스크립트는 원래 Day1/predict_v35.py를 그대로 옮긴 것 — 이미 제출해서 점수를 받은 코드라
팀 파이프라인(FeatureMaker/make_submission.py)으로 재구현하지 않고 실행 경로만 이 저장소에
맞게 고쳤다. 로직은 4. src/common/approach9_gene_override/의 3개 파일(features_v3_reference.py,
features_v35_override.py, count_weights.py) 그대로.

⚠️ 재실행해도 동일한 csv가 나온다고 보장 못 함(XGBoost hist+멀티스레드 non-determinism으로 추정,
실측 시 원본과 92.5%만 일치). 저장소의 5. submissions/approach9_v1_20260909_1827.csv는 이 스크립트를
돌린 결과가 아니라 2026-09-09 18:35:37에 실제 제출한 원본 파일 그대로다.

실행(참고용): python3 "4. src/submissions_source/approach9_v1_20260909_1827.py"
설명 문서: 2. team/approaches/approach9.md
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "approach9_gene_override"))
from features_v3_reference import ID, TARGET, gene_columns, InsightFeatures  # noqa: E402
from features_v35_override import build_features_v35  # noqa: E402
from count_weights import CountWeightFeatures  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "1. info" / "data"
SEED = 42

XGB_PARAMS = dict(
    n_estimators=100, learning_rate=0.1, max_depth=6, random_state=SEED,
    eval_metric="mlogloss", tree_method="hist", n_jobs=8,
)


def main():
    train = pd.read_csv(DATA / "train.csv")
    genes = gene_columns(train)
    print(f"train {train.shape}")

    le = LabelEncoder()
    y = le.fit_transform(train[TARGET])

    # train 전체로 fit (fold 없음 — 최종 모델)
    insight = InsightFeatures().fit(train)
    cwf = CountWeightFeatures().fit(train)
    Xtr = pd.concat([build_features_v35(train, genes, insight), cwf.transform(train)], axis=1)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr, y)
    print(f"학습 완료, 피처 {Xtr.shape[1]}개")

    # ⚠️ test.csv 최초 로드 — 결측 셀은 공식 파이프라인과 동일하게 WT로 간주
    test = pd.read_csv(DATA / "test.csv").fillna("WT")
    print(f"test {test.shape}")

    Xte = pd.concat([build_features_v35(test, genes, insight), cwf.transform(test)], axis=1)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    pred = model.predict(Xte)
    labels = le.inverse_transform(pred)

    sub = pd.read_csv(DATA / "sample_submission.csv")
    assert (sub[ID] == test[ID]).all(), "ID 순서 불일치"
    sub[TARGET] = labels

    out_dir = ROOT / "5. submissions"
    out = out_dir / "approach9_v1_20260909_1827.csv"
    sub.to_csv(out, index=False, encoding="UTF-8-sig")
    print(f"\n저장: {out}")
    print(sub[TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()
