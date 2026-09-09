"""주최측 공식 베이스라인 ([Baseline]_XGB를 활용한 암종 분류 AI 모델 개발.ipynb) 을 그대로 옮긴 스크립트.
데이터 경로만 이 저장소 구조(info/data)에 맞췄다. 검증 없이 학습 → test 예측 → 제출 파일.
정직 5-Fold로 측정한 이 방식의 Macro F1은 약 0.30 (docs/experiments_log.md).

실행: python3 info/baseline.py   → submissions/baseline_submission.csv
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "info" / "data"

# Load Data
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

# Data Preprocessing — SUBCLASS 라벨 인코딩, 유전자 컬럼 OrdinalEncoder (train으로만 fit)
le_subclass = LabelEncoder()
train["SUBCLASS"] = le_subclass.fit_transform(train["SUBCLASS"])
X = train.drop(columns=["SUBCLASS", "ID"])
y_subclass = train["SUBCLASS"]
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# Model Define and Train
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric="mlogloss")
model.fit(X_encoded, y_subclass)

# Inference
test_X = test.drop(columns=["ID"]).fillna("WT")          # test 결측은 상수 WT (test 통계 미사용)
X_test_encoded = test_X.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test_X[categorical_columns])
predictions = model.predict(X_test_encoded)
original_labels = le_subclass.inverse_transform(predictions)

# Submission
submission = pd.read_csv(DATA / "sample_submission.csv")
submission["SUBCLASS"] = original_labels
out = ROOT / "submissions" / "baseline_submission.csv"
submission.to_csv(out, encoding="UTF-8-sig", index=False)
print("saved", out)
