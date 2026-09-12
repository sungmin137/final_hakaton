"""approach11 v5(v7 파라미터 + 라우팅) 실제 test 제출 전, 정직 CV로 과적합 여부 확인용 스크립트.
train만 사용 (test.csv 안 읽음). approach11_v4와 같은 구조에 PARAM_SETS["mild_col"]만 적용.
"""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import PARAM_SETS, twin_groups, load_train, class_weights, FeatureMaker
import xgboost as xgb

train = load_train()
le = LabelEncoder()
y = le.fit_transform(train["SUBCLASS"] if "SUBCLASS" in train.columns else train[[c for c in train.columns if c not in ("ID",)][-1]])
from main import TARGET
y = le.fit_transform(train[TARGET])
groups = twin_groups(train)
params = PARAM_SETS["mild_col"]

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_main = np.zeros((len(train), len(le.classes_)))
oof_spec = np.full(len(train), -1)
PAIR = ["GBMLGG", "LGG"]
pair_idx = {c: le.transform([c])[0] for c in PAIR}

for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
    main_model = xgb.XGBClassifier(**params).fit(Xtr, y[tri])
    oof_main[vai] = main_model.predict_proba(Xva)

    tr_pair = tr_part[tr_part[TARGET].isin(PAIR)]
    y_pair = (tr_pair[TARGET] == "LGG").astype(int).to_numpy()
    fm2 = FeatureMaker("v4").fit(tr_pair)
    Xtr2, Xva2 = fm2.transform(tr_pair), fm2.transform(va_part)
    w = class_weights(y_pair)
    spec_model = xgb.XGBClassifier(**{**params, "objective": "binary:logistic"}).fit(Xtr2, y_pair, sample_weight=w)
    oof_spec[vai] = spec_model.predict(Xva2)
    print(f"fold{k} 완료")

main_pred = oof_main.argmax(1)
combined_pred = main_pred.copy()
route_mask = np.isin(main_pred, [pair_idx["GBMLGG"], pair_idx["LGG"]])
combined_pred[route_mask] = np.where(oof_spec[route_mask] == 0, pair_idx["GBMLGG"], pair_idx["LGG"])

macro_main = f1_score(y, main_pred, average="macro")
macro_combined = f1_score(y, combined_pred, average="macro")
print(f"\nmild_col(v7 파라미터) 메인 단독 정직 CV Macro F1: {macro_main:.4f}")
print(f"mild_col + 라우팅 앙상블 정직 CV Macro F1: {macro_combined:.4f}")
print(f"(참고) official 파라미터 v4 baseline: 0.4683 / 라우팅 적용: 0.4695")
