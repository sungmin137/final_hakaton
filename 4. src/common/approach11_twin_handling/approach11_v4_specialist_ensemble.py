"""GBMLGG/LGG 전용 서브모델을 메인 26클래스 모델에 실제로 적용했을 때 효과 검증.
규칙: 메인 모델의 예측이 GBMLGG 또는 LGG일 때만, 전용 서브모델의 판단으로 덮어씀
(정답을 몰라도 메인 모델의 1차 예측만으로 라우팅 가능 — 실제 추론에서도 쓸 수 있는 형태).
04_todo_day2.md STEP 4 참고.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, TARGET, FeatureMaker, XGB_PARAMS, SEED, twin_groups, load_train, class_weights

train = load_train()
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
groups = twin_groups(train)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_main = np.zeros((len(train), len(le.classes_)))
oof_spec = np.full(len(train), -1)  # -1=해당없음, 0=GBMLGG로 판정, 1=LGG로 판정

PAIR = ["GBMLGG", "LGG"]
pair_idx = {c: le.transform([c])[0] for c in PAIR}

for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
    tr_part, va_part = train.iloc[tri], train.iloc[vai]

    # 메인 모델(26클래스)
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
    main_model = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr, y[tri])
    oof_main[vai] = main_model.predict_proba(Xva)

    # 전용 서브모델(GBMLGG/LGG만, 가중치 적용) — 학습은 이 쌍만, 평가는 va_part 전체
    tr_pair = tr_part[tr_part[TARGET].isin(PAIR)]
    y_pair = (tr_pair[TARGET] == "LGG").astype(int).to_numpy()  # 0=GBMLGG, 1=LGG
    fm2 = FeatureMaker("v4").fit(tr_pair)
    Xtr2 = fm2.transform(tr_pair)
    Xva2 = fm2.transform(va_part)
    w = class_weights(y_pair)
    spec_model = xgb.XGBClassifier(**{**XGB_PARAMS, "objective": "binary:logistic"}).fit(Xtr2, y_pair, sample_weight=w)
    spec_pred = spec_model.predict(Xva2)  # 0=GBMLGG, 1=LGG
    oof_spec[vai] = spec_pred
    print(f"fold{k} 완료")

# 메인 모델 argmax 예측
main_pred = oof_main.argmax(1)

# 앙상블: 메인 예측이 GBMLGG/LGG일 때만 서브모델로 덮어씀
combined_pred = main_pred.copy()
route_mask = np.isin(main_pred, [pair_idx["GBMLGG"], pair_idx["LGG"]])
print(f"\n메인 모델이 GBMLGG/LGG로 예측한 행 수: {route_mask.sum()}")
combined_pred[route_mask] = np.where(oof_spec[route_mask] == 0, pair_idx["GBMLGG"], pair_idx["LGG"])

macro_main = f1_score(y, main_pred, average="macro")
macro_combined = f1_score(y, combined_pred, average="macro")
print(f"\n메인 모델 단독 Macro F1: {macro_main:.4f}")
print(f"서브모델 적용(라우팅) 후 Macro F1: {macro_combined:.4f}")

pc_main = dict(zip(le.classes_, f1_score(y, main_pred, average=None).round(4).tolist()))
pc_comb = dict(zip(le.classes_, f1_score(y, combined_pred, average=None).round(4).tolist()))
for c in ["GBMLGG", "LGG", "KIRC", "KIPAN"]:
    print(f"  {c}: 메인 {pc_main[c]} -> 적용 후 {pc_comb[c]}")

out_dir = ROOT / "6. experiments" / "2026-09-10_approach11_v4_specialist_ensemble"
out_dir.mkdir(parents=True, exist_ok=True)
res = dict(features="v4", note="GBMLGG/LGG 라우팅 앙상블: 메인 예측이 GBMLGG/LGG일 때만 서브모델로 덮어씀",
           main_only_macro_f1=round(macro_main, 4), combined_macro_f1=round(macro_combined, 4),
           routed_rows=int(route_mask.sum()),
           per_class_main=pc_main, per_class_combined=pc_comb)
(out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
np.save(out_dir / "oof_main_proba.npy", oof_main)
pd.DataFrame(confusion_matrix(y, combined_pred), index=le.classes_, columns=le.classes_
             ).to_csv(out_dir / "confusion_matrix_combined.csv")
print(f"\nsaved: {out_dir}")
