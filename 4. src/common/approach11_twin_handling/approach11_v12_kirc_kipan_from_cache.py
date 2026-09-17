"""approach11 v12: v11에서 캐시해둔 메인 모델 예측을 재사용해서, KIRC/KIPAN 저차원 XGBoost중요도
서브모델(v10e/v10f 기준 LOOCV 0.6493, 이번에 찾은 최선)을 메인 파이프라인에 통합 테스트.
메인 모델 재학습 없음 — fold 구성만 같은 시드로 재현하고, KIRC/KIPAN 서브모델만 새로 학습.
twin_rule은 넣지 않음(정직 CV에서 항상 매칭 0건이라 무의미함, v11에서 이미 확인).
"""
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, XGB_PARAMS, TARGET, twin_groups, load_train

PAIR = ["KIRC", "KIPAN"]
SEEDS = [42, 7, 123, 2024, 999]
K = 10

cache_path = ROOT / "6. experiments" / "2026-09-12_approach11_v11_integration" / "oof_main_cache.pkl"
with open(cache_path, "rb") as f:
    cache = pickle.load(f)
print(f"캐시 로드 완료: 시드 {list(cache.keys())}")

train = load_train()
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
pair_idx = {c: le.transform([c])[0] for c in PAIR}

genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets_all = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets_all).apply(lambda s: len(s) > 1)

summary = []
for sd in SEEDS:
    oof_main = cache[sd]["oof_main"]          # 캐시에서 불러옴 (재학습 없음)
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)   # 같은 시드 → 같은 fold 재현

    oof_routing = np.full(len(train), -1)
    main_pred_all = oof_main.argmax(1)

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        main_pred = main_pred_all[vai]

        tr_pair_mask = tr_part[TARGET].isin(PAIR) & ~is_twin_all.loc[tr_part.index]
        tr_pair = tr_part[tr_pair_mask]
        if len(tr_pair) < 10:
            oof_routing[vai] = main_pred
            continue
        X_tr = (tr_pair[genes] != "WT").astype(int).to_numpy()
        y_tr = (tr_pair[TARGET] == "KIRC").astype(int).to_numpy()

        ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(X_tr, y_tr)
        top_idx = np.argsort(-ranker.feature_importances_)[:K]
        spec = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr[:, top_idx], y_tr)

        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()
        route_mask = np.isin(main_pred, [pair_idx["KIRC"], pair_idx["KIPAN"]]) & (~va_is_twin)
        X_va = (va_part[genes] != "WT").astype(int).to_numpy()[:, top_idx]
        spec_pred = spec.predict(X_va) if len(va_part) else np.array([])

        pred_routing = main_pred.copy()
        idx = np.flatnonzero(route_mask)
        pred_routing[idx] = np.where(spec_pred[idx] == 1, pair_idx["KIRC"], pair_idx["KIPAN"])
        oof_routing[vai] = pred_routing
        print(f"  seed{sd} fold{k} 완료", flush=True)

    macro_main = f1_score(y_all, main_pred_all, average="macro")
    macro_routing = f1_score(y_all, oof_routing, average="macro")
    row = dict(seed=sd, main=round(macro_main, 4), routing=round(macro_routing, 4))
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
print("\n=== 시드별 결과 ===")
print(df.to_string(index=False))
print("\n=== 평균 / 표준편차 ===")
for col in ["main", "routing"]:
    print(f"{col}: mean={df[col].mean():.4f} std={df[col].std():.4f}")
