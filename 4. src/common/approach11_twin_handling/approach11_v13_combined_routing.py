"""approach11 v13: v11 캐시(메인 모델 예측) 재사용 — GBMLGG/LGG 저차원 서브모델(v7)과
KIRC/KIPAN 저차원 서브모델(v10e/v10f, XGBoost중요도 top-10)을 동시에 붙였을 때
효과가 누적되는지(+0.0007+0.0007≈+0.0014) 확인. 메인 모델 재학습 없음.
"""
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, XGB_PARAMS, TARGET, twin_groups, load_train, class_weights

GBM_MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
PAIR_GBM = ["GBMLGG", "LGG"]
PAIR_KID = ["KIRC", "KIPAN"]
SEEDS = [42, 7, 123, 2024, 999]
K_KID = 10

cache_path = ROOT / "6. experiments" / "2026-09-12_approach11_v11_integration" / "oof_main_cache.pkl"
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

train = load_train()
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
gbm_idx = {c: le.transform([c])[0] for c in PAIR_GBM}
kid_idx = {c: le.transform([c])[0] for c in PAIR_KID}

genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets_all = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets_all).apply(lambda s: len(s) > 1)

summary = []
for sd in SEEDS:
    oof_main = cache[sd]["oof_main"]
    main_pred_all = oof_main.argmax(1)
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)

    oof_gbm_only = np.full(len(train), -1)
    oof_kid_only = np.full(len(train), -1)
    oof_combined = np.full(len(train), -1)

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        main_pred = main_pred_all[vai]
        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()

        # --- GBMLGG/LGG 서브모델 (v7과 동일: 6개 마커) ---
        tr_gbm = tr_part[tr_part[TARGET].isin(PAIR_GBM) & ~is_twin_all.loc[tr_part.index]]
        gbm_pred_route = main_pred.copy()
        if len(tr_gbm) >= 10:
            Xg_tr = (tr_gbm[GBM_MARKERS] != "WT").astype(int).to_numpy()
            yg_tr = (tr_gbm[TARGET] == "LGG").astype(int).to_numpy()
            wg = class_weights(yg_tr)
            spec_g = LogisticRegression(max_iter=1000).fit(Xg_tr, yg_tr, sample_weight=wg)
            route_g = np.isin(main_pred, [gbm_idx["GBMLGG"], gbm_idx["LGG"]]) & (~va_is_twin)
            Xg_va = (va_part[GBM_MARKERS] != "WT").astype(int).to_numpy()
            pred_g = spec_g.predict(Xg_va)
            idxg = np.flatnonzero(route_g)
            gbm_pred_route[idxg] = np.where(pred_g[idxg] == 1, gbm_idx["LGG"], gbm_idx["GBMLGG"])

        # --- KIRC/KIPAN 서브모델 (v10e/v10f 최선: XGBoost중요도 top-10) ---
        tr_kid = tr_part[tr_part[TARGET].isin(PAIR_KID) & ~is_twin_all.loc[tr_part.index]]
        kid_pred_route = main_pred.copy()
        if len(tr_kid) >= 10:
            Xk_full_tr = (tr_kid[genes] != "WT").astype(int).to_numpy()
            yk_tr = (tr_kid[TARGET] == "KIRC").astype(int).to_numpy()
            ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(Xk_full_tr, yk_tr)
            top_idx = np.argsort(-ranker.feature_importances_)[:K_KID]
            spec_k = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xk_full_tr[:, top_idx], yk_tr)
            route_k = np.isin(main_pred, [kid_idx["KIRC"], kid_idx["KIPAN"]]) & (~va_is_twin)
            Xk_full_va = (va_part[genes] != "WT").astype(int).to_numpy()
            pred_k = spec_k.predict(Xk_full_va[:, top_idx])
            idxk = np.flatnonzero(route_k)
            kid_pred_route[idxk] = np.where(pred_k[idxk] == 1, kid_idx["KIRC"], kid_idx["KIPAN"])

        # --- 결합: 두 라우팅을 동시에 적용 (서로 다른 클래스라 겹치지 않음) ---
        combined = main_pred.copy()
        if len(tr_gbm) >= 10:
            combined[idxg] = gbm_pred_route[idxg]
        if len(tr_kid) >= 10:
            combined[idxk] = kid_pred_route[idxk]

        oof_gbm_only[vai] = gbm_pred_route
        oof_kid_only[vai] = kid_pred_route
        oof_combined[vai] = combined
        print(f"  seed{sd} fold{k} 완료", flush=True)

    row = dict(
        seed=sd,
        main=round(f1_score(y_all, main_pred_all, average="macro"), 4),
        gbm_only=round(f1_score(y_all, oof_gbm_only, average="macro"), 4),
        kid_only=round(f1_score(y_all, oof_kid_only, average="macro"), 4),
        combined=round(f1_score(y_all, oof_combined, average="macro"), 4),
    )
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
print("\n=== 시드별 결과 ===")
print(df.to_string(index=False))
print("\n=== 평균 / 표준편차 ===")
for col in ["main", "gbm_only", "kid_only", "combined"]:
    print(f"{col}: mean={df[col].mean():.4f} std={df[col].std():.4f}")
