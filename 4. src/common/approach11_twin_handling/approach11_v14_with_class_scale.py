"""approach11 v14: v11 캐시 재사용 + class_scale까지 적용해서 혜림님(approach19)과 같은 기준선으로 재비교.
메인 모델 재학습 없음. 각 시드마다 캐시된 OOF 확률에 class_scale(좌표상승)을 적용해서 "v7+class_scale"을
기준선으로 삼고, 그 위에 GBMLGG/LGG·KIRC/KIPAN 라우팅을 각각/결합으로 얹어 쌍대비교(delta) 계산.
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
from postprocess.class_scale import fit_class_scales

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
    scales = fit_class_scales(oof_main, y_all)                 # train OOF만으로 배율 산출 (test 미사용)
    scaled_proba = oof_main * scales
    main_pred_all = scaled_proba.argmax(1)                     # ★ 배율 적용 후 argmax = 새 기준선

    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)

    oof_gbm_only = np.full(len(train), -1)
    oof_kid_only = np.full(len(train), -1)
    oof_combined = np.full(len(train), -1)

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        main_pred = main_pred_all[vai]
        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()

        tr_gbm = tr_part[tr_part[TARGET].isin(PAIR_GBM) & ~is_twin_all.loc[tr_part.index]]
        gbm_pred_route = main_pred.copy()
        idxg = np.array([], dtype=int)
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

        tr_kid = tr_part[tr_part[TARGET].isin(PAIR_KID) & ~is_twin_all.loc[tr_part.index]]
        kid_pred_route = main_pred.copy()
        idxk = np.array([], dtype=int)
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

        combined = main_pred.copy()
        combined[idxg] = gbm_pred_route[idxg]
        combined[idxk] = kid_pred_route[idxk]

        oof_gbm_only[vai] = gbm_pred_route
        oof_kid_only[vai] = kid_pred_route
        oof_combined[vai] = combined
        print(f"  seed{sd} fold{k} 완료", flush=True)

    row = dict(
        seed=sd,
        main_scaled=round(f1_score(y_all, main_pred_all, average="macro"), 5),
        gbm_only=round(f1_score(y_all, oof_gbm_only, average="macro"), 5),
        kid_only=round(f1_score(y_all, oof_kid_only, average="macro"), 5),
        combined=round(f1_score(y_all, oof_combined, average="macro"), 5),
    )
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
df["delta_gbm"] = df["gbm_only"] - df["main_scaled"]
df["delta_kid"] = df["kid_only"] - df["main_scaled"]
df["delta_combined"] = df["combined"] - df["main_scaled"]

print("\n=== 시드별 결과 ===")
print(df.to_string(index=False))
print("\n=== 쌍대비교(delta) 평균/표준편차 ===")
for col in ["delta_gbm", "delta_kid", "delta_combined"]:
    m, s = df[col].mean(), df[col].std(ddof=1)
    print(f"{col}: mean={m:.5f}  std={s:.5f}  mean/std={m/s:.2f}  양수 시드={int((df[col]>0).sum())}/5")
