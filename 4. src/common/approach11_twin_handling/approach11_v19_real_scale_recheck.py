"""approach11 v19: v14~v18에서 잘못 쓴 "즉석 계산 배율" 대신, 실제로 모든 제출(v7~v12, 혜림님 v19)에
쓰인 진짜 고정 배율 파일(6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json)을
기준선으로 다시 계산. v11 캐시의 oof_main(v7=features v4+colsample0.7)에 이 고정 배율을 곱하는 게
실제 v7 제출 스크립트가 하는 것과 정확히 같은 방식(--class-scale-file 옵션).
"""
import json
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

# ★ 실제 고정 배율 파일 (성민님 v7~v12, 혜림님 v19가 전부 쓴 진짜 값)
real_scales_dict = json.load(open(ROOT / "6. experiments" / "2026-09-09_v4_xgb_cs" / "class_scales_recovered.json"))
real_scales = np.array([float(real_scales_dict[c]) for c in le.classes_])
print("실제 고정 배율(일부):", {c: round(real_scales_dict[c], 2) for c in ["GBMLGG", "LGG", "KIRC", "KIPAN"]})

genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets_all = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets_all).apply(lambda s: len(s) > 1)

summary = []
for sd in SEEDS:
    oof_main = cache[sd]["oof_main"]
    baseline_pred = (oof_main * real_scales).argmax(1)
    baseline_f1 = f1_score(y_all, baseline_pred, average="macro")

    # GBM/KID 배율만 1.0으로 끈 버전(라우팅에 가장 유리한 조건)
    scales_gbm_off = real_scales.copy(); scales_gbm_off[gbm_idx["GBMLGG"]] = 1.0; scales_gbm_off[gbm_idx["LGG"]] = 1.0
    scales_kid_off = real_scales.copy(); scales_kid_off[kid_idx["KIRC"]] = 1.0; scales_kid_off[kid_idx["KIPAN"]] = 1.0
    scales_both_off = real_scales.copy()
    scales_both_off[gbm_idx["GBMLGG"]] = 1.0; scales_both_off[gbm_idx["LGG"]] = 1.0
    scales_both_off[kid_idx["KIRC"]] = 1.0; scales_both_off[kid_idx["KIPAN"]] = 1.0

    main_pred_gbm = (oof_main * scales_gbm_off).argmax(1)
    main_pred_kid = (oof_main * scales_kid_off).argmax(1)
    main_pred_both = (oof_main * scales_both_off).argmax(1)

    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
    oof_a = np.full(len(train), -1)
    oof_b = np.full(len(train), -1)
    oof_c = np.full(len(train), -1)

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()

        tr_gbm = tr_part[tr_part[TARGET].isin(PAIR_GBM) & ~is_twin_all.loc[tr_part.index]]
        spec_g = None
        if len(tr_gbm) >= 10:
            Xg_tr = (tr_gbm[GBM_MARKERS] != "WT").astype(int).to_numpy()
            yg_tr = (tr_gbm[TARGET] == "LGG").astype(int).to_numpy()
            wg = class_weights(yg_tr)
            spec_g = LogisticRegression(max_iter=1000).fit(Xg_tr, yg_tr, sample_weight=wg)

        tr_kid = tr_part[tr_part[TARGET].isin(PAIR_KID) & ~is_twin_all.loc[tr_part.index]]
        spec_k, top_idx = None, None
        if len(tr_kid) >= 10:
            Xk_full_tr = (tr_kid[genes] != "WT").astype(int).to_numpy()
            yk_tr = (tr_kid[TARGET] == "KIRC").astype(int).to_numpy()
            ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(Xk_full_tr, yk_tr)
            top_idx = np.argsort(-ranker.feature_importances_)[:K_KID]
            spec_k = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xk_full_tr[:, top_idx], yk_tr)

        def route_gbm(pred_slice):
            out = pred_slice.copy()
            if spec_g is None: return out
            m = np.isin(pred_slice, [gbm_idx["GBMLGG"], gbm_idx["LGG"]]) & (~va_is_twin)
            idx = np.flatnonzero(m)
            if len(idx):
                Xva = (va_part.iloc[idx][GBM_MARKERS] != "WT").astype(int).to_numpy()
                p = spec_g.predict(Xva)
                out[idx] = np.where(p == 1, gbm_idx["LGG"], gbm_idx["GBMLGG"])
            return out

        def route_kid(pred_slice):
            out = pred_slice.copy()
            if spec_k is None: return out
            m = np.isin(pred_slice, [kid_idx["KIRC"], kid_idx["KIPAN"]]) & (~va_is_twin)
            idx = np.flatnonzero(m)
            if len(idx):
                Xva = (va_part.iloc[idx][genes] != "WT").astype(int).to_numpy()[:, top_idx]
                p = spec_k.predict(Xva)
                out[idx] = np.where(p == 1, kid_idx["KIRC"], kid_idx["KIPAN"])
            return out

        oof_a[vai] = route_gbm(main_pred_gbm[vai])
        oof_b[vai] = route_kid(main_pred_kid[vai])
        oof_c[vai] = route_kid(route_gbm(main_pred_both[vai]))
        print(f"  seed{sd} fold{k} 완료", flush=True)

    f1_a = f1_score(y_all, oof_a, average="macro")
    f1_b = f1_score(y_all, oof_b, average="macro")
    f1_c = f1_score(y_all, oof_c, average="macro")
    row = dict(seed=sd, baseline=round(baseline_f1, 5), A_gbm=round(f1_a, 5), B_kid=round(f1_b, 5), C_both=round(f1_c, 5))
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
df["A_vs_base"] = df["A_gbm"] - df["baseline"]
df["B_vs_base"] = df["B_kid"] - df["baseline"]
df["C_vs_base"] = df["C_both"] - df["baseline"]
print("\n=== 실제 고정 배율 기준 절대값 비교 ===")
print(df.to_string(index=False))
print("\n=== 델타 ===")
for col in ["A_vs_base", "B_vs_base", "C_vs_base"]:
    m, s = df[col].mean(), df[col].std(ddof=1)
    wins = int((df[col] > 0).sum())
    print(f"{col}: mean={m:.5f} std={s:.5f} mean/std={m/s:.2f} 기준선보다 높은 시드={wins}/5")
