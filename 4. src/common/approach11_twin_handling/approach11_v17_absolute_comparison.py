"""approach11 v17: "관계 확인용"이 아니라 "실제로 어떤 조합이 절대적으로 CV를 가장 높이는가"만 본다.
기준선 = class_scale 정상 적용, 라우팅 전혀 없음 (지금 성민님/혜림님 방식과 동일한 최선).
비교 대상:
  A) GBMLGG/LGG만 배율 끄고 라우팅
  B) KIRC/KIPAN만 배율 끄고 라우팅
  C) 둘 다 배율 끄고 둘 다 라우팅
전부 기준선(순수 class_scale)의 절대 macro F1과 직접 비교 — 델타가 양수여도 기준선을 못 넘으면 의미 없음.
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
    scales_full = fit_class_scales(oof_main, y_all)

    # 기준선: class_scale 정상 적용, 라우팅 없음
    baseline_pred = (oof_main * scales_full).argmax(1)
    baseline_f1 = f1_score(y_all, baseline_pred, average="macro")

    # A) GBM만 배율 끔
    scales_a = scales_full.copy(); scales_a[gbm_idx["GBMLGG"]] = 1.0; scales_a[gbm_idx["LGG"]] = 1.0
    main_pred_a = (oof_main * scales_a).argmax(1)
    # B) KID만 배율 끔
    scales_b = scales_full.copy(); scales_b[kid_idx["KIRC"]] = 1.0; scales_b[kid_idx["KIPAN"]] = 1.0
    main_pred_b = (oof_main * scales_b).argmax(1)
    # C) 둘 다 배율 끔
    scales_c = scales_full.copy()
    scales_c[gbm_idx["GBMLGG"]] = 1.0; scales_c[gbm_idx["LGG"]] = 1.0
    scales_c[kid_idx["KIRC"]] = 1.0; scales_c[kid_idx["KIPAN"]] = 1.0
    main_pred_c = (oof_main * scales_c).argmax(1)

    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)

    oof_a = np.full(len(train), -1)   # GBM 라우팅만 (A 배율 기준)
    oof_b = np.full(len(train), -1)   # KID 라우팅만 (B 배율 기준)
    oof_c = np.full(len(train), -1)   # 둘 다 라우팅 (C 배율 기준)

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

        def route_gbm(main_pred_slice):
            out = main_pred_slice.copy()
            if spec_g is None:
                return out
            route_mask = np.isin(main_pred_slice, [gbm_idx["GBMLGG"], gbm_idx["LGG"]]) & (~va_is_twin)
            idx = np.flatnonzero(route_mask)
            if len(idx):
                Xva = (va_part.iloc[idx][GBM_MARKERS] != "WT").astype(int).to_numpy()
                pred = spec_g.predict(Xva)
                out[idx] = np.where(pred == 1, gbm_idx["LGG"], gbm_idx["GBMLGG"])
            return out

        def route_kid(main_pred_slice):
            out = main_pred_slice.copy()
            if spec_k is None:
                return out
            route_mask = np.isin(main_pred_slice, [kid_idx["KIRC"], kid_idx["KIPAN"]]) & (~va_is_twin)
            idx = np.flatnonzero(route_mask)
            if len(idx):
                Xva = (va_part.iloc[idx][genes] != "WT").astype(int).to_numpy()[:, top_idx]
                pred = spec_k.predict(Xva)
                out[idx] = np.where(pred == 1, kid_idx["KIRC"], kid_idx["KIPAN"])
            return out

        oof_a[vai] = route_gbm(main_pred_a[vai])
        oof_b[vai] = route_kid(main_pred_b[vai])
        oof_c[vai] = route_kid(route_gbm(main_pred_c[vai]))
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
print("\n=== 절대값 비교 (기준선 = class_scale만, 라우팅 없음) ===")
print(df.to_string(index=False))
print("\n=== 기준선 대비 델타 ===")
for col in ["A_vs_base", "B_vs_base", "C_vs_base"]:
    m, s = df[col].mean(), df[col].std(ddof=1)
    wins = int((df[col] > 0).sum())
    print(f"{col}: mean={m:.5f} std={s:.5f} mean/std={m/s:.2f}  기준선보다 높은 시드={wins}/5")
