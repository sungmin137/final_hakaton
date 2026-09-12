"""approach11 v15: 순서를 고쳐서 재검증 — 메인모델 → 라우팅(확률에 섞어넣기) → class_scale.
v14는 순서가 메인모델 → class_scale → 라우팅(하드 덮어쓰기)이라, class_scale이 이미 최적화해놓은 배율 위에
라우팅이 같은 방향으로 한 번 더 밀어서 과잉교정(정밀도 손실 > 재현율 이득)이 난 것으로 추정됨.
이번엔 라우팅을 "하드 덮어쓰기"가 아니라 "해당 2개 클래스 확률 칸을 서브모델 확률로 교체"로 바꿔서,
class_scale이 라우팅 이후의 확률을 보고 새로 배율을 맞추게 함 — 중복 보정을 줄이는 게 목적.
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
n_classes = len(le.classes_)
gbm_idx = {c: le.transform([c])[0] for c in PAIR_GBM}
kid_idx = {c: le.transform([c])[0] for c in PAIR_KID}

genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets_all = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets_all).apply(lambda s: len(s) > 1)


def route_into_proba(oof_main, main_pred_all, pair, pair_idx_map, marker_fn, tri, vai, tr_part, va_part):
    """route 대상 행의 pair_idx_map 두 칸 확률을 서브모델 확률로 교체(나머지 24칸은 그대로, 다시 정규화)."""
    va_is_twin = is_twin_all.loc[va_part.index].to_numpy()
    tr_sub = tr_part[tr_part[TARGET].isin(pair) & ~is_twin_all.loc[tr_part.index]]
    proba_new = oof_main[vai].copy()
    if len(tr_sub) < 10:
        return proba_new
    Xtr, ytr, spec = marker_fn(tr_sub)
    main_pred = main_pred_all[vai]
    route_mask = np.isin(main_pred, list(pair_idx_map.values())) & (~va_is_twin)
    idx = np.flatnonzero(route_mask)
    if len(idx) == 0:
        return proba_new
    Xva = marker_fn(va_part.iloc[idx], transform_only=True)
    spec_proba = spec.predict_proba(Xva)  # [:,0]=클래스0, [:,1]=클래스1 (fit 시 정의된 순서)
    c0, c1 = list(pair_idx_map.values())[0], list(pair_idx_map.values())[1]
    pair_mass = proba_new[idx][:, [c0, c1]].sum(axis=1, keepdims=True)  # 기존 그 두 칸이 차지하던 확률 총량 유지
    proba_new[idx.reshape(-1, 1), [c0, c1]] = spec_proba * pair_mass
    return proba_new


def gbm_marker_fn_factory():
    def fn(df, transform_only=False):
        X = (df[GBM_MARKERS] != "WT").astype(int).to_numpy()
        return X
    return fn


summary = []
for sd in SEEDS:
    oof_main = cache[sd]["oof_main"]
    main_pred_all = oof_main.argmax(1)
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)

    oof_gbm_proba = oof_main.copy()
    oof_kid_proba = oof_main.copy()
    oof_combined_proba = oof_main.copy()

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()
        main_pred = main_pred_all[vai]

        # --- GBMLGG/LGG: 확률 교체 ---
        tr_gbm = tr_part[tr_part[TARGET].isin(PAIR_GBM) & ~is_twin_all.loc[tr_part.index]]
        gbm_slice = oof_main[vai].copy()
        if len(tr_gbm) >= 10:
            Xg_tr = (tr_gbm[GBM_MARKERS] != "WT").astype(int).to_numpy()
            yg_tr = (tr_gbm[TARGET] == "LGG").astype(int).to_numpy()  # 0=GBMLGG,1=LGG
            wg = class_weights(yg_tr)
            spec_g = LogisticRegression(max_iter=1000).fit(Xg_tr, yg_tr, sample_weight=wg)
            route_g = np.isin(main_pred, [gbm_idx["GBMLGG"], gbm_idx["LGG"]]) & (~va_is_twin)
            idxg = np.flatnonzero(route_g)
            if len(idxg):
                Xg_va = (va_part.iloc[idxg][GBM_MARKERS] != "WT").astype(int).to_numpy()
                spec_proba_g = spec_g.predict_proba(Xg_va)  # col0=GBMLGG(0), col1=LGG(1) 순서
                c_gbm, c_lgg = gbm_idx["GBMLGG"], gbm_idx["LGG"]
                pair_mass = gbm_slice[idxg][:, [c_gbm, c_lgg]].sum(axis=1, keepdims=True)
                gbm_slice[np.ix_(idxg, [c_gbm, c_lgg])] = spec_proba_g * pair_mass

        # --- KIRC/KIPAN: 확률 교체 ---
        tr_kid = tr_part[tr_part[TARGET].isin(PAIR_KID) & ~is_twin_all.loc[tr_part.index]]
        kid_slice = oof_main[vai].copy()
        if len(tr_kid) >= 10:
            Xk_full_tr = (tr_kid[genes] != "WT").astype(int).to_numpy()
            yk_tr = (tr_kid[TARGET] == "KIRC").astype(int).to_numpy()  # 0=KIPAN? need consistent order
            ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(Xk_full_tr, yk_tr)
            top_idx = np.argsort(-ranker.feature_importances_)[:K_KID]
            spec_k = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xk_full_tr[:, top_idx], yk_tr)
            route_k = np.isin(main_pred, [kid_idx["KIRC"], kid_idx["KIPAN"]]) & (~va_is_twin)
            idxk = np.flatnonzero(route_k)
            if len(idxk):
                Xk_full_va = (va_part.iloc[idxk][genes] != "WT").astype(int).to_numpy()[:, top_idx]
                spec_proba_k = spec_k.predict_proba(Xk_full_va)  # col idx per spec_k.classes_
                classes_order = list(spec_k.classes_)  # e.g. [0,1] -> 0=KIPAN label? y=1 if KIRC else 0
                col_kipan = classes_order.index(0)
                col_kirc = classes_order.index(1)
                c_kirc, c_kipan = kid_idx["KIRC"], kid_idx["KIPAN"]
                pair_mass = kid_slice[idxk][:, [c_kirc, c_kipan]].sum(axis=1, keepdims=True)
                ordered = np.stack([spec_proba_k[:, col_kirc], spec_proba_k[:, col_kipan]], axis=1)
                kid_slice[np.ix_(idxk, [c_kirc, c_kipan])] = ordered * pair_mass

        combined_slice = oof_main[vai].copy()
        if len(tr_gbm) >= 10 and len(idxg):
            combined_slice[np.ix_(idxg, [gbm_idx["GBMLGG"], gbm_idx["LGG"]])] = gbm_slice[np.ix_(idxg, [gbm_idx["GBMLGG"], gbm_idx["LGG"]])]
        if len(tr_kid) >= 10 and len(idxk):
            combined_slice[np.ix_(idxk, [kid_idx["KIRC"], kid_idx["KIPAN"]])] = kid_slice[np.ix_(idxk, [kid_idx["KIRC"], kid_idx["KIPAN"]])]

        oof_gbm_proba[vai] = gbm_slice
        oof_kid_proba[vai] = kid_slice
        oof_combined_proba[vai] = combined_slice
        print(f"  seed{sd} fold{k} 완료", flush=True)

    # class_scale을 각 버전(라우팅 반영된 확률)에 대해 새로 피팅
    def scaled_f1(oof_p):
        sc = fit_class_scales(oof_p, y_all)
        pred = (oof_p * sc).argmax(1)
        return f1_score(y_all, pred, average="macro")

    main_scaled = scaled_f1(oof_main)
    gbm_scaled = scaled_f1(oof_gbm_proba)
    kid_scaled = scaled_f1(oof_kid_proba)
    combined_scaled = scaled_f1(oof_combined_proba)

    row = dict(seed=sd, main=round(main_scaled, 5), gbm=round(gbm_scaled, 5),
               kid=round(kid_scaled, 5), combined=round(combined_scaled, 5))
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
df["delta_gbm"] = df["gbm"] - df["main"]
df["delta_kid"] = df["kid"] - df["main"]
df["delta_combined"] = df["combined"] - df["main"]
print("\n=== 결과 ===")
print(df.to_string(index=False))
print("\n=== 쌍대비교 ===")
for col in ["delta_gbm", "delta_kid", "delta_combined"]:
    m, s = df[col].mean(), df[col].std(ddof=1)
    print(f"{col}: mean={m:.5f} std={s:.5f} mean/std={m/s:.2f} 양수={int((df[col]>0).sum())}/5")
