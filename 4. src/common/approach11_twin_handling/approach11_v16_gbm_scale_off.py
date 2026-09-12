"""approach11 v16: class_scale은 정상 적용하되, GBMLGG/LGG 두 칸만 배율을 1.0(끔)으로 고정하고
나머지 24개 클래스는 그대로 둔 상태에서 라우팅(하드 덮어쓰기, v14와 동일 방식)을 결합.
class_scale과 라우팅이 "같은 두 클래스를 동시에 건드려서" 충돌한다는 가설이 맞다면,
GBMLGG/LGG 배율만 꺼주면 라우팅의 순수 효과가 다시 양수로 나와야 함.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, TARGET, twin_groups, load_train, class_weights
from postprocess.class_scale import fit_class_scales

GBM_MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
PAIR_GBM = ["GBMLGG", "LGG"]
SEEDS = [42, 7, 123, 2024, 999]

cache_path = ROOT / "6. experiments" / "2026-09-12_approach11_v11_integration" / "oof_main_cache.pkl"
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

train = load_train()
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
gbm_idx = {c: le.transform([c])[0] for c in PAIR_GBM}

genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets_all = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets_all).apply(lambda s: len(s) > 1)

summary = []
for sd in SEEDS:
    oof_main = cache[sd]["oof_main"]
    scales = fit_class_scales(oof_main, y_all)
    scales[gbm_idx["GBMLGG"]] = 1.0    # ★ GBMLGG 배율 끔
    scales[gbm_idx["LGG"]] = 1.0       # ★ LGG 배율 끔 (나머지 24개는 그대로)
    scaled_proba = oof_main * scales
    main_pred_all = scaled_proba.argmax(1)

    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
    oof_gbm_only = np.full(len(train), -1)

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]
        main_pred = main_pred_all[vai]
        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()

        tr_gbm = tr_part[tr_part[TARGET].isin(PAIR_GBM) & ~is_twin_all.loc[tr_part.index]]
        gbm_pred_route = main_pred.copy()
        if len(tr_gbm) >= 10:
            Xg_tr = (tr_gbm[GBM_MARKERS] != "WT").astype(int).to_numpy()
            yg_tr = (tr_gbm[TARGET] == "LGG").astype(int).to_numpy()
            wg = class_weights(yg_tr)
            spec_g = LogisticRegression(max_iter=1000).fit(Xg_tr, yg_tr, sample_weight=wg)
            route_g = np.isin(main_pred, [gbm_idx["GBMLGG"], gbm_idx["LGG"]]) & (~va_is_twin)
            idxg = np.flatnonzero(route_g)
            if len(idxg):
                Xg_va = (va_part.iloc[idxg][GBM_MARKERS] != "WT").astype(int).to_numpy()
                pred_g = spec_g.predict(Xg_va)
                gbm_pred_route[idxg] = np.where(pred_g == 1, gbm_idx["LGG"], gbm_idx["GBMLGG"])
        oof_gbm_only[vai] = gbm_pred_route
        print(f"  seed{sd} fold{k} 완료", flush=True)

    main_f1 = f1_score(y_all, main_pred_all, average="macro")
    gbm_f1 = f1_score(y_all, oof_gbm_only, average="macro")
    row = dict(seed=sd, main=round(main_f1, 5), gbm=round(gbm_f1, 5), delta=round(gbm_f1 - main_f1, 5))
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
print("\n=== 결과 ===")
print(df.to_string(index=False))
m, s = df["delta"].mean(), df["delta"].std(ddof=1)
print(f"\ndelta: mean={m:.5f} std={s:.5f} mean/std={m/s:.2f} 양수={int((df['delta']>0).sum())}/5")
