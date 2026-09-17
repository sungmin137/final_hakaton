"""approach11 v11: GBMLGG/LGG 저차원 서브모델(v7)을 26클래스 메인 파이프라인에 실제로 통합.
같은 실행에서 "라우팅만" vs "라우팅+twin_rule" 두 버전을 동시에 계산(메인 모델은 1번만 학습).
메인 모델 확률을 fold별로 저장해서, 나중에 서브모델/twin_rule 조합을 바꿔 재검증할 때 재학습 없이 재사용 가능.

메인 모델: v7과 동일(features v4, mild_col 파라미터, colsample_bytree 0.7)
서브모델: v7과 동일(IDH1/EGFR/ATRX/PTEN/IDH2/TP53, 로지스틱회귀, 임계값 0.5)
라우팅: 메인 예측이 GBMLGG/LGG이고 쌍둥이가 아닌 행만 서브모델로 덮어씀(보정 a+d)
twin_rule 시뮬레이션: 검증 행이 train(그 fold의 학습 fold)과 프로필이 완전히 같으면 FLIP 라벨 적용
"""
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, PARAM_SETS, TARGET, twin_groups, load_train, class_weights, FeatureMaker

MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
PAIR = ["GBMLGG", "LGG"]
FLIP = {"KIPAN": "KIRC", "KIRC": "KIPAN", "GBMLGG": "LGG", "LGG": "GBMLGG"}
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
pair_idx = {c: le.transform([c])[0] for c in PAIR}
params = PARAM_SETS["mild_col"]

genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets_all = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets_all).apply(lambda s: len(s) > 1)
n_mut_all = (train[genes] != "WT").sum(axis=1)

cache = {}  # seed -> {"oof_main": ndarray, "folds": [...]}
summary = []

for sd in SEEDS:
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)

    oof_main = np.zeros((len(train), len(le.classes_)))
    oof_routing = np.full(len(train), -1)
    oof_routing_twin = np.full(len(train), -1)

    for k, (tri, vai) in enumerate(skf.split(train, y_all, groups)):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]

        # 메인 모델 (1번만 학습)
        fm = FeatureMaker("v4").fit(tr_part)
        Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
        main_model = xgb.XGBClassifier(**params).fit(Xtr, y_all[tri])
        proba_va = main_model.predict_proba(Xva)
        oof_main[vai] = proba_va
        main_pred = proba_va.argmax(1)

        # 서브모델: 이 fold 학습 데이터의 비쌍둥이 GBMLGG/LGG
        tr_pair_mask = tr_part[TARGET].isin(PAIR) & ~is_twin_all.loc[tr_part.index]
        tr_pair = tr_part[tr_pair_mask]
        Xtr_spec = (tr_pair[MARKERS] != "WT").astype(int).to_numpy()
        y_spec = (tr_pair[TARGET] == "LGG").astype(int).to_numpy()
        w = class_weights(y_spec)
        spec = LogisticRegression(class_weight=None, max_iter=1000).fit(Xtr_spec, y_spec, sample_weight=w)

        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()
        route_mask = np.isin(main_pred, [pair_idx["GBMLGG"], pair_idx["LGG"]]) & (~va_is_twin)
        Xva_spec = (va_part[MARKERS] != "WT").astype(int).to_numpy()
        spec_pred = spec.predict(Xva_spec) if len(va_part) else np.array([])

        # 후처리 A: 라우팅만
        pred_routing = main_pred.copy()
        idx = np.flatnonzero(route_mask)
        pred_routing[idx] = np.where(spec_pred[idx] == 1, pair_idx["LGG"], pair_idx["GBMLGG"])
        oof_routing[vai] = pred_routing

        # 후처리 B: 라우팅 + twin_rule 시뮬레이션 (검증 행이 이 fold의 학습셋과 프로필 일치하면 FLIP)
        pred_routing_twin = pred_routing.copy()
        tr_lookup = {}
        tr_key = tr_part[genes].astype(str).agg("|".join, axis=1)
        for key, lab, nm in zip(tr_key, tr_part[TARGET], n_mut_all.loc[tr_part.index]):
            if nm > 0:
                tr_lookup.setdefault(key, []).append(lab)
        va_key = va_part[genes].astype(str).agg("|".join, axis=1)
        for i, key in enumerate(va_key):
            labs = tr_lookup.get(key)
            if labs:
                twin_lab = max(set(labs), key=labs.count)
                flipped = FLIP.get(twin_lab, twin_lab)
                pred_routing_twin[i] = le.transform([flipped])[0]
        oof_routing_twin[vai] = pred_routing_twin

        print(f"  seed{sd} fold{k} 완료", flush=True)

    macro_main = f1_score(y_all, oof_main.argmax(1), average="macro")
    macro_routing = f1_score(y_all, oof_routing, average="macro")
    macro_routing_twin = f1_score(y_all, oof_routing_twin, average="macro")
    row = dict(seed=sd, main=round(macro_main, 4), routing=round(macro_routing, 4), routing_twin=round(macro_routing_twin, 4))
    print(row, flush=True)
    summary.append(row)
    cache[sd] = dict(oof_main=oof_main)

df = pd.DataFrame(summary)
print("\n=== 시드별 결과 ===")
print(df.to_string(index=False))
print("\n=== 평균 / 표준편차 ===")
for col in ["main", "routing", "routing_twin"]:
    print(f"{col}: mean={df[col].mean():.4f} std={df[col].std():.4f}")

out_dir = ROOT / "6. experiments" / "2026-09-12_approach11_v11_integration"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "oof_main_cache.pkl", "wb") as f:
    pickle.dump(cache, f)
df.to_csv(out_dir / "summary.csv", index=False)
print(f"\nsaved: {out_dir}")
