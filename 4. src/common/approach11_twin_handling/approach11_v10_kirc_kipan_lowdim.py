"""approach11 v10: KIRC/KIPAN 전용 서브모델 — v7과 같은 방식(저차원 마커 + 로지스틱회귀)을 신장암에 적용.
문헌 마커(literature_map.py KIPAN/KIRC 변이율 참고)로 패널에 있는 것만 사용: VHL/MTOR/TP53/PTEN/MET/NF2/FH/PIK3CA/TSC1/CDKN2A.
핵심 driver(PBRM1/SETD2/BAP1)는 패널에 없어 제외.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train

MARKERS = ["VHL", "MTOR", "TP53", "PTEN", "MET", "NF2", "FH", "PIK3CA", "TSC1", "CDKN2A"]
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["KIRC", "KIPAN"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
print(f"비쌍둥이 표본: {len(pair)} (KIRC {sum(pair[TARGET]=='KIRC')}, KIPAN {sum(pair[TARGET]=='KIPAN')})")

X = (pair[MARKERS] != "WT").astype(int).to_numpy()
y = (pair[TARGET] == "KIRC").astype(int).to_numpy()   # 0=KIPAN, 1=KIRC

results = []
for sd in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
    pred = np.zeros(len(y), dtype=int)
    for tri, tei in skf.split(X, y):
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tri], y[tri])
        pred[tei] = m.predict(X[tei])
    results.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                         kirc_recall=recall_score(y, pred), kirc_precision=precision_score(y, pred, zero_division=0)))

mf1 = np.mean([r["macro_f1"] for r in results])
rec = np.mean([r["kirc_recall"] for r in results])
prec = np.mean([r["kirc_precision"] for r in results])
print(f"\n5-fold(5시드) 결과: macro F1={mf1:.4f}  KIRC재현율={rec:.1%}  KIRC정밀도={prec:.1%}")
print("(참고) 기존 XGBoost 4384차원 서브모델: KIRC 비쌍둥이 정확도 17~21%")

baseline = max(y.mean(), 1 - y.mean())
print(f"(참고) 항상 다수클래스(KIPAN) 예측 시 정확도: {baseline:.1%}")
