"""approach11 v10c: KIRC/KIPAN 서브모델에 GBMLGG/LGG v8과 동일한 정밀도 개선 3가지 적용.
  1) 마커 확장 — literature_map.py KIPAN/KIRC 항목 중 패널에 있는 건 이미 v10에서 전부 포함(VHL/MTOR/TP53/PTEN/MET/NF2/FH/PIK3CA/TSC1/CDKN2A, 10개). 추가로 더 찾을 건 없어 이 단계는 스킵.
  2) 조합 피처 — approach10_v1의 아이디어(VHL 있고 MET 없으면 KIRC 신호, 반대는 KIPAN 신호)를 명시적 피처로 추가
  3) 확률 임계값 스윕(0.4~0.7)
기준(v10): macro F1=0.6116, KIRC재현율=36.6%, KIRC정밀도=38.0%
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
y = (pair[TARGET] == "KIRC").astype(int).to_numpy()

Xm = (pair[MARKERS] != "WT").astype(int)
Xm["combo_vhl_notMET"] = Xm["VHL"] & (1 - Xm["MET"])     # approach10_v1의 KIRC 신호 아이디어 재사용
Xm["combo_met_notVHL"] = Xm["MET"] & (1 - Xm["VHL"])     # approach10_v1의 KIPAN 신호 아이디어 재사용
X = Xm.to_numpy()


def eval_config(threshold=0.5):
    results = []
    for sd in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
        proba = np.zeros(len(y))
        for tri, tei in skf.split(X, y):
            m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tri], y[tri])
            proba[tei] = m.predict_proba(X[tei])[:, list(m.classes_).index(1)]
        pred = (proba >= threshold).astype(int)
        results.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                             recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))
    return (np.mean([r["macro_f1"] for r in results]), np.mean([r["recall"] for r in results]),
            np.mean([r["precision"] for r in results]))


print("=== 마커10개 + 조합2개(VHL&~MET, MET&~VHL), 임계값 스윕 ===")
for th in [0.4, 0.5, 0.6, 0.7]:
    mf1, rec, prec = eval_config(th)
    print(f"임계값{th}: macro F1={mf1:.4f}  KIRC재현율={rec:.1%}  KIRC정밀도={prec:.1%}")

print(f"\n(참고) v10 기준(조합 없음, 임계값0.5): macro F1=0.6116  재현율=36.6%  정밀도=38.0%")
