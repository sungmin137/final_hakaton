"""approach11 v9c: v7 기준(6개 마커, 로지스틱회귀)에서 피처만 "있다/없다" → "변이 심각도 점수"로 교체.
KnowledgeFeatures(공용 코드, approach2_knowledge)의 variant_severity()를 재사용 — 직접 재구현하지 않음.
다른 건 그대로 — 변별력용 단일 변경.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train
from approach2_knowledge.knowledge_features import variant_severity  # 공용 코드 재사용

MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
y = (pair[TARGET] == "LGG").astype(int).to_numpy()

sev = np.zeros((len(pair), len(MARKERS)))
for j, g in enumerate(MARKERS):
    col = pair[g].to_numpy()
    for i, cell in enumerate(col):
        if cell == "WT":
            continue
        s = max(variant_severity(tok)[0] for tok in cell.split(" "))  # 셀에 변이 여러 개면 최댓값
        sev[i, j] = s
X = sev

skf_results = []
for sd in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
    pred = np.zeros(len(y), dtype=int)
    for tri, tei in skf.split(X, y):
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tri], y[tri])
        pred[tei] = m.predict(X[tei])
    skf_results.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                             recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))

mf1 = np.mean([r["macro_f1"] for r in skf_results])
rec = np.mean([r["recall"] for r in skf_results])
prec = np.mean([r["precision"] for r in skf_results])
print(f"변이 심각도 점수(6개 마커): macro F1={mf1:.4f}  LGG재현율={rec:.1%}  LGG정밀도={prec:.1%}")
print(f"(참고) v7 기준(있다/없다 이진): macro F1=0.7224  재현율=76.2%  정밀도=44.4%")
