"""approach11 v7b: v7의 원인 분리용. 차원 축소(6개 유전자)는 동일하게 유지하고,
평가 방식만 LOOCV 대신 5-fold로 바꿔서, "좋아진 게 차원 축소 때문인지 LOOCV 때문인지" 구분한다.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train

MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
X = (pair[MARKERS] != "WT").astype(int).to_numpy()
y = (pair[TARGET] == "LGG").astype(int).to_numpy()

for seed in [42, 7, 123, 2024, 999]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for tri, tei in skf.split(X, y):
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X[tri], y[tri])
        pred[tei] = model.predict(X[tei])
    macro_f1 = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred)
    gbmlgg_recall = cm[0, 0] / cm[0].sum()
    lgg_recall = cm[1, 1] / cm[1].sum()
    print(f"seed={seed}: macro F1={macro_f1:.4f}  GBMLGG재현율={gbmlgg_recall:.1%}  LGG재현율={lgg_recall:.1%}  acc={accuracy_score(y,pred):.1%}")
