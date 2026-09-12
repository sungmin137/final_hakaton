"""approach11 v9a: v7 기준(6개 마커, 로지스틱회귀)에서 모델만 비선형으로 교체 — 다른 건 그대로(변별력용 단일 변경).
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train

MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
X = (pair[MARKERS] != "WT").astype(int).to_numpy()
y = (pair[TARGET] == "LGG").astype(int).to_numpy()


def eval_model(make_model):
    out = []
    for sd in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
        pred = np.zeros(len(y), dtype=int)
        for tri, tei in skf.split(X, y):
            m = make_model()
            m.fit(X[tri], y[tri])
            pred[tei] = m.predict(X[tei])
        out.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                         recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))
    return out


for name, factory in [
    ("SVM(RBF, class_weight=balanced)", lambda: SVC(kernel="rbf", class_weight="balanced")),
    ("결정나무(depth=3, class_weight=balanced)", lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)),
]:
    res = eval_model(factory)
    mf1 = np.mean([r["macro_f1"] for r in res])
    rec = np.mean([r["recall"] for r in res])
    prec = np.mean([r["precision"] for r in res])
    print(f"{name}: macro F1={mf1:.4f}  LGG재현율={rec:.1%}  LGG정밀도={prec:.1%}")

print("\n(참고) v7 기준: 로지스틱회귀 macro F1=0.7224, 재현율=76.2%, 정밀도=44.4%")
