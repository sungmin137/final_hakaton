"""approach11 v10b: KIRC/KIPAN 서브모델에 GBMLGG/LGG에서 시도했던 3가지 보정(v9a/b/c)을 동일 적용.
v10(문헌 마커 10개, 로지스틱회귀, macro F1 0.6116)을 기준으로 각각 단일 변경만 테스트.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train
from approach2_knowledge.knowledge_features import variant_severity

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
X_base = (pair[MARKERS] != "WT").astype(int).to_numpy()
X_full = (pair[genes] != "WT").astype(int).to_numpy()


def eval_model(X_, make_model):
    out = []
    for sd in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
        pred = np.zeros(len(y), dtype=int)
        for tri, tei in skf.split(X_, y):
            m = make_model()
            m.fit(X_[tri], y[tri])
            pred[tei] = m.predict(X_[tei])
        out.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                         recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))
    return np.mean([r["macro_f1"] for r in out]), np.mean([r["recall"] for r in out]), np.mean([r["precision"] for r in out])


print("=== 기준(v10): 문헌 마커 10개, 로지스틱회귀 ===")
mf1, rec, prec = eval_model(X_base, lambda: LogisticRegression(class_weight="balanced", max_iter=1000))
print(f"macro F1={mf1:.4f}  KIRC재현율={rec:.1%}  KIRC정밀도={prec:.1%}\n")

print("=== 보정1: 비선형 모델 ===")
for name, factory in [("SVM(RBF)", lambda: SVC(kernel="rbf", class_weight="balanced")),
                       ("결정나무(depth=3)", lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0))]:
    mf1, rec, prec = eval_model(X_base, factory)
    print(f"{name}: macro F1={mf1:.4f}  재현율={rec:.1%}  정밀도={prec:.1%}")

print("\n=== 보정2: 카이제곱 자동 top-10 (전체 4,384유전자에서) ===")
K = 10
results = []
for sd in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
    pred = np.zeros(len(y), dtype=int)
    for tri, tei in skf.split(X_full, y):
        sel = SelectKBest(chi2, k=K).fit(X_full[tri], y[tri])
        idx = sel.get_support(indices=True)
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_full[tri][:, idx], y[tri])
        pred[tei] = m.predict(X_full[tei][:, idx])
    results.append(dict(macro_f1=f1_score(y, pred, average="macro"), recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))
print(f"macro F1={np.mean([r['macro_f1'] for r in results]):.4f}  재현율={np.mean([r['recall'] for r in results]):.1%}  정밀도={np.mean([r['precision'] for r in results]):.1%}")

print("\n=== 보정3: 변이 심각도 점수 ===")
sev = np.zeros((len(pair), len(MARKERS)))
for j, g in enumerate(MARKERS):
    col = pair[g].to_numpy()
    for i, cell in enumerate(col):
        if cell == "WT":
            continue
        sev[i, j] = max(variant_severity(tok)[0] for tok in cell.split(" "))
mf1, rec, prec = eval_model(sev, lambda: LogisticRegression(class_weight="balanced", max_iter=1000))
print(f"macro F1={mf1:.4f}  재현율={rec:.1%}  정밀도={prec:.1%}")
