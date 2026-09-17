"""approach11 v9b: v7 기준(6개 마커, 로지스틱회귀)에서 마커 선정 방식만 교체 — 문헌 수작업 대신
전체 4,384개 유전자에서 카이제곱으로 top-K 자동 선정 (성민님의 approach6 feature_select_cv.py와 같은 원리,
다만 그건 26클래스 메인 모델 4,230→600/1,500이었고 여기는 GBMLGG/LGG 이진분류 서브모델용으로 새로 적용).
다른 건 그대로(로지스틱회귀, 5-fold+5시드, 임계값 0.5) — 변별력용 단일 변경.
"""
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train

SEEDS = [42, 7, 123, 2024, 999]
K = 6  # v7과 같은 개수로 맞춰서 공정 비교

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
y = (pair[TARGET] == "LGG").astype(int).to_numpy()
X_full = (pair[genes] != "WT").astype(int).to_numpy()  # 전체 4,384개 유전자

results = []
chosen_sets = []
for sd in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
    pred = np.zeros(len(y), dtype=int)
    fold_chosen = []
    for tri, tei in skf.split(X_full, y):
        selector = SelectKBest(chi2, k=K).fit(X_full[tri], y[tri])   # train쪽에서만 top-K 선정(누수 방지)
        top_idx = selector.get_support(indices=True)
        fold_chosen.append(set(np.array(genes)[top_idx]))
        Xtr, Xte = X_full[tri][:, top_idx], X_full[tei][:, top_idx]
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, y[tri])
        pred[tei] = m.predict(Xte)
    results.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                         recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))
    chosen_sets.append(fold_chosen)

mf1 = np.mean([r["macro_f1"] for r in results])
rec = np.mean([r["recall"] for r in results])
prec = np.mean([r["precision"] for r in results])
print(f"카이제곱 자동 선정 top-{K}: macro F1={mf1:.4f}  LGG재현율={rec:.1%}  LGG정밀도={prec:.1%}")
print(f"(참고) v7 기준(문헌 마커 6개, 로지스틱회귀): macro F1=0.7224  재현율=76.2%  정밀도=44.4%")

from collections import Counter
all_chosen = Counter(g for seed_folds in chosen_sets for fold in seed_folds for g in fold)
print(f"\n자주 뽑힌 유전자 top-15: {all_chosen.most_common(15)}")
