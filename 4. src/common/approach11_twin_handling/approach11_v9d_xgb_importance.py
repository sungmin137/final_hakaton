"""approach11 v9d: GBMLGG/LGG 서브모델 마커 선정을 approach8과 같은 방식(XGBoost feature_importances_)으로 시도.
fold 내부(학습 데이터)에서만 랭킹용 XGBoost를 학습 — test/검증 누수 없음.
"""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from main import TARGET, XGB_PARAMS, load_train

K = 6  # v7과 같은 개수로 공정 비교
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
y = (pair[TARGET] == "LGG").astype(int).to_numpy()
X_full = (pair[genes] != "WT").astype(int).to_numpy()

results = []
chosen_all = []
for sd in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
    pred = np.zeros(len(y), dtype=int)
    fold_chosen = []
    for tri, tei in skf.split(X_full, y):
        ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(X_full[tri], y[tri])   # approach8과 동일 방식
        importances = ranker.feature_importances_
        top_idx = np.argsort(-importances)[:K]
        fold_chosen.append(set(np.array(genes)[top_idx]))
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_full[tri][:, top_idx], y[tri])
        pred[tei] = m.predict(X_full[tei][:, top_idx])
    results.append(dict(macro_f1=f1_score(y, pred, average="macro"),
                         recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0)))
    chosen_all.append(fold_chosen)

mf1 = np.mean([r["macro_f1"] for r in results])
rec = np.mean([r["recall"] for r in results])
prec = np.mean([r["precision"] for r in results])
print(f"XGBoost 중요도 기반 top-{K}: macro F1={mf1:.4f}  LGG재현율={rec:.1%}  LGG정밀도={prec:.1%}")
print(f"(참고) 문헌 마커(v7): macro F1=0.7224  재현율=76.2%  정밀도=44.4%")
print(f"(참고) 카이제곱(v9b): macro F1=0.7064")

from collections import Counter
all_chosen = Counter(g for seed_folds in chosen_all for fold in seed_folds for g in fold)
print(f"\n자주 뽑힌 유전자 top-15: {all_chosen.most_common(15)}")
