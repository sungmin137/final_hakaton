"""approach11 v10f: KIRC/KIPAN의 카이제곱·XGBoost중요도 자동선정 방식을 LOOCV로 재확인.
v10d에서 문헌 마커가 5-fold(0.6116)보다 LOOCV(0.5872)가 더 낮게 나온 걸 확인했으므로,
다른 두 자동 방식도 5-fold 숫자만 믿지 않고 LOOCV로 재검증한다.
"""
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut

from main import TARGET, XGB_PARAMS, load_train

K = 10
train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["KIRC", "KIPAN"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
y = (pair[TARGET] == "KIRC").astype(int).to_numpy()
X_full = (pair[genes] != "WT").astype(int).to_numpy()

loo = LeaveOneOut()

print("=== 카이제곱 top-10, LOOCV ===")
pred = np.zeros(len(y), dtype=int)
for tri, tei in loo.split(X_full):
    sel = SelectKBest(chi2, k=K).fit(X_full[tri], y[tri])
    idx = sel.get_support(indices=True)
    m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_full[tri][:, idx], y[tri])
    pred[tei] = m.predict(X_full[tei][:, idx])
print(f"macro F1={f1_score(y, pred, average='macro'):.4f}  재현율={recall_score(y, pred):.1%}  정밀도={precision_score(y, pred, zero_division=0):.1%}")

print("\n=== XGBoost 중요도 top-10, LOOCV ===")
pred2 = np.zeros(len(y), dtype=int)
for tri, tei in loo.split(X_full):
    ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(X_full[tri], y[tri])
    idx = np.argsort(-ranker.feature_importances_)[:K]
    m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_full[tri][:, idx], y[tri])
    pred2[tei] = m.predict(X_full[tei][:, idx])
print(f"macro F1={f1_score(y, pred2, average='macro'):.4f}  재현율={recall_score(y, pred2):.1%}  정밀도={precision_score(y, pred2, zero_division=0):.1%}")

print(f"\n(참고) 문헌 마커 LOOCV(v10d): macro F1=0.5872")
print(f"(참고) 카이제곱 5-fold(v10b): macro F1=0.6260 / XGBoost중요도 5-fold(v10e): macro F1=0.6170")
