"""approach11 v10d: KIRC/KIPAN v10 기준(10개 마커, 로지스틱회귀)을 LOOCV로 재검증.
GBMLGG/LGG(v7b)에서는 LOOCV와 5-fold가 거의 동일했지만, KIRC(58개)가 LGG(52개)보다도 소수클래스 불균형이
심해서(약 20:80 vs 85:15) 5-fold의 작은 검증셋(폴드당 12개 안팎)이 더 불안정할 수 있어 재확인.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut

from main import TARGET, load_train

MARKERS = ["VHL", "MTOR", "TP53", "PTEN", "MET", "NF2", "FH", "PIK3CA", "TSC1", "CDKN2A"]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["KIRC", "KIPAN"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
print(f"비쌍둥이 표본: {len(pair)} (KIRC {sum(pair[TARGET]=='KIRC')}, KIPAN {sum(pair[TARGET]=='KIPAN')})")

X = (pair[MARKERS] != "WT").astype(int).to_numpy()
y = (pair[TARGET] == "KIRC").astype(int).to_numpy()

loo = LeaveOneOut()
pred = np.zeros(len(y), dtype=int)
for tri, tei in loo.split(X):
    m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tri], y[tri])
    pred[tei] = m.predict(X[tei])

macro_f1 = f1_score(y, pred, average="macro")
rec = recall_score(y, pred)
prec = precision_score(y, pred, zero_division=0)
cm = confusion_matrix(y, pred)
print(f"\nLOOCV(n={len(y)}) 결과: macro F1={macro_f1:.4f}  KIRC재현율={rec:.1%}  KIRC정밀도={prec:.1%}")
print(f"혼동행렬(0=KIPAN,1=KIRC):\n{cm}")
print(f"\n(참고) 5-fold(5시드) 기준: macro F1=0.6116  재현율=36.6%  정밀도=38.0%")
