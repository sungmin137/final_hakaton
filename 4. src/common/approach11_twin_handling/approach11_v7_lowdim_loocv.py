"""approach11 v7: GBMLGG/LGG 전용 서브모델 — 마지막 시도.
회의(2026-09-11~12)에서 나온 제안 1+4를 결합: 피처를 문헌 근거 있는 소수 유전자로 줄이고(차원의 저주 해소),
LOOCV로 338개(비쌍둥이) 표본을 최대한 활용해 평가한다. train만 사용(test 안 읽음).

approach6(4,230→600/1,500, 메인 26클래스)의 "차원의 저주가 병목이 아니다"는 결론은
표본(6,201) >> 피처(4,230)인 상황에서 나온 것이라, 표본(52~338) << 피처(4,384)인
이 서브모델 상황과는 표본/피처 비율이 정반대라 적용되지 않는다(회의에서 확인).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut

from main import TARGET, load_train

# 문헌 근거 있는 마커 (approach9_gene_override/known_gene_roles.py, approach10 참고)
MARKERS = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
print(f"비쌍둥이 표본: {len(pair)} (GBMLGG {sum(pair[TARGET]=='GBMLGG')}, LGG {sum(pair[TARGET]=='LGG')})")

present = [g for g in MARKERS if g in pair.columns]
missing = [g for g in MARKERS if g not in pair.columns]
print(f"패널에 있는 마커: {present}")
if missing:
    print(f"패널에 없는 마커: {missing}")

X = (pair[present] != "WT").astype(int).to_numpy()
y = (pair[TARGET] == "LGG").astype(int).to_numpy()   # 0=GBMLGG, 1=LGG

loo = LeaveOneOut()
pred = np.zeros(len(y), dtype=int)
for tri, tei in loo.split(X):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X[tri], y[tri])
    pred[tei] = model.predict(X[tei])

macro_f1 = f1_score(y, pred, average="macro")
acc = accuracy_score(y, pred)
cm = confusion_matrix(y, pred)
print(f"\nLOOCV 결과 (n={len(y)}, 피처={len(present)}개: {present})")
print(f"이진 분류 macro F1: {macro_f1:.4f} / 정확도: {acc:.4f}")
print(f"혼동행렬 (행=진짜, 열=예측, 0=GBMLGG 1=LGG):\n{cm}")

gbmlgg_recall = cm[0, 0] / cm[0].sum()
lgg_recall = cm[1, 1] / cm[1].sum()
print(f"\nGBMLGG 재현율: {gbmlgg_recall:.1%} (전체 XGBoost 서브모델 기존 기록: 92%)")
print(f"LGG 재현율: {lgg_recall:.1%} (전체 XGBoost 서브모델 기존 기록: 32.7%)")

baseline_acc = max(y.mean(), 1 - y.mean())
print(f"\n(참고) 항상 다수 클래스(GBMLGG) 예측 시 정확도: {baseline_acc:.1%}")
