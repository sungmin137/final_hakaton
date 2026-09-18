"""approach11 v8: v7 서브모델 정밀도 개선안 3가지 — 서브모델만으로 검증(메인 모델 없음, 빠름).
  1) 마커 4개 추가 (NF1, PIK3CA, RB1, NOTCH1 — literature_map.py 문헌 변이율 근거)
  2) 조합 피처 (IDH1&ATRX = 성상세포종 아형, known_gene_roles.py 근거)
  3) 확률 임계값 스윕 (0.4~0.7) — LGG 정밀도 개선 목적
평가는 5-fold(+시드 5개) — v7b에서 LOOCV와 동등함을 이미 확인함.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from main import TARGET, load_train
from features.features import InsightFeatures  # 공용 COMBOS(IDH1+ATRX 포함)를 재사용, 직접 재구현하지 않음

BASE = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]
EXTRA = ["NF1", "PIK3CA", "RB1", "NOTCH1"]
MARKERS = BASE + EXTRA
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)

PAIR = ["GBMLGG", "LGG"]
pair = train[train[TARGET].isin(PAIR) & ~is_twin]
y = (pair[TARGET] == "LGG").astype(int).to_numpy()

Xm = (pair[MARKERS] != "WT").astype(int)
ins = InsightFeatures().fit(pair)                         # COMBOS 안에 ("IDH1","ATRX")가 이미 있음(features.py)
Xm["combo_IDH1_ATRX"] = ins.transform(pair)["combo_IDH1_ATRX"].to_numpy()  # 성상세포종 아형 조합, 공용 코드 재사용
Xm["combo_idh1_notEGFR"] = Xm["IDH1"] & (1 - Xm["EGFR"])   # 여긴 COMBOS에 없는 신규 조합이라 직접 계산
X = Xm.to_numpy()
feat_names = list(Xm.columns)


def eval_config(X_, y_, threshold=0.5):
    results = []
    for sd in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
        proba = np.zeros(len(y_))
        for tri, tei in skf.split(X_, y_):
            m = LogisticRegression(class_weight="balanced", max_iter=1000)
            m.fit(X_[tri], y_[tri])
            proba[tei] = m.predict_proba(X_[tei])[:, list(m.classes_).index(1)]
        pred = (proba >= threshold).astype(int)
        results.append(dict(
            macro_f1=f1_score(y_, pred, average="macro"),
            lgg_recall=recall_score(y_, pred),
            lgg_precision=precision_score(y_, pred, zero_division=0),
        ))
    return results


print(f"=== 마커 10개(기존6+신규4) + 조합2개, 임계값 스윕 ===")
for th in [0.4, 0.5, 0.6, 0.7]:
    res = eval_config(X, y, th)
    mf1 = np.mean([r["macro_f1"] for r in res])
    rec = np.mean([r["lgg_recall"] for r in res])
    prec = np.mean([r["lgg_precision"] for r in res])
    print(f"임계값{th}: macro F1={mf1:.4f}  LGG재현율={rec:.1%}  LGG정밀도={prec:.1%}")

print(f"\n=== 비교: 기존 6개 마커만, 임계값 0.5 (v7 기준) ===")
X6 = (pair[BASE] != "WT").astype(int).to_numpy()
res6 = eval_config(X6, y, 0.5)
print(f"macro F1={np.mean([r['macro_f1'] for r in res6]):.4f}  "
      f"LGG재현율={np.mean([r['lgg_recall'] for r in res6]):.1%}  "
      f"LGG정밀도={np.mean([r['lgg_precision'] for r in res6]):.1%}")
