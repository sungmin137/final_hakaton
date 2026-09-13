"""접근 14 v8: 쌍 재판기를 argmax 뒤 라우팅이 아니라 **확률 단계**에서 결합.
블렌드 확률 p에서 쌍 (A,B)의 합 m=p_A+p_B는 그대로 두고, 쌍 안의 비율만 재판기 q와 기하 혼합:
  p_A' : p_B' = (p_A/p_B)^(1-v) · (q_A/q_B)^v
재판기: 혜림 approach19 HNSC/STES(빈도차 상위 40 유전자 LR C=0.1) · 혜성 approach11 v7 GBMLGG/LGG(문헌 마커 6개 LR).
정직 CV에서는 fold의 학습 부분만으로 재판기를 fit 한다. test 통계는 쓰지 않는다.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from main import ID, TARGET, SEED

MARKERS_GBM_LGG = ["IDH1", "EGFR", "ATRX", "PTEN", "IDH2", "TP53"]   # approach11 v7 (혜성)
PAIRS = {"HNSC_STES": ("HNSC", "STES"), "GBMLGG_LGG": ("GBMLGG", "LGG")}


def _binary(df, genes): return (df[genes].to_numpy() != "WT").astype(np.int8)


class PairReferee:
    """fit(train_part) → proba(df) = P(A) (쌍의 첫 클래스 확률)."""
    def __init__(self, pair, mode, C=0.1, top_k=40):
        self.pair, self.mode, self.C, self.top_k = pair, mode, C, top_k
    def fit(self, df):
        genes = [c for c in df.columns if c not in (ID, TARGET)]
        a, b = self.pair; m = df[TARGET].isin(self.pair).to_numpy(); d = df[m]; y = (d[TARGET].to_numpy() == a).astype(int)
        if self.mode == "top40":
            X = _binary(d, genes); self.cols = np.argsort(np.abs(X[y == 1].mean(0) - X[y == 0].mean(0)))[-self.top_k:]; self.genes = genes
        else:
            self.genes = [g for g in MARKERS_GBM_LGG if g in genes]; self.cols = np.arange(len(self.genes)); X = _binary(d, self.genes)
        self.lr = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED).fit(X[:, self.cols], y)
        return self
    def proba(self, df):
        X = _binary(df, self.genes if self.mode != "top40" else self.genes)[:, self.cols]
        return self.lr.predict_proba(X)[:, 1]


def mix_pair(p, ia, ib, qa, v, eps=1e-9):
    """p: (n,K) 확률, qa: (n,) 재판기 P(A). 쌍 합 유지, 비율만 기하 혼합."""
    p = p.copy(); m = p[:, ia] + p[:, ib]
    la = (1 - v) * np.log(p[:, ia] + eps) + v * np.log(qa + eps); lb = (1 - v) * np.log(p[:, ib] + eps) + v * np.log(1 - qa + eps)
    z = np.exp(la - np.maximum(la, lb)); zb = np.exp(lb - np.maximum(la, lb)); p[:, ia] = m * z / (z + zb); p[:, ib] = m * zb / (z + zb)
    return p
