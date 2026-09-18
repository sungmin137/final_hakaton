"""치환 스펙트럼 NB 점수표 피처 (2026-09-15, train 전용). 혜림 접근16의 MultinomialNB(383개 횟수: 방향 있는 치환 380 + 절단·동의·기타)를
파트너 모델이 아니라 **v4s 모델 안의 피처(클래스별 로그확률 26열 + max·margin)**로 넣는다(접근 14 레시피: 새 정보는 클래스별 집약 점수 형태로).
학습 행은 내부 5-Fold OOF(자기 라벨 미사용), 쌍둥이는 같은 fold. test는 train 전체 fit 후 transform만."""
import re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
AA = "ACDEFGHIKLMNPQRSTVWY"; _MIS = re.compile(r"^([A-Z])(\d+)([A-Z])$")
PAIRS = {f"{a}{b}": i for i, (a, b) in enumerate((a, b) for a in AA for b in AA if a != b)}

def spectrum_counts(G):
    X = np.zeros((len(G), len(PAIRS) + 3))
    for i in range(len(G)):
        for j in np.nonzero(G[i] != "WT")[0]:
            for t in G[i, j].split(" "):
                m = _MIS.match(t)
                if m and m.group(1) != m.group(3) and m.group(1) + m.group(3) in PAIRS: X[i, PAIRS[m.group(1) + m.group(3)]] += 1
                elif m: X[i, -2] += 1
                elif t.endswith("*") or "fs" in t: X[i, -3] += 1
                else: X[i, -1] += 1
    return X

class SpectrumNBFeatures:
    def __init__(self, alpha=0.5, n_inner=5, seed=42, normalized=False, prefix="snb"):
        self.alpha, self.n_inner, self.seed, self.normalized, self.prefix = alpha, n_inner, seed, normalized, prefix   # normalized: 토큰 수로 나눈 평균 로그우도비(유계) — 27차 교훈
    def _score(self, m, X):
        if not self.normalized: return m.predict_proba(X)
        W = m.feature_log_prob_ - m.feature_log_prob_.mean(0, keepdims=True)       # 클래스별 로그우도 - 클래스 평균 (비율 형태)
        n = np.maximum(X.sum(1, keepdims=True), 1); S = (X @ W.T) / n; S[X.sum(1) == 0] = 0.0
        full = np.zeros((len(X), len(self.classes))); full[:, m.classes_] = S; return np.round(full, 4)
    def _cols(self, P):
        V = P if self.normalized else np.log(P + 1e-9)
        P_ = self.prefix; out = pd.DataFrame(np.round(V, 4), columns=[f"{P_}_{c}" for c in self.classes]); out[f"{P_}_max"] = out.max(1); s = np.sort(out.to_numpy()[:, :len(self.classes)], 1); out[f"{P_}_margin"] = s[:, -1] - s[:, -2]; return out
    def fit(self, train):
        from main import twin_groups
        self.genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]; X = spectrum_counts(train[self.genes].to_numpy())
        le = LabelEncoder().fit(train["SUBCLASS"]); y = le.transform(train["SUBCLASS"]); self.classes = list(le.classes_)
        self.model = MultinomialNB(alpha=self.alpha).fit(X, y)
        oof = np.zeros((len(train), len(self.classes)))
        for tri, vai in StratifiedGroupKFold(self.n_inner, shuffle=True, random_state=self.seed).split(X, y, twin_groups(train)):
            m = MultinomialNB(alpha=self.alpha).fit(X[tri], y[tri])
            if self.normalized: oof[vai] = self._score(m, X[vai])
            else: p = m.predict_proba(X[vai]); full = np.zeros((len(vai), len(self.classes))); full[:, m.classes_] = p; oof[vai] = full
        self._train_index = train.index; self._train_oof = self._cols(oof).set_index(train.index); return self
    def transform(self, df):
        if df.index.equals(self._train_index): return self._train_oof
        X = spectrum_counts(df[self.genes].to_numpy()); return self._cols(self._score(self.model, X) if self.normalized else self.model.predict_proba(X)).set_index(df.index)
