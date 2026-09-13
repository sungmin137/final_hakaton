"""접근 14 — 접근 1 점수(cw_) 고도화. 메인 모델 이득의 77%를 만드는 클래스별 집약 점수를 세 방향으로 확장한다.
 (i)  유형 분리: missense / 절단(LoF) / 동의 토큰을 따로 세어 유전자·변이 단위 NB 점수 (같은 유전자라도 '어떻게' 바뀌었는지 구분)
 (ii) 위치 구간: (유전자, 아미노산 위치//25) 단위 NB 점수 — hotspot one-hot이 너무 희소해 트리가 못 쓴 정보를 점수로 집약
 (iii) 군집 상대 점수: 저변이·무표지 군집(SARC·PRAD·PCPG·THYM·OV)과 상피암 군집(STES·LUSC·HNSC·LUAD·BLCA) 안에서만 정규화한 점수
학습 행은 내부 5-Fold OOF(자기 라벨 미사용), 완전 동일 프로필 쌍둥이는 하나로만 센다(접근 7 교훈). test는 fit 없이 transform만.
"""
from __future__ import annotations
import hashlib, re
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ID, TARGET = "ID", "SUBCLASS"
_MIS = re.compile(r"^([A-Z])(\d+)([A-Z])$")
LOWMUT = ["SARC", "PRAD", "PCPG", "THYM", "OV"]; EPI = ["STES", "LUSC", "HNSC", "LUAD", "BLCA"]


def _kind(t):
    if t.endswith("*") or "fs" in t: return "lof"
    m = _MIS.match(t)
    if m: return "syn" if m.group(1) == m.group(3) else "mis"
    return "oth"


def _pos(t):
    m = re.match(r"^[A-Z]?(\d+)", t); return int(m.group(1)) if m else -1


# 토큰 생성기: 행 → 토큰 리스트
def tok_type_gene(genes, row):        # gene:kind
    out = []
    for j in np.nonzero(row != "WT")[0]:
        ks = {_kind(t) for t in row[j].split(" ")}; out += [f"{genes[j]}:{k}" for k in ks]
    return out
def tok_type_variant_mis(genes, row):  # gene:variant, missense만
    return [f"{genes[j]}:{t}" for j in np.nonzero(row != "WT")[0] for t in row[j].split(" ") if _kind(t) == "mis"]
def tok_bucket(genes, row):            # gene:pos//25 (기능성 변이만)
    out = set()
    for j in np.nonzero(row != "WT")[0]:
        for t in row[j].split(" "):
            if _kind(t) in ("mis", "lof", "oth") and _pos(t) >= 0: out.add(f"{genes[j]}:b{_pos(t)//25}")
    return list(out)
TOKENIZERS = {"tgene": tok_type_gene, "tvar_mis": tok_type_variant_mis, "bucket": tok_bucket}


class TokenNB:
    """토큰 집합 → 클래스별 Bernoulli NB 로그우도비 점수 (train 부분 fit, 쌍둥이 중복 제거)."""
    def __init__(self, tokenizer, alpha=0.5, min_count=2):
        self.tok, self.alpha, self.min_count = tokenizer, alpha, min_count
    def _mat(self, df, fit):
        G = df[self.genes].to_numpy(); rows, cols = [], []
        if fit: self.vocab = {}
        for i in range(len(df)):
            for t in set(self.tok(self.genes, G[i])):
                if t in self.vocab: rows.append(i); cols.append(self.vocab[t])
                elif fit: self.vocab[t] = len(self.vocab); rows.append(i); cols.append(self.vocab[t])
        return sparse.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(len(df), len(self.vocab)))
    def fit(self, df):
        self.genes = [c for c in df.columns if c not in (ID, TARGET)]
        h = pd.util.hash_pandas_object(df[self.genes], index=False).to_numpy(); keep = ~pd.Series(h).duplicated().to_numpy()   # 쌍둥이 1개만
        d = df[keep]; X = self._mat(d, fit=True)
        le = LabelEncoder(); y = le.fit_transform(d[TARGET]); self.classes_ = le.classes_; K = len(self.classes_)
        Y = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(K, len(y)))
        n_cf = np.asarray((Y @ X).todense()); n_c = np.asarray(Y.sum(1)).ravel()[:, None]; a = self.alpha
        n_f = n_cf.sum(0, keepdims=True); n = n_c.sum()
        keepf = (n_f.ravel() >= self.min_count)
        W = np.log((n_cf + a) / (n_c + 2 * a)) - np.log((n_f - n_cf + a) / (n - n_c + 2 * a)); W[:, ~keepf] = 0
        self.W = W; self.b = np.log(n_c.ravel() / n); return self
    def scores(self, df):
        X = self._mat(df, fit=False); S = np.round(np.asarray(X @ self.W.T) / np.maximum(np.asarray(X.sum(1)).ravel(), 1)[:, None], 4)
        return S + self.b


def _cols(S, classes, tag):
    out = pd.DataFrame(S, columns=[f"cwp_{tag}_{c}" for c in classes]); out[f"cwp_{tag}_max"] = S.max(1); s = np.sort(S, 1); out[f"cwp_{tag}_margin"] = s[:, -1] - s[:, -2]
    return out


class CWPlusFeatures:
    """fit(train) → transform(train)은 내부 OOF, transform(other)은 전체 fit 점수. + 군집 상대 점수."""
    def __init__(self, n_inner=5, seed=42): self.n_inner, self.seed = n_inner, seed
    def fit(self, train):
        y = train[TARGET].to_numpy(); skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed)
        self.models = {k: TokenNB(f).fit(train) for k, f in TOKENIZERS.items()}; self.classes = list(self.models["tgene"].classes_)
        parts = []
        for k, f in TOKENIZERS.items():
            oof = np.zeros((len(train), len(self.classes)))
            for tri, vai in skf.split(train, y):
                m = TokenNB(f).fit(train.iloc[tri]); oof[vai] = m.scores(train.iloc[vai])
            parts.append(_cols(oof, self.classes, k).set_index(train.index))
        self._train_index = train.index; self._train_oof = self._finish(pd.concat(parts, axis=1)); return self
    def _finish(self, X):
        # (iii) 군집 상대 점수: tvar_mis 점수를 군집 안에서 log-softmax
        for name, cl in [("lowmut", LOWMUT), ("epi", EPI)]:
            cols = [f"cwp_tvar_mis_{c}" for c in cl if f"cwp_tvar_mis_{c}" in X.columns]
            S = X[cols].to_numpy(); S = S - S.max(1, keepdims=True); L = S - np.log(np.exp(S).sum(1, keepdims=True))
            for c, j in zip(cl, range(len(cols))): X[f"cwp_{name}_{c}"] = L[:, j]
            X[f"cwp_{name}_margin"] = np.sort(L, 1)[:, -1] - np.sort(L, 1)[:, -2]
        return X
    def transform(self, df):
        if self._train_index is not None and df.index.equals(self._train_index): return self._train_oof
        parts = [_cols(m.scores(df), self.classes, k).set_index(df.index) for k, m in self.models.items()]
        return self._finish(pd.concat(parts, axis=1))
