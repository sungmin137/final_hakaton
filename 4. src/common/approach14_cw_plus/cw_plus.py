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
HI_MUT = 31                            # v7(2026-09-13): 위치 구간 점수를 변이 31개 이상 행에만 (v1에서 저변이 클래스를 망친 희소 토큰을 고변이 구간에 한정)
def tok_bucket_hi(genes, row):         # gene:pos//25, 변이 수 >= HI_MUT 인 행만 (그 외 토큰 없음 → 점수 0)
    return tok_bucket(genes, row) if (row != "WT").sum() >= HI_MUT else []
# --- 2026-09-14 후보 4종 (approach14.md v10 절) ---
def tok_pos(genes, row):               # (1) gene:정확 위치 — hotspot 위치 단위 (V600E·V600K → BRAF:600). 변이 단위보다 덜 희소, 25aa 구간보다 특이적
    out = set()
    for j in np.nonzero(row != "WT")[0]:
        for t in row[j].split(" "):
            if _kind(t) in ("mis", "lof", "oth") and _pos(t) >= 0: out.add(f"{genes[j]}:p{_pos(t)}")
    return list(out)
_AA_CLASS = {**{a: "H" for a in "AVILMFWY"}, **{a: "P" for a in "STNQC"}, **{a: "+" for a in "KRH"}, **{a: "-" for a in "DE"}, "G": "G", "P": "G"}
def tok_prop(genes, row):              # (2) gene:성질변화 — missense의 아미노산 성질 클래스 전이 (H 소수성 / P 극성 / +,- 전하 / G 특수)
    out = set()
    for j in np.nonzero(row != "WT")[0]:
        for t in row[j].split(" "):
            m = _MIS.match(t)
            if m and m.group(1) != m.group(3): out.add(f"{genes[j]}:{_AA_CLASS.get(m.group(1), '?')}>{_AA_CLASS.get(m.group(3), '?')}")
    return list(out)
def tok_band_gene(genes, row):         # (4) 변이 수 밴드(lo ≤10 / mid 11~30 / hi ≥31)별로 gene:kind 통계를 따로 학습
    n = (row != "WT").sum(); band = "lo" if n <= 10 else ("mid" if n <= 30 else "hi")
    return [f"{band}|{t}" for t in tok_type_gene(genes, row)]
class PairTokenizer:                   # (3) 자주 변이되는 상위 K 유전자 사이의 공변이 쌍 A&B (K는 fit 데이터로만 결정)
    def __init__(self, k=40): self.k, self.top = k, None
    def prepare(self, df, genes):
        freq = (df[genes] != "WT").sum(0); self.top = set(freq.sort_values(ascending=False).index[:self.k])
    def __call__(self, genes, row):
        g = sorted(genes[j] for j in np.nonzero(row != "WT")[0] if genes[j] in self.top)
        return [f"{a}&{b}" for i, a in enumerate(g) for b in g[i + 1:]]
TOKENIZERS_ALL = {"tgene": tok_type_gene, "tvar_mis": tok_type_variant_mis, "bucket": tok_bucket, "bucket_hi": tok_bucket_hi,
                  "pos": tok_pos, "prop": tok_prop, "band": tok_band_gene, "pair": PairTokenizer}
TOKENIZERS = {"tgene": tok_type_gene}          # v2: 유전자 단위 유형 분리만 (v1의 희소 토큰 계열은 저변이 클래스에 해로웠음)


class TokenNB:
    """토큰 집합 → 클래스별 Bernoulli NB 로그우도비 점수 (train 부분 fit, 쌍둥이 중복 제거)."""
    def __init__(self, tokenizer, alpha=0.5, min_count=2, fit_tokened_only=False):
        self.tok, self.alpha, self.min_count, self.fit_tokened_only = tokenizer, alpha, min_count, fit_tokened_only
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
        if hasattr(self.tok, "prepare"): self.tok.prepare(df, self.genes)   # fit 데이터로만 상위 유전자 결정
        h = pd.util.hash_pandas_object(df[self.genes], index=False).to_numpy(); keep = ~pd.Series(h).duplicated().to_numpy()   # 쌍둥이 1개만
        d = df[keep]; X = self._mat(d, fit=True)
        le = LabelEncoder().fit(df[TARGET]); self.classes_ = le.classes_; K = len(self.classes_)   # 클래스 목록은 항상 전체 기준
        if self.fit_tokened_only:                # 토큰이 있는 행(고변이)만으로 클래스 통계를 낸다 (없는 클래스는 n_c=0 → 스무딩)
            m = np.asarray(X.sum(1)).ravel() > 0; d, X = d[m], X[m]
        y = le.transform(d[TARGET])
        Y = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(K, len(y)))
        n_cf = np.asarray((Y @ X).todense()); n_c = np.asarray(Y.sum(1)).ravel()[:, None]; a = self.alpha
        n_f = n_cf.sum(0, keepdims=True); n = n_c.sum()
        keepf = (n_f.ravel() >= self.min_count)
        W = np.log((n_cf + a) / (n_c + 2 * a)) - np.log((n_f - n_cf + a) / (n - n_c + 2 * a)); W[:, ~keepf] = 0
        self.W = W; self.b = np.log((n_c.ravel() + a) / (n + K * a)) if self.fit_tokened_only else np.log(n_c.ravel() / n); return self   # 기존 모드는 v2/v3 재현 위해 그대로
    def scores(self, df):
        X = self._mat(df, fit=False); n_tok = np.asarray(X.sum(1)).ravel()
        S = np.round(np.asarray(X @ self.W.T) / np.maximum(n_tok, 1)[:, None], 4)
        S = S + self.b
        S[n_tok == 0] = 0.0                      # v2: 토큰 없는 행은 사전확률 상수 대신 0 (저변이 클래스 붕괴 방지)
        return S


def _cols(S, classes, tag):
    out = pd.DataFrame(S, columns=[f"cwp_{tag}_{c}" for c in classes]); out[f"cwp_{tag}_max"] = S.max(1); s = np.sort(S, 1); out[f"cwp_{tag}_margin"] = s[:, -1] - s[:, -2]
    return out


class CWPlusFeatures:
    """fit(train) → transform(train)은 내부 OOF, transform(other)은 전체 fit 점수. + 군집 상대 점수."""
    def __init__(self, n_inner=5, seed=42, extra=None):
        self.n_inner, self.seed = n_inner, seed
        self.toks = dict(TOKENIZERS); self.only = set()
        if extra:                                # 예: "bucket_hi" → 고변이 한정 위치 구간 점수 추가 (fit도 토큰 있는 행만)
            self.toks[extra] = TOKENIZERS_ALL[extra]
            if extra == "bucket_hi": self.only.add(extra)
    def _nb(self, k, f):
        f = f() if isinstance(f, type) else f     # PairTokenizer처럼 상태 있는 생성기는 fit마다 새 인스턴스
        return TokenNB(f, fit_tokened_only=(k in self.only))
    def fit(self, train):
        y = train[TARGET].to_numpy(); skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed)
        self.models = {k: self._nb(k, f).fit(train) for k, f in self.toks.items()}; self.classes = list(self.models["tgene"].classes_)
        parts = []
        for k, f in self.toks.items():
            oof = np.zeros((len(train), len(self.classes)))
            for tri, vai in skf.split(train, y):
                m = self._nb(k, f).fit(train.iloc[tri]); oof[vai] = m.scores(train.iloc[vai])
            parts.append(_cols(oof, self.classes, k).set_index(train.index))
        self._train_index = train.index; self._train_oof = self._finish(pd.concat(parts, axis=1)); return self
    def _finish(self, X):
        # (iii) 군집 상대 점수: tvar_mis 점수를 군집 안에서 log-softmax
        for name, cl in [("lowmut", LOWMUT), ("epi", EPI)]:
            cols = [f"cwp_tgene_{c}" for c in cl if f"cwp_tgene_{c}" in X.columns]
            S = X[cols].to_numpy(); S = S - S.max(1, keepdims=True); L = S - np.log(np.exp(S).sum(1, keepdims=True))
            for c, j in zip(cl, range(len(cols))): X[f"cwp_{name}_{c}"] = L[:, j]
            X[f"cwp_{name}_margin"] = np.sort(L, 1)[:, -1] - np.sort(L, 1)[:, -2]
        return X
    def transform(self, df):
        if self._train_index is not None and df.index.equals(self._train_index): return self._train_oof
        parts = [_cols(m.scores(df), self.classes, k).set_index(df.index) for k, m in self.models.items()]
        return self._finish(pd.concat(parts, axis=1))
