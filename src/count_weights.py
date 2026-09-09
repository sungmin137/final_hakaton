"""접근 1 — 클래스별 변이 '개수' 기반 가중치 모델. train.csv만 사용.

아이디어(회의): 같은 클래스끼리 묶어 변이를 개수 순으로 정리하고,
많이 나온 변이일수록 큰 가중치를 준다. 샘플의 변이들에 대해 클래스별 가중치를
합산해 가장 큰 클래스로 예측한다.

level  : gene    = 유전자 단위 (변이 유무)
         variant = (유전자, 변이) 단위 (예: IDH1:R132H)
scheme : count = 클래스 내 등장 개수 그대로 (회의안 그대로)
         rate  = 개수 / 클래스 인원  (클래스 크기 보정)
         spec  = rate × (이 클래스 rate / 전체 클래스 rate 합)  (특이성 보정)
         nb    = Bernoulli Naive Bayes 로그 우도비 (개수 기반의 확률적 정식화)
agg    : sum  = 가중치 합 / mean = 샘플 변이 수로 나눔 (초과변이 샘플 보정)

실행: PYTHONPATH=src python3 src/count_weights.py   → 모든 조합 5-Fold 평가
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
ID, TARGET = "ID", "SUBCLASS"


# ---------------------------------------------------------------- 희소 행렬 만들기
def gene_matrix(df: pd.DataFrame, genes: list[str]) -> sparse.csr_matrix:
    return sparse.csr_matrix((df[genes] != "WT").to_numpy(dtype=np.float32))


def variant_tokens(df: pd.DataFrame, genes: list[str]):
    """행별 'GENE:variant' 토큰 리스트 생성기."""
    G = df[genes].to_numpy()
    for i in range(len(df)):
        row = G[i]
        toks = []
        for j in np.nonzero(row != "WT")[0]:
            g = genes[j]
            toks.extend(f"{g}:{t}" for t in row[j].split(" "))
        yield toks


def variant_matrix(df: pd.DataFrame, genes: list[str], vocab: dict | None = None):
    """vocab이 None이면 df에서 vocab을 만들고(fit), 있으면 그 vocab으로만 변환(transform)."""
    fit = vocab is None
    if fit:
        vocab = {}
    rows, cols = [], []
    for i, toks in enumerate(variant_tokens(df, genes)):
        for t in set(toks):
            if t in vocab:
                cols.append(vocab[t]); rows.append(i)
            elif fit:
                vocab[t] = len(vocab); cols.append(vocab[t]); rows.append(i)
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(df), len(vocab)))
    return X, vocab


# ---------------------------------------------------------------- 가중치 모델
class CountWeightModel:
    def __init__(self, level="gene", scheme="count", agg="sum", alpha=0.5):
        self.level, self.scheme, self.agg, self.alpha = level, scheme, agg, alpha
        self.genes: list[str] = []
        self.vocab: dict | None = None
        self.classes_: np.ndarray | None = None
        self.W: np.ndarray | None = None          # (n_classes, n_features)
        self.b: np.ndarray | None = None          # 클래스별 절편 (nb용)

    def _X(self, df: pd.DataFrame, fit: bool):
        if self.level == "gene":
            return gene_matrix(df, self.genes)
        X, vocab = variant_matrix(df, self.genes, None if fit else self.vocab)
        if fit:
            self.vocab = vocab
        return X

    def fit(self, df: pd.DataFrame) -> "CountWeightModel":
        self.genes = [c for c in df.columns if c not in (ID, TARGET)]
        X = self._X(df, fit=True)
        le = LabelEncoder(); y = le.fit_transform(df[TARGET]); self.classes_ = le.classes_
        K = len(self.classes_)
        Y = sparse.csr_matrix((np.ones(len(y), np.float32), (y, np.arange(len(y)))), shape=(K, len(y)))
        n_cf = np.asarray((Y @ X).todense())            # 클래스 c에서 피처 f 등장 개수
        n_c = np.asarray(Y.sum(1)).ravel()[:, None]      # 클래스 인원
        a = self.alpha
        if self.scheme == "count":
            W = n_cf
        elif self.scheme == "rate":
            W = n_cf / n_c
        elif self.scheme == "spec":
            rate = n_cf / n_c
            W = rate * rate / (rate.sum(0, keepdims=True) + 1e-9)
        elif self.scheme == "nb":
            n_f = n_cf.sum(0, keepdims=True); n = n_c.sum()
            p_in = (n_cf + a) / (n_c + 2 * a)                       # P(f | c)
            p_out = (n_f - n_cf + a) / (n - n_c + 2 * a)            # P(f | not c)
            W = np.log(p_in) - np.log(p_out)
        else:
            raise ValueError(self.scheme)
        self.W = W.astype(np.float32)
        self.b = np.log(n_c.ravel() / n_c.sum()).astype(np.float32) if self.scheme == "nb" else np.zeros(K, np.float32)
        return self

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df, fit=False)
        S = np.asarray(X @ self.W.T)
        if self.agg == "mean":
            S = S / np.maximum(np.asarray(X.sum(1)).ravel(), 1)[:, None]
        return S + self.b

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.scores(df).argmax(1)]


# ---------------------------------------------------------------- 평가
def cv_eval(train: pd.DataFrame, level: str, scheme: str, agg: str, n_splits=5, seed=42) -> dict:
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(train), dtype=object)
    for tri, vai in skf.split(train, y):
        m = CountWeightModel(level, scheme, agg).fit(train.iloc[tri])
        pred[vai] = m.predict(train.iloc[vai])
    return dict(level=level, scheme=scheme, agg=agg,
                macro_f1=round(f1_score(y, pred, average="macro"), 4),
                acc=round(accuracy_score(y, pred), 4))


def main() -> None:
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    res = []
    for level, scheme, agg in itertools.product(["gene", "variant"], ["count", "rate", "spec", "nb"], ["sum", "mean"]):
        t = time.time(); r = cv_eval(train, level, scheme, agg); r["sec"] = round(time.time() - t)
        res.append(r); print(r)
    df = pd.DataFrame(res).sort_values("macro_f1", ascending=False)
    out = ROOT / "experiments" / "2026-09-09_approach1_count_weight"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cv_results.csv", index=False)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- XGB용 피처 (접근 1 점수)
SCORE_SPECS = [("variant", "nb", "mean"), ("variant", "count", "sum"), ("variant", "rate", "sum"),
               ("gene", "spec", "sum"), ("gene", "rate", "sum")]


def _score_cols(model: CountWeightModel, df: pd.DataFrame, tag: str) -> pd.DataFrame:
    S = model.scores(df)
    cols = [f"cw_{tag}_{c}" for c in model.classes_]
    out = pd.DataFrame(S, columns=cols, index=df.index)
    out[f"cw_{tag}_max"] = S.max(1)
    out[f"cw_{tag}_margin"] = np.sort(S, 1)[:, -1] - np.sort(S, 1)[:, -2]
    return out


class CountWeightFeatures:
    """fit(train) → transform(train)은 내부 K-Fold OOF 점수, transform(other)은 전체 fit 점수."""

    def __init__(self, specs=SCORE_SPECS, n_inner=5, seed=42):
        self.specs, self.n_inner, self.seed = specs, n_inner, seed
        self.models: dict[str, CountWeightModel] = {}
        self._train_index = None
        self._train_oof: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> "CountWeightFeatures":
        y = train[TARGET].to_numpy()
        skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed)
        parts = []
        for level, scheme, agg in self.specs:
            tag = f"{level}_{scheme}"
            self.models[tag] = CountWeightModel(level, scheme, agg).fit(train)
            oof = []
            for tri, vai in skf.split(train, y):
                m = CountWeightModel(level, scheme, agg).fit(train.iloc[tri])
                oof.append(_score_cols(m, train.iloc[vai], tag))
            parts.append(pd.concat(oof).loc[train.index])
        self._train_index = train.index
        self._train_oof = pd.concat(parts, axis=1)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._train_index is not None and df.index.equals(self._train_index):
            return self._train_oof
        return pd.concat([_score_cols(m, df, tag) for tag, m in self.models.items()], axis=1)
