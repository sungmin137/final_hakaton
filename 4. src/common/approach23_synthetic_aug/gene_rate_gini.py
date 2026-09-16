"""approach23 v25: 변이 유전자 집합의 모집단-희귀도(rate) 분포에 대한 Gini 계수.
rate(g)는 반드시 해당 CV fold의 TRAIN 부분집합만으로 fit한다(validation/test 정보 사용 금지).
Gini는 "burden이 아니라, 변이된 유전자들이 흔한 유전자에 몰려있는지 vs 드문 유전자로 분산돼 있는지"를 본다.
degenerate 규칙: mutation 0개 -> gini=0.0(정보 없음, 중립값), 1개 -> gini=0.0(수학적으로 정의됨, 단일값은 불평등 0).
"""
import numpy as np
import pandas as pd


def fit_gene_rate(train_part: pd.DataFrame, genes: list[str]) -> pd.Series:
    """해당 fold의 TRAIN 부분집합만 사용. validation/test는 절대 들어오면 안 됨."""
    sub = train_part[genes]
    rate = (sub != "WT").mean(axis=0)
    return rate


def _gini(x: np.ndarray) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * xs).sum() / (n * s))


def gene_rate_gini_features(df: pd.DataFrame, genes: list[str], rate: pd.Series) -> pd.DataFrame:
    G = df[genes].to_numpy() != "WT"
    rate_arr = rate.reindex(genes).to_numpy()
    n = len(df)
    gini = np.zeros(n)
    n_genes_rated = np.zeros(n, dtype=np.int32)
    for i in range(n):
        idx = np.flatnonzero(G[i])
        n_genes_rated[i] = len(idx)
        gini[i] = _gini(rate_arr[idx])
    return pd.DataFrame({
        "gene_rate_gini": np.round(gini, 4),
    }, index=df.index)
