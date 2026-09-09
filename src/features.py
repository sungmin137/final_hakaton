"""피처 생성. train/test 모두 같은 함수를 쓰되, 통계 fit이 필요한 단계는 없다(순수 행 단위 변환).
따라서 test 누수 없음. test.csv 로드는 predict 단계에서만 한다.
"""
import re
import numpy as np
import pandas as pd

ID, TARGET = "ID", "SUBCLASS"


def gene_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (ID, TARGET)]


def _mut_type(tok: str) -> str:
    if tok.endswith("*"):
        return "nonsense"
    if "fs" in tok:
        return "frameshift"
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
    if m:
        return "synonymous" if m.group(1) == m.group(3) else "missense"
    return "other"


def build_features(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """v1: 유전자별 변이 여부(0/1) + 샘플 단위 카운트 피처."""
    sub = df[genes]
    is_mut = (sub != "WT").astype(np.uint8)
    is_mut.columns = [f"g_{g}" for g in genes]

    # 셀 단위 변이 유형 카운트
    flat = sub.to_numpy().ravel()
    n_rows, n_cols = sub.shape
    counts = {k: np.zeros(n_rows, dtype=np.int32)
              for k in ("missense", "synonymous", "nonsense", "frameshift", "other", "n_tokens")}
    for idx in np.flatnonzero(flat != "WT"):
        r = idx // n_cols
        for tok in flat[idx].split(" "):
            counts[_mut_type(tok)][r] += 1
            counts["n_tokens"][r] += 1

    agg = pd.DataFrame({
        "n_mut_genes": is_mut.sum(axis=1).to_numpy(),
        **{(k if k == "n_tokens" else f"n_{k}"): v for k, v in counts.items()},
    }, index=df.index)
    agg["n_functional"] = agg["n_missense"] + agg["n_nonsense"] + agg["n_frameshift"] + agg["n_other"]
    agg["log_n_mut_genes"] = np.log1p(agg["n_mut_genes"])
    agg["syn_ratio"] = agg["n_synonymous"] / agg["n_tokens"].clip(lower=1)

    return pd.concat([is_mut, agg], axis=1)
