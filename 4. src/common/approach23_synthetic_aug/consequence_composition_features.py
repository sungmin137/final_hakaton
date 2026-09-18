"""approach23 v24: mutation consequence '조합 구조' 최소 표현.
기존 코드(features.py/spectrum.py)는 missense/nonsense/frameshift/synonymous/other 각각의
개수·비율만 가지고 있고, "한 샘플 안에서 몇 종류가 함께 나타나는가"라는 co-occurrence는
어디에도 없음(인벤토리 확인 완료). LoF 비중(nonsense+frameshift 비율)은 spectrum.py의
spt_trunc와 byte-identical이라 제외. pair/triple 플래그는 의도적으로 만들지 않음(1차 실험 범위 제한).
fit 통계 없음(행 내부 계산) -> test 누수 없음.
"""
import numpy as np
import pandas as pd

from features.features import _mut_type


def consequence_composition_features(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    G = df[genes].to_numpy(); n = len(df)
    MAJOR = ("missense", "nonsense", "frameshift", "synonymous")
    present = np.zeros((n, len(MAJOR)), dtype=bool)
    idx = {t: k for k, t in enumerate(MAJOR)}
    for i, j in zip(*np.nonzero(G != "WT")):
        for tok in G[i, j].split(" "):
            t = _mut_type(tok)
            if t in idx:
                present[i, idx[t]] = True
    n_types = present.sum(axis=1)
    all4_present = (n_types == len(MAJOR)).astype(np.uint8)
    return pd.DataFrame({
        "csq_n_types": n_types,
        "csq_all4_present": all4_present,
    }, index=df.index)
