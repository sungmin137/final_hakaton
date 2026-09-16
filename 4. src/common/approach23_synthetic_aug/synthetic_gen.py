"""GBMLGG/LGG 전용 synthetic 데이터 생성 (2-도너 crossover 교란).

외부 데이터·biological prior 없음 — 호출자가 넘긴 pool_df(반드시 train 파티션으로 한정된 실제 행)만 사용한다.
같은 클래스의 실제 행 A를 베이스로 삼고, A·B(둘 다 실제 행)가 다른 유전자 중 일부만 B의 실제 토큰 값으로
치환한다. 유전자별 독립 확률 샘플링과 달리 A의 co-mutation 블록 대부분이 그대로 유지되고,
mutation burden도 A·B 두 실제 표본 사이 값으로 자연스럽게 떨어진다.

2. team/approaches/approach23.md 참고.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.features import ID, TARGET, gene_columns


def generate_synthetic(pool_df: pd.DataFrame, target_class: str, n: int, seed: int,
                        swap_frac: float = 0.2, id_prefix: str = "SYN", max_redraw: int = 20) -> pd.DataFrame:
    """pool_df: 이미 target_class로 필터링되고, train 파티션(fold의 학습 쪽 또는 전체 train)으로
    한정된 DataFrame이어야 한다 — validation/test 행이 섞이지 않게 하는 것은 호출자 책임.

    반환: pool_df와 같은 유전자 컬럼 + ID + SUBCLASS를 가진 synthetic n행.
    """
    genes = gene_columns(pool_df)
    G = pool_df[genes].to_numpy(dtype=object)
    n_pool = len(pool_df)
    if n_pool < 2:
        raise ValueError(f"{target_class}: 도너 풀이 너무 작음 ({n_pool}행)")

    seen = {tuple(row) for row in G}   # 실제 행과의 완전중복 방지
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n):
        candidate = None
        for _ in range(max_redraw):
            ia, ib = rng.choice(n_pool, size=2, replace=False)
            a, b = G[ia], G[ib]
            diff_idx = np.flatnonzero(a != b)
            if len(diff_idx) == 0:
                continue
            k = min(max(1, round(swap_frac * len(diff_idx))), len(diff_idx))
            swap_pos = rng.choice(diff_idx, size=k, replace=False)
            cand = a.copy()
            cand[swap_pos] = b[swap_pos]
            key = tuple(cand)
            if key not in seen:
                candidate = cand
                seen.add(key)
                break
        if candidate is None:   # max_redraw번 다 실패(사실상 거의 발생하지 않음): 마지막 후보라도 사용
            candidate = cand
            seen.add(tuple(cand))
        rows.append(candidate)

    out = pd.DataFrame(rows, columns=genes)
    out.insert(0, TARGET, target_class)
    out.insert(0, ID, [f"{id_prefix}_{target_class}_{seed}_{j:04d}" for j in range(n)])
    return out
