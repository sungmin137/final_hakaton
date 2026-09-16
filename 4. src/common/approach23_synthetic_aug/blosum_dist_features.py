"""approach23 v23: 샘플 전체 BLOSUM62 분포 요약 중 기존 KnowledgeFeatures(kf_bl_mean, kf_severe_ratio)와
중복되지 않는 것만: 표준편차, 보수적(높은 점수) 치환 비율. 60유전자 제한 없이 전체 missense 대상.
fit 통계 없음(행 내부 계산) -> test 누수 없음. 개별 gene/hotspot 피처 없음, threshold 1개만 사용.
"""
import numpy as np
import pandas as pd

from approach2_knowledge.knowledge_features import variant_severity

CONSERVATIVE_BL = 1.0   # BLOSUM62 >= 1이면 "보수적(허용되는) 치환"으로 간주 (양의 점수 = 진화적으로 흔한 치환)


def blosum_dist_features(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    G = df[genes].to_numpy(); n = len(df)
    bl_values = [[] for _ in range(n)]
    for i, j in zip(*np.nonzero(G != "WT")):
        for tok in G[i, j].split(" "):
            _, bl, _, _, _ = variant_severity(tok)
            if not np.isnan(bl):
                bl_values[i].append(bl)
    bl_std = np.zeros(n)
    conservative_ratio = np.zeros(n)
    for i, vals in enumerate(bl_values):
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        bl_std[i] = arr.std()
        conservative_ratio[i] = (arr >= CONSERVATIVE_BL).mean()
    return pd.DataFrame({
        "bld_std": np.round(bl_std, 4),
        "bld_conservative_ratio": np.round(conservative_ratio, 4),
    }, index=df.index)
