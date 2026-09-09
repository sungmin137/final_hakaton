"""도메인 지식 기반 변이 특성 피처 (접근 2와 연결). 환자 단위 외부 데이터는 사용하지 않는다.

각 missense 변이 X###Y 에 대해:
  - BLOSUM62(X,Y)      : 진화적 치환 허용도. 낮을수록(음수) 드물고 파괴적인 치환   [Henikoff 1992]
  - Δ소수성            : Kyte-Doolittle hydropathy 차이의 절댓값                     [Kyte & Doolittle 1982]
  - Δ부피              : 잔기 부피(Å^3) 차이의 절댓값                                [Zamyatnin 1972]
  - Δ전하              : 전하(+1/0/-1) 변화 여부
종결/프레임시프트는 최대 심각도(=1)로, 동의 변이는 0으로 둔다.

샘플 단위 집계: 심각 변이 수, 평균/최소 BLOSUM, 최대 Δ특성, 심각도 합, 그리고
train에서 변이가 많은 상위 유전자(TOP_GENES개)별 최소 BLOSUM(=그 유전자가 얼마나 세게 망가졌나).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

_B62 = substitution_matrices.load("BLOSUM62")
_HYDRO = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9,
              M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
_VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7,
            K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
_CHARGE = dict(R=1, K=1, H=0.5, D=-1, E=-1)
_MIS = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def variant_severity(tok: str) -> tuple[float, float, float, float, float]:
    """(severity 0~1, blosum, d_hydro, d_vol, d_charge). 비-missense는 blosum 등을 NaN."""
    if tok.endswith("*") or "fs" in tok:
        return 1.0, np.nan, np.nan, np.nan, np.nan
    m = _MIS.match(tok)
    if not m:
        return 0.5, np.nan, np.nan, np.nan, np.nan            # in-frame indel 등
    a, b = m.group(1), m.group(3)
    if a == b:
        return 0.0, np.nan, np.nan, np.nan, np.nan            # 동의 변이
    if a not in _HYDRO or b not in _HYDRO:
        return 0.5, np.nan, np.nan, np.nan, np.nan
    bl = float(_B62[a, b])
    dh = abs(_HYDRO[a] - _HYDRO[b]); dv = abs(_VOL[a] - _VOL[b]); dc = abs(_CHARGE.get(a, 0) - _CHARGE.get(b, 0))
    sev = min(1.0, max(0.0, (2 - bl) / 6))                    # BLOSUM +2→0, -4→1
    return sev, bl, dh, dv, dc


class KnowledgeFeatures:
    def __init__(self, top_genes: int = 60):
        self.top_genes = top_genes
        self.genes: list[str] = []
        self.sel: list[str] = []

    def fit(self, df: pd.DataFrame) -> "KnowledgeFeatures":
        self.genes = [c for c in df.columns if c not in ("ID", "SUBCLASS")]
        rate = (df[self.genes] != "WT").mean()
        self.sel = list(rate.sort_values(ascending=False).head(self.top_genes).index)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        G = df[self.genes].to_numpy(); n = len(df)
        sel_idx = {g: i for i, g in enumerate(self.sel)}
        gene_min_bl = np.full((n, len(self.sel)), 5.0)        # 변이 없으면 +5 (무해)
        sev_sum = np.zeros(n); n_severe = np.zeros(n); n_mis = np.zeros(n)
        bl_sum = np.zeros(n); bl_min = np.full(n, 5.0)
        dh_max = np.zeros(n); dv_max = np.zeros(n); dc_sum = np.zeros(n)
        for i, j in zip(*np.nonzero(G != "WT")):
            g = self.genes[j]
            for tok in G[i, j].split(" "):
                sev, bl, dh, dv, dc = variant_severity(tok)
                sev_sum[i] += sev; n_severe[i] += sev >= 0.67
                if not np.isnan(bl):
                    n_mis[i] += 1; bl_sum[i] += bl; bl_min[i] = min(bl_min[i], bl)
                    dh_max[i] = max(dh_max[i], dh); dv_max[i] = max(dv_max[i], dv); dc_sum[i] += dc
                    if g in sel_idx:
                        gene_min_bl[i, sel_idx[g]] = min(gene_min_bl[i, sel_idx[g]], bl)
                elif sev == 1.0 and g in sel_idx:
                    gene_min_bl[i, sel_idx[g]] = -5.0                  # LoF는 최악으로
        out = pd.DataFrame({
            "kf_sev_sum": sev_sum, "kf_n_severe": n_severe,
            "kf_bl_mean": np.where(n_mis > 0, bl_sum / np.maximum(n_mis, 1), 0.0), "kf_bl_min": bl_min,
            "kf_dhydro_max": dh_max, "kf_dvol_max": dv_max, "kf_dcharge_sum": dc_sum,
            "kf_severe_ratio": n_severe / np.maximum(n_mis, 1),
        }, index=df.index)
        gm = pd.DataFrame(gene_min_bl, columns=[f"kf_minbl_{g}" for g in self.sel], index=df.index)
        return pd.concat([out, gm], axis=1)
