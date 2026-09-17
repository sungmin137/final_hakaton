"""approach23 v21: gene identity -> biological functional module(pathway) 표현.
기존 approach4_literature/literature_map.py의 GENE_INFO/PATHWAYS(Sanchez-Vega et al. 2018 Cell,
21개 경로·189개 유전자, 고정 도메인 지식)를 재사용한다 — 새로 만들지 않음, 중복 방지.
기존 approach4의 pw_<PATHWAY>(단순 변이 유전자 개수)와 달리, 여기서는 유형 인지(LoF vs missense)를
추가한다 — 이게 approach4에 없던 부분이라 "새 정보축"인지 확인 대상.
fit 통계 없음(GENE_INFO는 고정 상수) -> test 누수 불가.
"""
import re
import numpy as np
import pandas as pd

from approach4_literature.literature_map import GENE_INFO, PATHWAYS

_MIS = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _kind(t: str) -> str:
    if t.endswith("*") or "fs" in t:
        return "lof"
    m = _MIS.match(t)
    if m:
        return "syn" if m.group(1) == m.group(3) else "mis"
    return "other"


def pathway_module_features(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """샘플별: 경로별 변이 유전자 수(cnt, 기존 approach4와 동일), 경로별 LoF 변이 수(lof, 신규),
    경로별 missense 변이 수(mis, 신규). 21개 경로 x 3 = 63열."""
    gset = set(genes)
    pw_of = {g: v["pathway"] for g, v in GENE_INFO.items() if g in gset}
    gene_idx = {g: j for j, g in enumerate(genes) if g in pw_of}
    n = len(df)
    pi = {p: i for i, p in enumerate(PATHWAYS)}
    cnt = np.zeros((n, len(PATHWAYS)))
    lof = np.zeros((n, len(PATHWAYS)))
    mis = np.zeros((n, len(PATHWAYS)))
    G = df[list(gene_idx.keys())].to_numpy()
    cols = list(gene_idx.keys())
    for j, g in enumerate(cols):
        p = pi[pw_of[g]]
        col = G[:, j]
        mut_rows = np.flatnonzero(col != "WT")
        for i in mut_rows:
            toks = col[i].split(" ")
            kinds = [_kind(t) for t in toks]
            if any(k in ("lof", "mis", "other") for k in kinds):
                cnt[i, p] += 1
            if "lof" in kinds:
                lof[i, p] += 1
            if "mis" in kinds:
                mis[i, p] += 1
    out = {}
    for p in PATHWAYS:
        out[f"pwm_{p}_cnt"] = cnt[:, pi[p]]
        out[f"pwm_{p}_lof"] = lof[:, pi[p]]
        out[f"pwm_{p}_mis"] = mis[:, pi[p]]
    return pd.DataFrame(out, index=df.index)
