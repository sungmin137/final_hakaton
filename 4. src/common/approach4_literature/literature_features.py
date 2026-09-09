"""접근 4 — 문헌 지식 기반 피처 (v5). 환자 단위 외부 데이터 없음. fit 통계 없음(순수 지식 + 행 단위 계산) → test 누수 불가.

샘플별로:
  lit_<CLASS>_score  : 그 암종의 문헌 driver 중 기능성 변이가 있는 유전자의 문헌 빈도(%) 가중 합 (26개)
  lit_<CLASS>_n      : 그 암종의 문헌 driver 중 변이된 유전자 수 (26개)
  lit_<CLASS>_hs     : 그 암종의 문헌 driver 중 문헌 hotspot 위치와 일치하는 변이 수 (26개)
  pw_<PATHWAY>       : 경로별 기능성 변이 유전자 수 (21개)
  role_og_hotspot    : oncogene의 hotspot 위치 missense 수  ← "활성화" 신호
  role_tsg_lof       : tumor suppressor의 종결·프레임시프트 수 ← "기능 상실" 신호
  role_consistent    : 위 둘의 합 (문헌 역할과 일치하는 변이 수), role_inconsistent: OG의 LoF + TSG의 비-hotspot missense
  lit_best_class_margin: lit score 1위-2위 차이
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd
from approach4_literature.literature_map import GENE_INFO, CLASS_DRIVERS, PATHWAYS, hotspot_match

_MIS = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _kind(t: str) -> str:
    if t.endswith("*") or "fs" in t: return "lof"
    m = _MIS.match(t)
    if m: return "syn" if m.group(1) == m.group(3) else "mis"
    return "other"


class LiteratureFeatures:
    def __init__(self):
        self.genes: list[str] = []; self.classes = sorted(CLASS_DRIVERS)

    def fit(self, df: pd.DataFrame) -> "LiteratureFeatures":
        self.genes = [c for c in df.columns if c not in ("ID", "SUBCLASS")]
        gset = set(self.genes)
        self.cls_w = {c: {g: (f if f else 5.0) for g, f in d.items() if g in gset} for c, d in CLASS_DRIVERS.items()}
        self.pw_of = {g: v["pathway"] for g, v in GENE_INFO.items() if g in gset}
        self.role_of = {g: v["role"] for g, v in GENE_INFO.items() if g in gset}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        G = df[self.genes].to_numpy(); n = len(df); C = self.classes; ci = {c: i for i, c in enumerate(C)}; pi = {p: i for i, p in enumerate(PATHWAYS)}
        score = np.zeros((n, len(C))); cnt = np.zeros((n, len(C))); hsn = np.zeros((n, len(C))); pw = np.zeros((n, len(PATHWAYS)))
        og_hs = np.zeros(n); tsg_lof = np.zeros(n); og_lof = np.zeros(n); tsg_mis = np.zeros(n)
        for i, j in zip(*np.nonzero(G != "WT")):
            g = self.genes[j]; toks = G[i, j].split(" ")
            kinds = [_kind(t) for t in toks]; functional = any(k in ("lof", "mis", "other") for k in kinds)
            hs = [hotspot_match(g, t) for t, k in zip(toks, kinds) if k == "mis"]; n_hs = sum(h is not None for h in hs)
            n_lof = kinds.count("lof"); n_mis = kinds.count("mis")
            if functional:
                for c, w in self.cls_w.items():
                    if g in w:
                        score[i, ci[c]] += w[g]; cnt[i, ci[c]] += 1; hsn[i, ci[c]] += n_hs
                p = self.pw_of.get(g)
                if p: pw[i, pi[p]] += 1
            r = self.role_of.get(g)
            if r in ("OG", "OG/TSG"): og_hs[i] += n_hs; og_lof[i] += n_lof if r == "OG" else 0
            if r in ("TSG", "OG/TSG"): tsg_lof[i] += n_lof; tsg_mis[i] += (n_mis - n_hs) if r == "TSG" else 0
        out = {}
        for c in C:
            out[f"lit_{c}_score"] = score[:, ci[c]]; out[f"lit_{c}_n"] = cnt[:, ci[c]]; out[f"lit_{c}_hs"] = hsn[:, ci[c]]
            out[f"lit_{c}_norm"] = score[:, ci[c]] / max(sum(self.cls_w[c].values()), 1.0)   # 그 암종 driver 총 가중치 대비 비율
        for p in PATHWAYS: out[f"pw_{p}"] = pw[:, pi[p]]
        s = np.sort(score, 1); out["lit_best_class_margin"] = s[:, -1] - s[:, -2]; out["lit_total_score"] = score.sum(1)
        out["role_og_hotspot"] = og_hs; out["role_tsg_lof"] = tsg_lof; out["role_consistent"] = og_hs + tsg_lof; out["role_inconsistent"] = og_lof + tsg_mis
        return pd.DataFrame(out, index=df.index)
