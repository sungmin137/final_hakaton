"""접근 7 — 전처리 확장 피처. train fold에서만 fit, test 통계 미사용.

(1) 유전자별 변이 유형 분리 이진화: 학습 fold에서 변이 샘플 ≥ MIN_N 인 유전자에 대해 mis_<g>(비-hotspot 포함 missense), lof_t_<g>(종결·프레임시프트), syn_<g>(동의) 0/1
(2) 단백질 도메인 구간 피처: 문헌(UniProt/TCGA 논문) 기준 기능 도메인 아미노산 범위. dom_<gene>_<domain> = 그 구간에 기능성 변이가 있으면 1
(3) 변이 부담 정규화·희귀 변이:
    frac_functional, frac_lof, frac_syn (샘플 변이 중 비율), frac_driver (문헌 driver 유전자 변이 / 변이 유전자 수),
    n_private (학습 fold에 한 번도 없던 변이 토큰 수), n_singleton (학습 fold에서 정확히 1명에게만 있던 토큰 수), frac_private,
      ※ 학습 fold 자신을 transform 할 때는 leave-one-out: 자기 자신을 제외한 개수(count-1)로 계산해 검증·test 행과 뜻을 맞춘다.
    n_mut_rank (학습 fold 기준 변이 수 백분위, 0~1)
"""
from __future__ import annotations
import re
from collections import Counter
import numpy as np, pandas as pd
from approach4_literature.literature_map import GENE_INFO

MIN_N = 20
_MIS = re.compile(r"^([A-Z])(\d+)([A-Z*])$")
_POS = re.compile(r"^[A-Z]?(\d+)")

# 문헌 기반 단백질 도메인 (아미노산 위치 범위, 근사). 출처: UniProt 도메인 주석, TCGA marker paper, Vogelstein 2013
DOMAINS: dict[str, dict[str, tuple[int, int]]] = {
    "TP53": {"TAD": (1, 61), "PRD": (62, 94), "DBD": (95, 292), "tetramer": (320, 356), "Cterm": (357, 393)},
    "PIK3CA": {"ABD": (16, 105), "RBD": (187, 289), "C2": (330, 487), "helical": (517, 694), "kinase": (797, 1068)},
    "APC": {"armadillo": (453, 767), "rep15": (1020, 1169), "MCR": (1286, 1513), "rep20_rest": (1514, 2033), "basic": (2200, 2400)},
    "PTEN": {"phosphatase": (14, 185), "C2": (186, 351)},
    "BRAF": {"RBD": (155, 227), "CRD": (234, 280), "kinase": (457, 717)},
    "EGFR": {"extracellular": (25, 645), "kinase": (712, 979)},
    "ERBB2": {"extracellular": (23, 652), "kinase": (720, 987)},
    "KIT": {"extracellular": (1, 520), "juxtamembrane": (544, 581), "kinase": (589, 937)},
    "MET": {"juxtamembrane": (963, 1010), "kinase": (1078, 1345)},
    "FGFR3": {"extracellular": (23, 375), "kinase": (472, 748)},
    "CTNNB1": {"degron_exon3": (1, 60), "armadillo": (141, 664)},
    "NFE2L2": {"Neh2": (1, 86)},
    "FBXW7": {"WD40": (370, 700)},
    "VHL": {"beta": (63, 154), "alpha": (155, 192)},
    "NOTCH1": {"EGF": (20, 1426), "NRR_HD": (1450, 1730), "PEST": (2400, 2555)},
    "ATRX": {"ADD": (159, 296), "helicase": (2000, 2492)},
    "RB1": {"pocketA": (373, 579), "pocketB": (640, 771)},
    "SPOP": {"MATH": (28, 166)},
    "GATA3": {"ZnF": (260, 350)},
    "KMT2D": {"SET": (5397, 5513)},
    "HRAS": {"switch": (10, 75)},
    "NF1": {"GRD": (1198, 1530)},
    "POLE": {"exonuclease": (268, 471)},
    "RHOA": {"switchI": (28, 44)},
    "MYD88": {"TIR": (159, 296)},
    "EZH2": {"SET": (612, 727)},
    "CREBBP": {"HAT": (1342, 1649)},
    "NPM1": {"Cterm_exon12": (280, 294)},
    "CDKN2A": {"ankyrin": (1, 156)},
    "IDH1": {"active": (125, 140)}, "IDH2": {"active": (135, 180)},
    "MTOR": {"kinase": (2181, 2431)}, "PIK3R1": {"iSH2": (430, 600)},
    "SMARCA4": {"helicase": (770, 1200)}, "ARID5B": {"ARID": (310, 400)},
    "CDH1": {"cadherin": (155, 700)}, "MAP3K1": {"kinase": (1240, 1500)},
    "ERCC2": {"helicase": (1, 760)}, "RXRA": {"LBD": (230, 460)}, "ELF3": {"ETS": (275, 360)},
}


def _kind(tok: str) -> str:
    if tok.endswith("*") or "fs" in tok: return "lof"
    m = _MIS.match(tok)
    if m: return "syn" if m.group(1) == m.group(3) else "mis"
    return "other"


def _pos(tok: str) -> int | None:
    m = _POS.match(tok); return int(m.group(1)) if m else None


class PreprocessFeatures:
    def fit(self, df: pd.DataFrame) -> "PreprocessFeatures":
        self.genes = [c for c in df.columns if c not in ("ID", "SUBCLASS")]
        M = (df[self.genes] != "WT"); cnt = M.sum(0)
        self.type_genes = [g for g in self.genes if cnt[g] >= MIN_N]
        G = df[self.genes].to_numpy(); tokc: Counter = Counter()
        # 완전 동일 프로필(쌍둥이)은 하나로만 센다 — 쌍둥이 행의 모든 변이가 'singleton 공유'로 잡혀 라벨 누수처럼 작동하는 것을 방지
        import hashlib
        seen = set(); self._dup_row = np.zeros(len(df), bool)
        for i in range(len(df)):
            h = hashlib.md5("|".join(G[i]).encode()).hexdigest()
            if h in seen: self._dup_row[i] = True
            else: seen.add(h)
        Mn = M.to_numpy()
        for i, j in zip(*np.nonzero(Mn)):
            if self._dup_row[i]: continue
            for t in G[i, j].split(" "): tokc[(self.genes[j], t)] += 1
        self.tok_count = tokc
        self.n_mut_sorted = np.sort(M.sum(1).to_numpy())
        self.driver_genes = {g for g in GENE_INFO if g in set(self.genes)}
        self.dom = [(g, d, lo, hi) for g, ds in DOMAINS.items() if g in set(self.genes) for d, (lo, hi) in ds.items()]
        self._fit_index = df.index
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        G = df[self.genes].to_numpy(); n = len(df); gi = {g: k for k, g in enumerate(self.genes)}
        is_fit = self._fit_index is not None and df.index.equals(self._fit_index)
        loo_row = (lambda i: 0 if self._dup_row[i] else 1) if is_fit else (lambda i: 0)   # 학습 행이면 자기 자신 제외(중복 행은 이미 안 세었음)
        ti = {g: k for k, g in enumerate(self.type_genes)}
        T = np.zeros((n, len(self.type_genes), 3), np.uint8)             # mis, lof, syn
        di = {(g, d): k for k, (g, d, _, _) in enumerate(self.dom)}; D = np.zeros((n, len(self.dom)), np.uint8)
        dom_by_gene: dict[str, list] = {}
        for g, d, lo, hi in self.dom: dom_by_gene.setdefault(g, []).append((d, lo, hi))
        n_tok = np.zeros(n); n_fun = np.zeros(n); n_lof = np.zeros(n); n_syn = np.zeros(n); n_priv = np.zeros(n); n_single = np.zeros(n)
        n_genes = np.zeros(n); n_drv = np.zeros(n)
        for i, j in zip(*np.nonzero(G != "WT")):
            g = self.genes[j]; n_genes[i] += 1
            if g in self.driver_genes: n_drv[i] += 1
            for t in G[i, j].split(" "):
                k = _kind(t); n_tok[i] += 1
                if k != "syn": n_fun[i] += 1
                if k == "lof": n_lof[i] += 1
                if k == "syn": n_syn[i] += 1
                c = self.tok_count.get((g, t), 0) - loo_row(i)
                if c <= 0: n_priv[i] += 1
                elif c == 1: n_single[i] += 1
                if g in ti and k in ("mis", "lof", "syn"): T[i, ti[g], {"mis": 0, "lof": 1, "syn": 2}[k]] = 1
                if g in dom_by_gene and k != "syn":
                    p = _pos(t)
                    if p is not None:
                        for d, lo, hi in dom_by_gene[g]:
                            if lo <= p <= hi: D[i, di[(g, d)]] = 1
        cols = {}
        for g, k in ti.items():
            cols[f"mis_{g}"] = T[:, k, 0]; cols[f"lof_t_{g}"] = T[:, k, 1]; cols[f"syn_{g}"] = T[:, k, 2]
        for (g, d), k in di.items(): cols[f"dom_{g}_{d}"] = D[:, k]
        den = np.maximum(n_tok, 1)
        cols.update({"frac_functional": n_fun / den, "frac_lof": n_lof / den, "frac_syn": n_syn / den,
                     "frac_driver": n_drv / np.maximum(n_genes, 1), "n_private": n_priv, "n_singleton": n_single,
                     "frac_private": n_priv / den, "n_mut_rank": np.searchsorted(self.n_mut_sorted, n_genes) / max(len(self.n_mut_sorted), 1)})
        return pd.DataFrame(cols, index=df.index)
