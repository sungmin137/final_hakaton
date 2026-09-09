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


# ====================================================================== v3 — 인사이트 기반 피처
# 3. docs/03·04·07 분석에서 나온 것: (1) 변이 '위치'가 유전자보다 특이적, (2) 변이 개수·유형 비율이 강함,
# (3) 종양억제유전자는 종결/프레임시프트로 망가짐, (4) 동반 조합, (5) 초과변이/무변이 특수 그룹.
# fit이 필요한 것(어떤 hotspot·LoF 유전자를 쓸지)은 train 부분에서만 정한다.

# 프로필 분석에서 lift가 높았던 동반 조합 (3. docs/04)
COMBOS = [("IDH1", "TP53"), ("IDH1", "ATRX"), ("ATRX", "TP53"), ("APC", "TP53"), ("TP53", "CDKN2A"),
          ("TP53", "NOTCH1"), ("PTEN", "PIK3CA"), ("PTEN", "CTNNB1"), ("BRAF", "TP53"), ("VHL", "TP53"),
          ("TP53", "NFE2L2"), ("CDKN2A", "NOTCH1"), ("ERCC2", "FGFR3"), ("KIT", "NCOR2")]


def _is_lof(tok: str) -> bool:
    return tok.endswith("*") or "fs" in tok


class InsightFeatures:
    """v3 추가 피처. fit(train part) → transform(any)."""

    def __init__(self, hotspot_min: int = 5, lof_min: int = 5):
        self.hotspot_min, self.lof_min = hotspot_min, lof_min
        self.genes: list[str] = []
        self.hotspots: list[str] = []      # "GENE:variant"
        self.lof_genes: list[str] = []

    def fit(self, df: pd.DataFrame) -> "InsightFeatures":
        self.genes = gene_columns(df)
        G = df[self.genes].to_numpy()
        var_cnt: dict[str, int] = {}
        lof_cnt: dict[str, int] = {}
        for i, j in zip(*np.nonzero(G != "WT")):
            g = self.genes[j]
            seen = set()
            for tok in G[i, j].split(" "):
                key = f"{g}:{tok}"
                if key not in seen:
                    var_cnt[key] = var_cnt.get(key, 0) + 1; seen.add(key)
                if _is_lof(tok):
                    lof_cnt[g] = lof_cnt.get(g, 0) + 1
        self.hotspots = sorted(k for k, v in var_cnt.items() if v >= self.hotspot_min)
        self.lof_genes = sorted(g for g, v in lof_cnt.items() if v >= self.lof_min)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        G = df[self.genes].to_numpy()
        n = len(df)
        hs_idx = {k: i for i, k in enumerate(self.hotspots)}
        lof_idx = {g: i for i, g in enumerate(self.lof_genes)}
        H = np.zeros((n, len(self.hotspots)), np.uint8)
        L = np.zeros((n, len(self.lof_genes)), np.uint8)
        n_lof_genes = np.zeros(n, np.int32)
        for i, j in zip(*np.nonzero(G != "WT")):
            g = self.genes[j]; has_lof = False
            for tok in G[i, j].split(" "):
                k = hs_idx.get(f"{g}:{tok}")
                if k is not None:
                    H[i, k] = 1
                if _is_lof(tok):
                    has_lof = True
            if has_lof:
                n_lof_genes[i] += 1
                if g in lof_idx:
                    L[i, lof_idx[g]] = 1
        out = pd.DataFrame(H, columns=[f"hs_{k}" for k in self.hotspots], index=df.index)
        out = pd.concat([out, pd.DataFrame(L, columns=[f"lof_{g}" for g in self.lof_genes], index=df.index)], axis=1)

        mut = (df[self.genes] != "WT")
        n_mut = mut.sum(axis=1).to_numpy()
        gi = {g: i for i, g in enumerate(self.genes)}
        M = mut.to_numpy()
        for a, b in COMBOS:
            if a in gi and b in gi:
                out[f"combo_{a}_{b}"] = (M[:, gi[a]] & M[:, gi[b]]).astype(np.uint8)
        out["n_lof_genes"] = n_lof_genes
        out["lof_gene_ratio"] = n_lof_genes / np.maximum(n_mut, 1)
        out["is_zero_mut"] = (n_mut == 0).astype(np.uint8)
        out["is_hyper"] = (n_mut > 300).astype(np.uint8)
        out["n_mut_bucket"] = np.digitize(n_mut, [1, 11, 31, 101, 301])   # 0,1-10,11-30,31-100,101-300,>300
        out["n_hotspots"] = H.sum(1)
        return out
