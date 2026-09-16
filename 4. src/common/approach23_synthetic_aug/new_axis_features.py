"""approach23 v20: v11(v4s)에 추가로 검토하는 신규 정보축 후보 D(스펙트럼 집중도/엔트로피)와
E(hotspot까지의 연속 거리). 둘 다 train-only(fit 없거나 train fold로만 fit), 외부데이터 없음.
"""
import re
import numpy as np
import pandas as pd

_POS = re.compile(r"(\d+)")


# ---------------------------------------------------------------- 후보 D: 스펙트럼 집중도/엔트로피
def spectrum_concentration_features(sp_df: pd.DataFrame) -> pd.DataFrame:
    """spectrum_features()가 반환한 sp_ 380열(행별 합=1)에서 집중도/엔트로피 요약.
    fit 통계 없음 — 행 내부 계산만."""
    sp_cols = [c for c in sp_df.columns if c.startswith("sp_")]
    M = sp_df[sp_cols].to_numpy()
    eps = 1e-9
    entropy = -(M * np.log(M + eps)).sum(axis=1)
    sorted_desc = -np.sort(-M, axis=1)
    top1 = sorted_desc[:, 0]
    top3 = sorted_desc[:, :3].sum(axis=1)
    return pd.DataFrame({
        "spec_entropy": np.round(entropy, 4),
        "spec_top1_share": np.round(top1, 4),
        "spec_top3_share": np.round(top3, 4),
    }, index=sp_df.index)


# ---------------------------------------------------------------- 후보 E: hotspot까지의 연속 거리
_MIS = re.compile(r"^[A-Z](\d+)[A-Z]$")


def _positions_in_cell(cell: str):
    """한 셀(예: 'R132H', 'A1064V C401W')에서 아미노산 위치들을 뽑는다. WT면 빈 리스트."""
    if cell == "WT":
        return []
    out = []
    for tok in cell.split(" "):
        m = _MIS.match(tok)
        if m:
            out.append(int(m.group(1)))
        else:
            nums = _POS.findall(tok)
            if nums:
                out.append(int(nums[0]))
    return out


class HotspotDistance:
    """train(tr_part)에서만 유전자별 hotspot 위치(carrier>=MIN_CARRIERS)를 정하고,
    임의의 df에 대해 '이 샘플의 변이가 그 유전자의 hotspot들과 얼마나 가까운가'를 요약한다."""

    def __init__(self, min_carriers: int = 3, near_threshold: int = 10):
        self.min_carriers = min_carriers
        self.near_threshold = near_threshold
        self.hotspots: dict[str, list[int]] = {}

    def fit(self, df: pd.DataFrame, genes: list[str]) -> "HotspotDistance":
        for g in genes:
            cells = df.loc[df[g] != "WT", g]
            if len(cells) < self.min_carriers:
                continue
            pos_counter: dict[int, int] = {}
            for cell in cells:
                for p in set(_positions_in_cell(cell)):
                    pos_counter[p] = pos_counter.get(p, 0) + 1
            hs = [p for p, c in pos_counter.items() if c >= self.min_carriers]
            if hs:
                self.hotspots[g] = sorted(hs)
        return self

    def transform(self, df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
        n = len(df)
        min_dist = np.full(n, 999.0)
        n_near = np.zeros(n)
        n_mut_with_hotspot_gene = np.zeros(n)
        G = df[genes].to_numpy()
        gene_idx = {g: j for j, g in enumerate(genes) if g in self.hotspots}
        for gname, j in gene_idx.items():
            hs = np.array(self.hotspots[gname])
            col = G[:, j]
            mut_rows = np.flatnonzero(col != "WT")
            for i in mut_rows:
                positions = _positions_in_cell(col[i])
                if not positions:
                    continue
                n_mut_with_hotspot_gene[i] += 1
                d = min(min(abs(p - hs)) for p in positions)
                if d < min_dist[i]:
                    min_dist[i] = d
                if d <= self.near_threshold:
                    n_near[i] += 1
        return pd.DataFrame({
            "hs_min_dist": np.round(np.minimum(min_dist, 999), 1),
            "hs_n_near": n_near,
            "hs_n_genes_with_hotspot": n_mut_with_hotspot_gene,
        }, index=df.index)
