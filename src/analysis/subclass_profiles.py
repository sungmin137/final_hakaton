"""SUBCLASS별 변이 프로필. train.csv만 사용.

클래스마다:
  1) 특이 유전자   : 클래스 내 변이율 vs 나머지 변이율 (lift), Fisher 정확검정 p값
  2) 특이 변이     : (유전자, 변이) 단위 클래스 내 빈도와 클래스 집중도
  3) 동반 변이 조합: 특이 유전자들 사이의 동시 변이 빈도 (클래스 내 vs 배경)
  4) 유형 프로필   : 샘플당 변이수, LoF 비율, 동의 변이 비율, 초과변이 비율, 무변이 비율
  5) 표지 커버리지 : 상위 표지 유전자 중 하나라도 변이가 있는 샘플 비율

실행: python3 src/analysis/subclass_profiles.py
출력: experiments/subclass_profiles/*.csv, docs/04_subclass_profiles.md
"""
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "subclass_profiles"
DOC = ROOT / "docs" / "04_subclass_profiles.md"

TOP_GENES, TOP_VARS, TOP_PAIRS = 12, 10, 8
MIN_RATE, MIN_COUNT = 0.05, 4


def mut_type(tok: str) -> str:
    if tok.endswith("*"):
        return "nonsense"
    if "fs" in tok:
        return "frameshift"
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
    if m:
        return "synonymous" if m.group(1) == m.group(3) else "missense"
    return "other"


def main() -> None:
    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    genes = [c for c in tr.columns if c not in ("ID", "SUBCLASS")]
    y = tr["SUBCLASS"].to_numpy()
    M = (tr[genes] != "WT").to_numpy(dtype=np.uint8)          # any mutation
    n_mut = M.sum(1)

    # 셀 단위 변이 유형 카운트 + (gene, variant) 카운트
    lof = np.zeros(len(tr), int); syn = np.zeros(len(tr), int); tot = np.zeros(len(tr), int)
    var_cls: dict = defaultdict(Counter); var_tot: Counter = Counter()
    G = tr[genes].to_numpy()
    for i, j in zip(*np.nonzero(M)):
        for tok in G[i, j].split(" "):
            t = mut_type(tok); tot[i] += 1
            lof[i] += t in ("nonsense", "frameshift"); syn[i] += t == "synonymous"
            var_cls[(genes[j], tok)][y[i]] += 1; var_tot[(genes[j], tok)] += 1

    classes = sorted(set(y))
    gene_rows, var_rows, pair_rows, prof_rows, md = [], [], [], [], []
    md.append("# SUBCLASS별 변이 프로필 (train 6,201명, test 미사용)\n")
    md.append("각 클래스에서 **다른 클래스 대비 유독 많이 변이되는 유전자**(lift = 클래스 내 변이율 ÷ 나머지 변이율), "
              "구체적 변이 위치, 함께 나타나는 유전자 조합을 정리했다. p값은 Fisher 정확검정.\n")
    md.append("## 한눈에 보기\n\n| 클래스 | n | 변이수 중앙값 | LoF% | 동의% | 무변이% | 초과변이% | 핵심 표지 (변이율) | 표지 커버리지 |\n|---|---|---|---|---|---|---|---|---|")
    summary_rows = []

    for c in classes:
        idx = y == c; n_c = idx.sum(); Mc, Mo = M[idx], M[~idx]
        r_in, r_out = Mc.mean(0), Mo.mean(0)
        cnt_in = Mc.sum(0)
        # 1) 특이 유전자
        cand = np.where((r_in >= MIN_RATE) & (cnt_in >= MIN_COUNT))[0]
        rows = []
        for j in cand:
            a, b = int(cnt_in[j]), int(n_c - cnt_in[j]); cc, d = int(Mo[:, j].sum()), int((~idx).sum() - Mo[:, j].sum())
            _, p = fisher_exact([[a, b], [cc, d]], alternative="greater")
            rows.append(dict(subclass=c, gene=genes[j], rate_in=r_in[j], rate_out=r_out[j],
                             lift=(r_in[j] + 1e-9) / (r_out[j] + 1e-9), n_in=a, p=p))
        gdf = pd.DataFrame(rows)
        gdf["score"] = -np.log10(gdf.p.clip(1e-300)) * np.log2(gdf.lift.clip(1))
        gdf = gdf.sort_values("score", ascending=False).head(TOP_GENES)
        gene_rows.append(gdf)
        top_genes = gdf.gene.tolist()

        # 2) 특이 변이
        vr = []
        for (g, tok), cl in var_cls.items():
            k = cl.get(c, 0)
            if k >= MIN_COUNT:
                vr.append(dict(subclass=c, gene=g, variant=tok, type=mut_type(tok), n_in=k,
                               rate_in=k / n_c, share=k / var_tot[(g, tok)]))
        vdf = pd.DataFrame(vr)
        if len(vdf):
            vdf["score"] = vdf.n_in * vdf.share
            vdf = vdf.sort_values("score", ascending=False).head(TOP_VARS)
        var_rows.append(vdf)

        # 3) 동반 변이 조합 (특이 유전자 상위 8개 사이)
        pr = []
        gi = {g: genes.index(g) for g in top_genes[:8]}
        for g1, g2 in combinations(gi, 2):
            both_in = (Mc[:, gi[g1]] & Mc[:, gi[g2]]).mean()
            both_out = (Mo[:, gi[g1]] & Mo[:, gi[g2]]).mean()
            if both_in >= 0.05:
                pr.append(dict(subclass=c, pair=f"{g1}+{g2}", rate_in=both_in, rate_out=both_out,
                               lift=(both_in + 1e-9) / (both_out + 1e-9)))
        pdf = pd.DataFrame(pr).sort_values("rate_in", ascending=False).head(TOP_PAIRS) if pr else pd.DataFrame()
        pair_rows.append(pdf)

        # 4) 유형 프로필 / 5) 커버리지
        t_c = tot[idx].clip(min=1)
        cov_genes = top_genes[:3]
        coverage = Mc[:, [genes.index(g) for g in cov_genes]].any(1).mean() if cov_genes else 0
        prof = dict(subclass=c, n=int(n_c), median_mut=float(np.median(n_mut[idx])),
                    lof_pct=100 * (lof[idx] / t_c).mean(), syn_pct=100 * (syn[idx] / t_c).mean(),
                    zero_pct=100 * (n_mut[idx] == 0).mean(), hyper_pct=100 * (n_mut[idx] > 300).mean(),
                    markers=", ".join(f"{g} {r:.0%}" for g, r in zip(gdf.gene.head(3), gdf.rate_in.head(3))),
                    coverage=coverage)
        prof_rows.append(prof)
        summary_rows.append(f"| {c} | {n_c} | {prof['median_mut']:.0f} | {prof['lof_pct']:.0f} | {prof['syn_pct']:.0f} | "
                            f"{prof['zero_pct']:.0f} | {prof['hyper_pct']:.0f} | {prof['markers']} | {coverage:.0%} |")

    md.extend(summary_rows)
    md.append("\n> LoF% = 샘플당 변이 중 종결/프레임시프트 비율 평균. 표지 커버리지 = 상위 3개 표지 유전자 중 하나라도 변이가 있는 샘플 비율.\n")

    # 클래스별 상세
    for c, gdf, vdf, pdf, prof in zip(classes, gene_rows, var_rows, pair_rows, prof_rows):
        md.append(f"\n---\n## {c}  (n={prof['n']}, 변이수 중앙값 {prof['median_mut']:.0f}, 무변이 {prof['zero_pct']:.0f}%, 초과변이 {prof['hyper_pct']:.0f}%)\n")
        md.append("**특이 유전자** (클래스 내 변이율 / 나머지 변이율 → lift)\n\n| 유전자 | 클래스 내 | 나머지 | lift | p |\n|---|---|---|---|---|")
        for r in gdf.itertuples():
            md.append(f"| {r.gene} | {r.rate_in:.0%} ({r.n_in}) | {r.rate_out:.1%} | {r.lift:.1f}× | {r.p:.1e} |")
        md.append("\n**특이 변이 위치** (클래스 내 인원, 이 변이 전체 중 이 클래스 비율)\n\n| 유전자 | 변이 | 유형 | 인원 | 클래스 내 비율 | 클래스 집중도 |\n|---|---|---|---|---|---|")
        for r in vdf.itertuples():
            md.append(f"| {r.gene} | {r.variant} | {r.type} | {r.n_in} | {r.rate_in:.0%} | {r.share:.0%} |")
        if len(pdf):
            md.append("\n**동반 변이 조합** (두 유전자가 함께 변이된 샘플 비율)\n\n| 조합 | 클래스 내 | 나머지 | lift |\n|---|---|---|---|")
            for r in pdf.itertuples():
                md.append(f"| {r.pair} | {r.rate_in:.0%} | {r.rate_out:.1%} | {r.lift:.0f}× |")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(gene_rows).to_csv(OUT / "genes.csv", index=False)
    pd.concat(var_rows).to_csv(OUT / "variants.csv", index=False)
    pd.concat([p for p in pair_rows if len(p)]).to_csv(OUT / "pairs.csv", index=False)
    pd.DataFrame(prof_rows).to_csv(OUT / "profile.csv", index=False)
    DOC.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(summary_rows))
    print("\nwritten", DOC)


if __name__ == "__main__":
    main()
