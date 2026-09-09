"""접근 4 — 문헌 지식 vs train 데이터 대조 분석. train.csv만 사용, test는 결측 '관찰'만(모델 조정 없음).
산출: 6. experiments/approach4_literature/*.md (문서에 삽입할 표), *.csv
실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach4_literature/literature_analysis.py"
"""
import re
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from main import ROOT, load_train
from approach4_literature.literature_map import GENE_INFO, CLASS_DRIVERS, PATHWAYS, hotspot_match

OUT = ROOT / "6. experiments" / "approach4_literature"; OUT.mkdir(parents=True, exist_ok=True)


def mut_type(t):
    if t.endswith("*"): return "nonsense"
    if "fs" in t: return "frameshift"
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", t)
    return ("synonymous" if m.group(1) == m.group(3) else "missense") if m else "other"


def main():
    tr = load_train(); genes = [c for c in tr.columns if c not in ("ID", "SUBCLASS")]; gset = set(genes)
    y = tr.SUBCLASS.to_numpy(); classes = sorted(set(y)); n_c = pd.Series(y).value_counts()
    M = (tr[genes] != "WT"); rate = M.groupby(y).mean() * 100
    G = tr[genes].to_numpy()
    # 셀 단위: 기능성 변이·hotspot·LoF 집계
    func = np.zeros(M.shape, bool); hs_hit = defaultdict(Counter); lof_cnt = defaultdict(Counter); tok_cnt = defaultdict(Counter)
    for i, j in zip(*np.nonzero(M.to_numpy())):
        g = genes[j]
        for t in G[i, j].split(" "):
            ty = mut_type(t)
            if ty != "synonymous": func[i, j] = True
            if ty in ("nonsense", "frameshift"): lof_cnt[g][y[i]] += 1
            h = hotspot_match(g, t)
            if h: hs_hit[g][(y[i], h)] += 1
            if ty != "synonymous": tok_cnt[g][t] += 1
    frate = pd.DataFrame(func, columns=genes).groupby(y).mean() * 100

    # 1) 암종별 문헌 driver vs 관측 (패널 내) + 패널에 없는 driver
    rows, missing_rows = [], []
    for c, drv in CLASS_DRIVERS.items():
        for g, lit in drv.items():
            if g in gset:
                info = GENE_INFO.get(g, {}); obs = float(rate.loc[c, g]); obsf = float(frate.loc[c, g])
                hs = hs_hit[g]; hs_c = sum(v for (cc, h), v in hs.items() if cc == c); hs_all = sum(hs.values())
                lof_c = lof_cnt[g][c]; n = int(n_c[c])
                rows.append(dict(cls=c, gene=g, role=info.get("role", "?"), pathway=info.get("pathway", "?"), lit_pct=lit, obs_pct=round(obs, 1),
                                 obs_func_pct=round(obsf, 1), n_mut=int(M[g][y == c].sum()), hotspot_n=hs_c, hotspot_share=round(hs_c / hs_all, 2) if hs_all else None,
                                 lof_n=lof_c, lof_ratio=round(lof_c / max(int(M[g][y == c].sum()), 1), 2),
                                 diff=None if lit is None else round(obs - lit, 1)))
            else:
                missing_rows.append(dict(cls=c, gene=g, lit_pct=lit, role=GENE_INFO.get(g, {}).get("role", "?"), pathway=GENE_INFO.get(g, {}).get("pathway", "?")))
    lit_df = pd.DataFrame(rows); lit_df.to_csv(OUT / "literature_vs_data.csv", index=False)
    miss_df = pd.DataFrame(missing_rows); miss_df.to_csv(OUT / "drivers_missing_from_panel.csv", index=False)

    # 2) 경로별 변이율 (암종 × 경로): 경로 내 유전자 중 하나라도 기능성 변이
    pw_genes = {p: [g for g, v in GENE_INFO.items() if v["pathway"] == p and g in gset] for p in PATHWAYS}
    pw = pd.DataFrame({p: pd.DataFrame(func[:, [genes.index(g) for g in gl]]).any(axis=1).groupby(y).mean() * 100 for p, gl in pw_genes.items() if gl}).round(0)
    pw.to_csv(OUT / "pathway_by_class.csv")

    # 3) 문헌 hotspot 관측 표 (유전자·위치별 상위 암종)
    hs_rows = []
    for g, cnt in hs_hit.items():
        by_h = defaultdict(Counter)
        for (c, h), v in cnt.items(): by_h[h][c] += v
        for h, cc in by_h.items():
            tot = sum(cc.values()); top = cc.most_common(3)
            hs_rows.append(dict(gene=g, hotspot=h, n=tot, top=", ".join(f"{c} {v}" for c, v in top), top_share=round(top[0][1] / tot, 2)))
    hs_df = pd.DataFrame(hs_rows).sort_values("n", ascending=False); hs_df.to_csv(OUT / "hotspots_observed.csv", index=False)

    # 4) 데이터에서만 보이는 상위 유전자 (문헌 driver 목록에 없는데 특정 암종에 집중)
    novel = []
    for c in classes:
        r_in = frate.loc[c]; r_out = pd.DataFrame(func, columns=genes)[y != c].mean() * 100
        lift = (r_in + 0.5) / (r_out + 0.5); score = r_in * np.log2(lift.clip(1))
        for g in score.sort_values(ascending=False).head(6).index:
            if g not in CLASS_DRIVERS.get(c, {}) and r_in[g] >= 5:
                novel.append(dict(cls=c, gene=g, obs_func_pct=round(float(r_in[g]), 1), others_pct=round(float(r_out[g]), 1), lift=round(float(lift[g]), 1), in_literature_any=g in GENE_INFO))
    nov_df = pd.DataFrame(novel); nov_df.to_csv(OUT / "data_only_markers.csv", index=False)

    # 5) 결측 분석 — train: 결측 없음, 전부 WT 유전자, 변이 0 행 / test: 관찰만
    allwt = [g for g in genes if M[g].sum() == 0]
    zero_rows = (M.sum(axis=1) == 0)
    te = pd.read_csv(ROOT / "1. info/data/test.csv"); tgenes = [c for c in te.columns if c != "ID"]
    na = te[tgenes].isna(); na_cols = na.sum()[na.sum() > 0].sort_values(ascending=False)
    na_info = pd.DataFrame({"test_na_cells": na_cols, "train_mut_pct": [round(float(M[g].mean() * 100), 2) for g in na_cols.index],
                            "literature_driver": [g in GENE_INFO for g in na_cols.index],
                            "train_top_class": [f"{rate[g].idxmax()} {rate[g].max():.0f}%" for g in na_cols.index]})
    na_info.to_csv(OUT / "test_missing_columns.csv")
    missing_md = {"train_na": int(tr.isna().values.sum()), "allwt_genes": len(allwt), "allwt_lit": [g for g in allwt if g in GENE_INFO],
                  "zero_rows": int(zero_rows.sum()), "zero_rows_cls": pd.Series(y[zero_rows]).value_counts().head(5).to_dict(),
                  "test_na_cells": int(na.values.sum()), "test_na_rows": int(na.any(axis=1).sum()), "test_na_cols": int((na.sum() > 0).sum()),
                  "test_na_rows_max": int(na.sum(axis=1).max())}

    # ---- 요약 출력
    print("== 암종별 문헌 driver 커버리지 ==")
    cov = lit_df.groupby("cls").agg(in_panel=("gene", "size")).join(miss_df.groupby("cls").agg(missing=("gene", "size"))).fillna(0).astype(int)
    cov["missing_lit_pct_sum"] = miss_df.groupby("cls").lit_pct.sum().round(0); print(cov.to_string())
    print("\n== 문헌 빈도 대비 관측 차이 큰 항목 (|diff|>=15) ==")
    print(lit_df.dropna(subset=["diff"]).loc[lambda d: d["diff"].abs() >= 15].sort_values("diff")[["cls", "gene", "lit_pct", "obs_pct", "obs_func_pct", "diff"]].to_string(index=False))
    print("\n== 관측된 문헌 hotspot 상위 15 =="); print(hs_df.head(15).to_string(index=False))
    print("\n== 데이터에서만 보이는 표지 =="); print(nov_df.to_string(index=False))
    print("\n== 결측 ==", missing_md); print(na_info.head(12).to_string())
    print("\n== 경로별 기능성 변이율 (일부) =="); print(pw[["RTK-RAS", "PI3K", "TP53", "WNT", "Chromatin", "IDH", "NRF2", "Heme"]].astype(int).to_string())


if __name__ == "__main__":
    main()
