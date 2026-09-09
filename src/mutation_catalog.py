"""train.csv의 모든 변이를 (유전자, 변이) 단위로 카탈로그화한다. test.csv는 읽지 않는다.

각 변이에 대해:
  - type      : missense / nonsense(stop) / frameshift / synonymous / inframe_indel / other
  - lof       : 기능 상실(loss-of-function) 여부 = nonsense 또는 frameshift
  - count     : train에서 등장한 샘플 수
  - top_class : 가장 많이 등장한 SUBCLASS와 그 비율
  - hotspot   : count >= HOTSPOT_MIN 인 반복 변이 (driver 후보)
유전자 단위로도 집계해 oncogene형(반복 missense 집중) / tumor-suppressor형(LoF 비율 높음)을 구분한다.

실행: python3 src/mutation_catalog.py
출력: experiments/mutation_catalog/variants.csv, genes.csv
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "mutation_catalog"
HOTSPOT_MIN = 5


def mut_type(tok: str) -> str:
    if tok.endswith("*"):
        return "nonsense"
    if "fs" in tok:
        return "frameshift"
    if ">" in tok:
        return "inframe_indel"
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
    if m:
        return "synonymous" if m.group(1) == m.group(3) else "missense"
    return "other"


def main() -> None:
    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    genes = [c for c in tr.columns if c not in ("ID", "SUBCLASS")]
    labels = tr["SUBCLASS"].to_numpy()

    var_count: Counter = Counter()
    var_class: dict = defaultdict(Counter)
    for g in genes:
        col = tr[g].to_numpy()
        for i in (col != "WT").nonzero()[0]:
            for tok in col[i].split(" "):
                var_count[(g, tok)] += 1
                var_class[(g, tok)][labels[i]] += 1

    rows = []
    for (g, tok), n in var_count.items():
        cls = var_class[(g, tok)]
        top, top_n = cls.most_common(1)[0]
        t = mut_type(tok)
        rows.append(dict(gene=g, variant=tok, type=t, lof=t in ("nonsense", "frameshift"),
                         count=n, n_classes=len(cls), top_class=top,
                         top_share=round(top_n / n, 3), hotspot=n >= HOTSPOT_MIN))
    var = pd.DataFrame(rows).sort_values("count", ascending=False)

    # 유전자 단위 집계
    gdf = var.groupby("gene").agg(
        n_variants=("variant", "size"), n_samples=("count", "sum"),
        n_lof=("count", lambda s: s[var.loc[s.index, "lof"]].sum()),
        n_missense=("count", lambda s: s[var.loc[s.index, "type"] == "missense"].sum()),
        n_synonymous=("count", lambda s: s[var.loc[s.index, "type"] == "synonymous"].sum()),
        n_hotspots=("hotspot", "sum"),
        max_hotspot=("count", "max"),
    )
    gdf["lof_ratio"] = (gdf.n_lof / gdf.n_samples).round(3)
    gdf["hotspot_ratio"] = (gdf.max_hotspot / gdf.n_samples).round(3)  # 단일 변이 집중도

    def role(r):
        if r.n_samples < 20:
            return "low_evidence"
        if r.hotspot_ratio >= 0.15 and r.max_hotspot >= 10:
            return "oncogene_like"       # 같은 위치 반복 (BRAF V600E형)
        if r.lof_ratio >= 0.30:
            return "tumor_suppressor_like"  # 종결/프레임시프트로 망가짐 (TP53, APC형)
        return "passenger_like"
    gdf["role"] = gdf.apply(role, axis=1)
    gdf = gdf.sort_values("n_samples", ascending=False)

    OUT.mkdir(parents=True, exist_ok=True)
    var.to_csv(OUT / "variants.csv", index=False)
    gdf.to_csv(OUT / "genes.csv")

    print("고유 변이 수", len(var), "| 유형별 개수(변이 단위):")
    print(var.type.value_counts().to_string())
    print("\n[반복 변이(hotspot, count>=5) 상위 25]")
    print(var[var.hotspot].head(25)[["gene", "variant", "type", "count", "n_classes", "top_class", "top_share"]].to_string(index=False))
    print("\n[유전자 역할 분류]\n", gdf.role.value_counts().to_string())
    for r in ("oncogene_like", "tumor_suppressor_like"):
        print(f"\n[{r} 상위 15]")
        print(gdf[gdf.role == r].head(15)[["n_samples", "n_lof", "n_missense", "lof_ratio", "max_hotspot", "hotspot_ratio"]].to_string())
    print("\n[nonsense(stop) 변이 상위 10 — 어떤 유전자가 잘리는가]")
    print(var[var.type == "nonsense"].head(10)[["gene", "variant", "count", "top_class", "top_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
