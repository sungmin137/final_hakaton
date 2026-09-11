"""v3.5 = v3(팀장님 sungmin 브랜치, features_v3_reference.py) + 유전자 역할 오버라이드.

배경: mutation_catalog.py의 role() 규칙(hotspot_ratio>=0.15 / lof_ratio>=0.30)이
TP53, NOTCH1, CTNNB1, KIT, EGFR, MET 6개를 전부 'passenger_like'로 잘못 분류함.
(known_gene_roles.py의 diff_with_catalog()로 확인, day1/todo_day1.md 참고)

이 파일은 팀 저장소 코드를 수정하지 않고, 그 위에 얹는 새로운 피처만 추가한다.
build_features, InsightFeatures는 features_v3_reference.py(팀 코드 그대로 복사, 미수정)를 그대로 import.
"""
import numpy as np
import pandas as pd

from features_v3_reference import ID, TARGET, gene_columns, build_features, InsightFeatures, _is_lof

# known_gene_roles.py에서 확인한 오버라이드 대상 6개 + 역할
GENE_OVERRIDES = {
    "TP53":   "dual",
    "NOTCH1": "dual",
    "CTNNB1": "oncogene",
    "KIT":    "oncogene",
    "EGFR":   "oncogene",
    "MET":    "oncogene",
}


def build_override_features(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """오버라이드 대상 6유전자마다 ovr_<gene>_missense만 생성.

    2026-09-10 실측(day1/check_importance.py, feature_importances_)으로 확인됨:
    - ovr_<gene>_any, ovr_<gene>_lof는 기존 v1의 g_<gene>(그 유전자 변이 유무)과 완전히
      중복된 정보라 XGB가 거의 안 씀(순위 3,000위대/6,566개, 중요도≈0). 그래서 삭제함.
    - ovr_<gene>_missense만 실제로 채택됨(ovr_EGFR_missense 19위, ovr_CTNNB1_missense 17위,
      ovr_TP53_missense 172위) — g_<gene>에는 없던 "missense 여부"라는 새 정보라서 살아남음.
    """
    G = df[genes].to_numpy()
    gi = {g: i for i, g in enumerate(genes)}
    n = len(df)
    out = {}
    for gene in GENE_OVERRIDES:
        if gene not in gi:
            continue
        j = gi[gene]
        col = G[:, j]
        is_mut = (col != "WT")
        has_mis = np.zeros(n, dtype=np.uint8)
        for i in np.nonzero(is_mut)[0]:
            for tok in col[i].split(" "):
                if not _is_lof(tok):
                    has_mis[i] = 1
        out[f"ovr_{gene}_missense"] = has_mis
    return pd.DataFrame(out, index=df.index)


def build_features_v35(df: pd.DataFrame, genes: list[str], insight: InsightFeatures) -> pd.DataFrame:
    """v1(build_features) + v3(InsightFeatures.transform) + 오버라이드(6유전자)."""
    v1 = build_features(df, genes)
    v3 = insight.transform(df)
    ovr = build_override_features(df, genes)
    return pd.concat([v1, v3, ovr], axis=1)
