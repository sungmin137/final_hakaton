"""도메인 지식으로 검증한 유전자 역할 참조 테이블 (수동 큐레이션, 외부 데이터 아님).

mutation_catalog.py의 role()은 이 대회 train 데이터의 통계(hotspot_ratio, lof_ratio)만으로
oncogene_like / tumor_suppressor_like / passenger_like를 자동 분류한다.
이 테이블은 그 자동 분류가 실제 암유전학 지식과 맞는지 대조하기 위한 "정답지" 역할이다.

role:
  oncogene           = 특정 위치(hotspot) 활성화 변이로 암 유발 (스위치 고정형)
  tumor_suppressor   = 기능 상실(LoF)로 암 유발 (브레이크 소실형)
  dual               = 위 두 성질을 상황에 따라 다 가짐 → 자동 분류 규칙이 잘못 판단하기 쉬움
                       → 수동 오버라이드 후보
confidence: high = 교과서/WHO 분류 기준급으로 확립된 지식
"""

KNOWN_GENE_ROLES = {
    "TP53":   dict(role="dual",             confidence="high", note="게놈의 수호자, 거의 모든 암종에서 흔함. LoF+missense hotspot 둘 다 driver"),
    "NOTCH1": dict(role="dual",             confidence="high", note="HNSC에서는 종양억제유전자(LoF)로 작용, 다른 맥락에선 종양유전자로도 작용"),
    "IDH1":   dict(role="oncogene",         confidence="high", note="R132H: WHO 뇌종양 분류 기준. GBMLGG/LGG"),
    "IDH2":   dict(role="oncogene",         confidence="high", note="R140Q: AML(LAML) 대표 hotspot"),
    "BRAF":   dict(role="oncogene",         confidence="high", note="V600E: THCA/SKCM/COAD, 표적치료제 처방 기준"),
    "HRAS":   dict(role="oncogene",         confidence="medium", note="Q61R 활성화 변이"),
    "PIK3CA": dict(role="oncogene",         confidence="high", note="E545K/H1047R: BRCA/UCEC/HNSC/CESC"),
    "CTNNB1": dict(role="oncogene",         confidence="high", note="LIHC/UCEC hotspot (exon3 안정화 변이)"),
    "KIT":    dict(role="oncogene",         confidence="high", note="TGCT(정상피종) 대표 드라이버"),
    "APC":    dict(role="tumor_suppressor", confidence="high", note="COAD 대표, Wnt 경로"),
    "VHL":    dict(role="tumor_suppressor", confidence="high", note="KIRC/KIPAN 대표"),
    "PTEN":   dict(role="tumor_suppressor", confidence="high", note="UCEC/GBMLGG, PI3K/AKT 경로 억제자"),
    "RB1":    dict(role="tumor_suppressor", confidence="high", note="세포주기 체크포인트"),
    "CDKN2A": dict(role="tumor_suppressor", confidence="high", note="HNSC/PAAD, CDK4/6 억제자"),
    "NF1":    dict(role="tumor_suppressor", confidence="medium", note="RAS 경로 억제자, 긴 유전자라 passenger 혼입 주의"),
    "ATRX":   dict(role="tumor_suppressor", confidence="high", note="IDH1+ATRX 동반 = 성상세포종 아형(WHO 기준)"),
    "CDH1":   dict(role="tumor_suppressor", confidence="high", note="BRCA(소엽성)"),
    "GATA3":  dict(role="tumor_suppressor", confidence="medium", note="BRCA 특이적 반복 변이(전사인자, 고전적 driver는 아님)"),
    "SPOP":   dict(role="other_driver",     confidence="high", note="PRAD 분자 아형 정의"),
    "NPM1":   dict(role="other_driver",     confidence="high", note="AML(LAML) 진단 30%, frameshift로 세포질 오위치"),
    "EGFR":   dict(role="oncogene",         confidence="high", note="원발 GBM 대표 드라이버"),
    "MET":    dict(role="oncogene",         confidence="high", note="유두형신세포암(KIRP, KIPAN 하위) 특이 드라이버"),
}

# ⚠️ 통계는 강하지만 생물학적 근거가 의심되는 케이스 (docs/03 참고)
SUSPECTED_ARTIFACTS = {
    "ACC 전용 반복 변이(LRIG1 L24V, SOWAHC 등)": "germline 다형성/배치 아티팩트 가능성. "
        "통계상 ACC를 거의 100% 맞히지만 실제 암 발생 기전과 무관할 수 있음. "
        "가중치는 주되, 대회 발표 시 이 한계를 설명할 것.",
}


def diff_with_catalog(genes_csv_path: str):
    """mutation_catalog.py가 만든 genes.csv(자동 분류)와 이 수동 테이블을 대조해서
    불일치하는 유전자만 뽑아낸다. 이게 곧 '오버라이드 후보' 목록이다."""
    import pandas as pd
    auto = pd.read_csv(genes_csv_path, index_col=0)  # index: gene, column: role 등

    rows = []
    for gene, known in KNOWN_GENE_ROLES.items():
        if gene not in auto.index:
            rows.append(dict(gene=gene, known_role=known["role"], auto_role="(패널에 없음)",
                              mismatch=None, confidence=known["confidence"]))
            continue
        auto_role = auto.loc[gene, "role"]
        # dual로 알려진 유전자가 자동으로는 oncogene_like/tumor_suppressor_like/passenger_like 중
        # 하나로만 단정됐다면 그게 바로 오버라이드가 필요한 지점
        mismatch = (known["role"] == "dual" and auto_role != "dual") or \
                   (known["role"] == "oncogene" and auto_role != "oncogene_like") or \
                   (known["role"] == "tumor_suppressor" and auto_role != "tumor_suppressor_like")
        rows.append(dict(gene=gene, known_role=known["role"], auto_role=auto_role,
                          mismatch=mismatch, confidence=known["confidence"]))
    return pd.DataFrame(rows).sort_values(["mismatch", "confidence"], ascending=[False, False])


if __name__ == "__main__":
    # 실행 예: PYTHONPATH=. python3 known_gene_roles.py
    # (mutation_catalog.py를 먼저 돌려서 experiments/mutation_catalog/genes.csv를 만들어둔 뒤 실행)
    df = diff_with_catalog("experiments/mutation_catalog/genes.csv")
    print(df.to_string(index=False))
    print(f"\n오버라이드 후보(mismatch=True): {df[df.mismatch == True]['gene'].tolist()}")
