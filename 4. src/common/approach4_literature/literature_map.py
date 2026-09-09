"""접근 4 — 문헌 기반 암종별 변이 지식 표. 환자 단위 데이터는 포함하지 않는다 (규칙: 논문·교과서 지식만).

출처(요약):
- TCGA 암종별 marker paper (BRCA 2012, COAD 2012, LUAD 2014, LUSC 2012, HNSC 2015, KIRC 2013, KIRP 2016, LGG 2015, GBM 2013,
  UCEC 2013, BLCA 2014/2017, STAD 2014, ESCA 2017, OV 2011, PAAD 2017, PRAD 2015, SKCM 2015, THCA 2014, LAML 2013, LIHC 2017,
  CESC 2017, SARC 2017, TGCT 2018, THYM 2018, PCPG 2017, ACC 2016, DLBC: Reddy 2017 Cell / Chapuy 2018 Nat Med)
- Bailey et al. 2018 Cell, "Comprehensive Characterization of Cancer Driver Genes and Mutations" (암종별 driver 목록)
- Vogelstein et al. 2013 Science, "Cancer Genome Landscapes" (oncogene / tumor suppressor 20/20 규칙)
- Sanchez-Vega et al. 2018 Cell, "Oncogenic Signaling Pathways in TCGA" (10개 경로 유전자 집합)
- Chang et al. 2016 Nat Biotechnol / 2018 Cancer Discov (hotspot 변이 목록)

ROLE : "OG" oncogene(활성화 missense hotspot), "TSG" tumor suppressor(절단·LoF), "OG/TSG" 양면
HOTSPOT: 문헌상 대표 위치 (변이 문자열 접두 일치로 매칭, 예: "R132" ← R132H/R132C)
"""
from __future__ import annotations

# ---------------------------------------------------------------- 유전자 역할·경로·hotspot (전체 암종 공통 지식)
GENE_INFO: dict[str, dict] = {
    # RTK-RAS
    "EGFR": dict(role="OG", pathway="RTK-RAS", hotspots=["L858", "T790", "A289", "G598", "R108", "L861"]),
    "ERBB2": dict(role="OG", pathway="RTK-RAS", hotspots=["S310", "L755", "V777", "V842"]),
    "ERBB3": dict(role="OG", pathway="RTK-RAS", hotspots=["V104", "E928"]),
    "MET": dict(role="OG", pathway="RTK-RAS", hotspots=["D1010", "Y1003", "H1094"]),
    "KIT": dict(role="OG", pathway="RTK-RAS", hotspots=["D816", "W557", "V559", "K642", "N822"]),
    "RET": dict(role="OG", pathway="RTK-RAS", hotspots=["M918", "C634"]),
    "FGFR3": dict(role="OG", pathway="RTK-RAS", hotspots=["S249", "R248", "Y373", "G370"]),
    "FGFR1": dict(role="OG", pathway="RTK-RAS", hotspots=["N546"]),
    "HRAS": dict(role="OG", pathway="RTK-RAS", hotspots=["Q61", "G12", "G13"]),
    "MRAS": dict(role="OG", pathway="RTK-RAS", hotspots=["Q71"]),
    "BRAF": dict(role="OG", pathway="RTK-RAS", hotspots=["V600", "K601", "G469", "D594", "G466"]),
    "MAP2K1": dict(role="OG", pathway="RTK-RAS", hotspots=["K57", "P124", "E203"]),
    "MAPK1": dict(role="OG", pathway="RTK-RAS", hotspots=["E322"]),
    "NF1": dict(role="TSG", pathway="RTK-RAS", hotspots=[]),
    "RASA1": dict(role="TSG", pathway="RTK-RAS", hotspots=[]),
    "RIT1": dict(role="OG", pathway="RTK-RAS", hotspots=["M90", "A57"]),
    "PTPN11": dict(role="OG", pathway="RTK-RAS", hotspots=["E76", "A72", "D61"]),
    "RAC1": dict(role="OG", pathway="RTK-RAS", hotspots=["P29"]),
    "SOS1": dict(role="OG", pathway="RTK-RAS", hotspots=[]),
    "CBL": dict(role="TSG", pathway="RTK-RAS", hotspots=["Y371"]),
    # PI3K
    "PIK3CA": dict(role="OG", pathway="PI3K", hotspots=["E542", "E545", "H1047", "Q546", "N345", "C420", "E453", "R88", "G118"]),
    "PIK3R1": dict(role="TSG", pathway="PI3K", hotspots=["R348", "K567", "N564", "D560"]),
    "PTEN": dict(role="TSG", pathway="PI3K", hotspots=["R130", "R233", "R173", "R335"]),
    "AKT1": dict(role="OG", pathway="PI3K", hotspots=["E17"]),
    "MTOR": dict(role="OG", pathway="PI3K", hotspots=["S2215", "L2427", "F1888", "E1799", "T1977"]),
    "TSC1": dict(role="TSG", pathway="PI3K", hotspots=[]), "TSC2": dict(role="TSG", pathway="PI3K", hotspots=[]),
    "RPS6KA3": dict(role="TSG", pathway="PI3K", hotspots=[]),
    # TP53 / cell cycle
    "TP53": dict(role="TSG", pathway="TP53", hotspots=["R175", "R248", "R273", "R282", "G245", "R249", "Y220", "R213", "R196", "R306", "R342", "H179", "C176", "V157"]),
    "ATM": dict(role="TSG", pathway="TP53", hotspots=[]), "CHEK2": dict(role="TSG", pathway="TP53", hotspots=["K373"]),
    "MDM2": dict(role="OG", pathway="TP53", hotspots=[]), "MDM4": dict(role="OG", pathway="TP53", hotspots=[]),
    "PPM1D": dict(role="OG", pathway="TP53", hotspots=[]),
    "CDKN2A": dict(role="TSG", pathway="CellCycle", hotspots=["R80", "R58", "W110", "H83", "D84", "E88"]),
    "CDKN1A": dict(role="TSG", pathway="CellCycle", hotspots=[]), "CDKN1B": dict(role="TSG", pathway="CellCycle", hotspots=[]),
    "RB1": dict(role="TSG", pathway="CellCycle", hotspots=[]), "CDK4": dict(role="OG", pathway="CellCycle", hotspots=["R24"]),
    "CDK12": dict(role="TSG", pathway="CellCycle", hotspots=[]), "CCND1": dict(role="OG", pathway="CellCycle", hotspots=[]),
    # WNT
    "APC": dict(role="TSG", pathway="WNT", hotspots=["R1450", "R876", "R1114", "E1309", "R213", "R232", "Q1378", "R1432", "R554", "S1465", "E1554"]),
    "CTNNB1": dict(role="OG", pathway="WNT", hotspots=["S45", "S37", "S33", "D32", "G34", "T41", "K335", "N387"]),
    "AXIN1": dict(role="TSG", pathway="WNT", hotspots=[]), "AXIN2": dict(role="TSG", pathway="WNT", hotspots=[]),
    "ZNRF3": dict(role="TSG", pathway="WNT", hotspots=[]), "TCF7L2": dict(role="TSG", pathway="WNT", hotspots=[]),
    "AMER1": dict(role="TSG", pathway="WNT", hotspots=[]), "GSK3B": dict(role="TSG", pathway="WNT", hotspots=[]),
    "SOX9": dict(role="TSG", pathway="WNT", hotspots=[]),
    # NOTCH
    "NOTCH1": dict(role="OG/TSG", pathway="NOTCH", hotspots=[]), "NOTCH2": dict(role="OG/TSG", pathway="NOTCH", hotspots=[]),
    "NOTCH3": dict(role="OG/TSG", pathway="NOTCH", hotspots=[]),
    "FBXW7": dict(role="TSG", pathway="NOTCH", hotspots=["R465", "R479", "R505", "R278", "S582"]),
    "CREBBP": dict(role="TSG", pathway="NOTCH", hotspots=["R1446", "Y1503", "R1173"]), "EP300": dict(role="TSG", pathway="NOTCH", hotspots=["D1399"]),
    "SPEN": dict(role="TSG", pathway="NOTCH", hotspots=[]),
    # Hippo
    "NF2": dict(role="TSG", pathway="Hippo", hotspots=[]), "FAT1": dict(role="TSG", pathway="Hippo", hotspots=[]),
    "LATS1": dict(role="TSG", pathway="Hippo", hotspots=[]), "LATS2": dict(role="TSG", pathway="Hippo", hotspots=[]),
    "SAV1": dict(role="TSG", pathway="Hippo", hotspots=[]),
    # TGF-beta
    "TGFBR2": dict(role="TSG", pathway="TGFb", hotspots=[]), "TGFBR1": dict(role="TSG", pathway="TGFb", hotspots=[]),
    "SMAD2": dict(role="TSG", pathway="TGFb", hotspots=[]), "SMAD3": dict(role="TSG", pathway="TGFb", hotspots=[]),
    "SMAD4": dict(role="TSG", pathway="TGFb", hotspots=["R361", "D351"]), "ACVR2A": dict(role="TSG", pathway="TGFb", hotspots=["K437"]),
    "ACVR1B": dict(role="TSG", pathway="TGFb", hotspots=[]),
    # MYC
    "MYC": dict(role="OG", pathway="MYC", hotspots=[]), "MAX": dict(role="TSG", pathway="MYC", hotspots=[]),
    "MGA": dict(role="TSG", pathway="MYC", hotspots=[]),
    # NRF2
    "NFE2L2": dict(role="OG", pathway="NRF2", hotspots=["D29", "R34", "G31", "E79", "T80", "D77", "E82", "W24", "L30"]),
    "KEAP1": dict(role="TSG", pathway="NRF2", hotspots=[]), "CUL3": dict(role="TSG", pathway="NRF2", hotspots=[]),
    # chromatin / epigenetic
    "ARID1A": dict(role="TSG", pathway="Chromatin", hotspots=[]), "ARID1B": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "ARID2": dict(role="TSG", pathway="Chromatin", hotspots=[]), "ARID5B": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "KMT2D": dict(role="TSG", pathway="Chromatin", hotspots=[]), "KMT2C": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "KMT2A": dict(role="TSG", pathway="Chromatin", hotspots=[]), "KMT2B": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "SETD2": dict(role="TSG", pathway="Chromatin", hotspots=[]), "PBRM1": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "BAP1": dict(role="TSG", pathway="Chromatin", hotspots=[]), "ATRX": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "DAXX": dict(role="TSG", pathway="Chromatin", hotspots=[]), "SMARCA4": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "SMARCB1": dict(role="TSG", pathway="Chromatin", hotspots=[]), "EZH2": dict(role="OG/TSG", pathway="Chromatin", hotspots=["Y641", "A677", "A687"]),
    "KDM6A": dict(role="TSG", pathway="Chromatin", hotspots=[]), "KDM5C": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "NSD1": dict(role="TSG", pathway="Chromatin", hotspots=[]), "CHD4": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "CTCF": dict(role="TSG", pathway="Chromatin", hotspots=[]), "ASXL1": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "DNMT3A": dict(role="TSG", pathway="Chromatin", hotspots=["R882"]), "TET2": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    "BCOR": dict(role="TSG", pathway="Chromatin", hotspots=[]), "ZBTB20": dict(role="TSG", pathway="Chromatin", hotspots=[]),
    # IDH / metabolism
    "IDH1": dict(role="OG", pathway="IDH", hotspots=["R132"]), "IDH2": dict(role="OG", pathway="IDH", hotspots=["R140", "R172"]),
    "SDHB": dict(role="TSG", pathway="IDH", hotspots=[]), "SDHD": dict(role="TSG", pathway="IDH", hotspots=[]),
    "FH": dict(role="TSG", pathway="IDH", hotspots=[]),
    # DNA repair / MMR / POLE
    "MLH1": dict(role="TSG", pathway="DNArepair", hotspots=[]), "MSH2": dict(role="TSG", pathway="DNArepair", hotspots=[]),
    "MSH6": dict(role="TSG", pathway="DNArepair", hotspots=[]), "PMS2": dict(role="TSG", pathway="DNArepair", hotspots=[]),
    "POLE": dict(role="OG", pathway="DNArepair", hotspots=["P286", "V411", "S459", "A456", "S297", "F367"]),
    "BRCA1": dict(role="TSG", pathway="DNArepair", hotspots=[]), "BRCA2": dict(role="TSG", pathway="DNArepair", hotspots=[]),
    "ATR": dict(role="TSG", pathway="DNArepair", hotspots=[]), "ERCC2": dict(role="TSG", pathway="DNArepair", hotspots=["N238", "D312"]),
    # splicing / RNA
    "SF3B1": dict(role="OG", pathway="Splicing", hotspots=["K700", "K666", "E622", "R625"]),
    "U2AF1": dict(role="OG", pathway="Splicing", hotspots=["S34", "Q157"]), "SRSF2": dict(role="OG", pathway="Splicing", hotspots=["P95"]),
    "RBM10": dict(role="TSG", pathway="Splicing", hotspots=[]),
    # 혈액암 / 림프종
    "NPM1": dict(role="OG", pathway="Heme", hotspots=["W288", "W290", "L287"]), "FLT3": dict(role="OG", pathway="Heme", hotspots=["D835"]),
    "RUNX1": dict(role="TSG", pathway="Heme", hotspots=[]), "CEBPA": dict(role="TSG", pathway="Heme", hotspots=[]),
    "WT1": dict(role="TSG", pathway="Heme", hotspots=[]), "GATA2": dict(role="TSG", pathway="Heme", hotspots=[]),
    "STAG2": dict(role="TSG", pathway="Heme", hotspots=[]), "MYD88": dict(role="OG", pathway="Heme", hotspots=["L265"]),
    "CD79B": dict(role="OG", pathway="Heme", hotspots=["Y196"]), "CARD11": dict(role="OG", pathway="Heme", hotspots=[]),
    "BCL2": dict(role="OG", pathway="Heme", hotspots=[]), "BCL6": dict(role="OG", pathway="Heme", hotspots=[]),
    "TNFAIP3": dict(role="TSG", pathway="Heme", hotspots=[]), "PIM1": dict(role="OG", pathway="Heme", hotspots=[]),
    "BTG1": dict(role="TSG", pathway="Heme", hotspots=[]), "BTG2": dict(role="TSG", pathway="Heme", hotspots=[]),
    "SOCS1": dict(role="TSG", pathway="Heme", hotspots=[]), "B2M": dict(role="TSG", pathway="Immune", hotspots=[]),
    "CD58": dict(role="TSG", pathway="Immune", hotspots=[]), "TNFRSF14": dict(role="TSG", pathway="Heme", hotspots=[]),
    "HLA-A": dict(role="TSG", pathway="Immune", hotspots=[]), "HLA-B": dict(role="TSG", pathway="Immune", hotspots=[]),
    "CASP8": dict(role="TSG", pathway="Immune", hotspots=[]), "JAK1": dict(role="TSG", pathway="Immune", hotspots=[]),
    # 호르몬 / 조직 특이
    "GATA3": dict(role="TSG", pathway="Lineage", hotspots=[]), "ESR1": dict(role="OG", pathway="Lineage", hotspots=["Y537", "D538", "E380"]),
    "FOXA1": dict(role="OG", pathway="Lineage", hotspots=[]), "SPOP": dict(role="TSG", pathway="Lineage", hotspots=["F133", "F102", "W131", "Y87", "F125"]),
    "AR": dict(role="OG", pathway="Lineage", hotspots=["L702", "H875", "T878"]), "MED12": dict(role="OG", pathway="Lineage", hotspots=["G44", "L1224"]),
    "GTF2I": dict(role="OG", pathway="Lineage", hotspots=["L424"]), "GNAS": dict(role="OG", pathway="Lineage", hotspots=["R201", "Q227"]),
    "PRKAR1A": dict(role="TSG", pathway="Lineage", hotspots=[]), "MEN1": dict(role="TSG", pathway="Lineage", hotspots=[]),
    "RHOA": dict(role="OG", pathway="Lineage", hotspots=["Y42", "G17", "R5"]), "CDH1": dict(role="TSG", pathway="Lineage", hotspots=[]),
    "ELF3": dict(role="TSG", pathway="Lineage", hotspots=[]), "RXRA": dict(role="OG", pathway="Lineage", hotspots=["S427"]),
    "TBX3": dict(role="TSG", pathway="Lineage", hotspots=[]), "MAP3K1": dict(role="TSG", pathway="RTK-RAS", hotspots=[]),
    "MAP2K4": dict(role="TSG", pathway="RTK-RAS", hotspots=[]), "RUNX1T1": dict(role="OG", pathway="Heme", hotspots=[]),
    "VHL": dict(role="TSG", pathway="Hypoxia", hotspots=[]), "EPAS1": dict(role="OG", pathway="Hypoxia", hotspots=["A530", "P531"]),
    "TMEM127": dict(role="TSG", pathway="Hypoxia", hotspots=[]), "ALB": dict(role="TSG", pathway="Lineage", hotspots=[]),
    "APOB": dict(role="TSG", pathway="Lineage", hotspots=[]), "RPL22": dict(role="TSG", pathway="Ribosome", hotspots=["K15"]),
    "RPL5": dict(role="TSG", pathway="Ribosome", hotspots=[]), "EIF1AX": dict(role="OG", pathway="Translation", hotspots=["A113"]),
    "PPP2R1A": dict(role="OG", pathway="PI3K", hotspots=["P179", "R183", "S256"]), "PPP6C": dict(role="TSG", pathway="CellCycle", hotspots=["R264"]),
    "DDX3X": dict(role="TSG", pathway="Splicing", hotspots=[]), "PDGFRA": dict(role="OG", pathway="RTK-RAS", hotspots=["D842"]),
    "CSDE1": dict(role="TSG", pathway="Lineage", hotspots=[]), "LZTR1": dict(role="TSG", pathway="RTK-RAS", hotspots=[]),
    "TSHR": dict(role="OG", pathway="Lineage", hotspots=[]), "STK11": dict(role="TSG", pathway="PI3K", hotspots=[]),
    "KRAS": dict(role="OG", pathway="RTK-RAS", hotspots=["G12", "G13", "Q61", "A146"]), "NRAS": dict(role="OG", pathway="RTK-RAS", hotspots=["Q61", "G12", "G13"]),
    "TERT": dict(role="OG", pathway="Telomere", hotspots=[]), "CIC": dict(role="TSG", pathway="RTK-RAS", hotspots=[]),
    "FUBP1": dict(role="TSG", pathway="MYC", hotspots=[]), "RNF43": dict(role="TSG", pathway="WNT", hotspots=["G659"]),
    "ZFHX3": dict(role="TSG", pathway="Lineage", hotspots=[]), "AJUBA": dict(role="TSG", pathway="Hippo", hotspots=[]),
    "ZNF750": dict(role="TSG", pathway="Lineage", hotspots=[]), "FGFR2": dict(role="OG", pathway="RTK-RAS", hotspots=["N549", "S252"]),
    "EPHA2": dict(role="TSG", pathway="RTK-RAS", hotspots=[]), "SOX2": dict(role="OG", pathway="Lineage", hotspots=[]),
}

# ---------------------------------------------------------------- 암종별 문헌 driver 목록 (대표 빈도는 문헌 근사치, %)
# freq: 해당 암종에서 체세포 변이(SNV/indel) 빈도의 문헌 근사값. 없으면 None. 목적: 데이터 관측치와 비교.
CLASS_DRIVERS: dict[str, dict[str, float | None]] = {
    "ACC":    {"TP53": 16, "CTNNB1": 16, "ZNRF3": 19, "PRKAR1A": 8, "CDKN2A": 4, "RPL22": 5, "MEN1": 7, "DAXX": 6, "TERT": 6, "NF1": 5},
    "BLCA":   {"TP53": 48, "KMT2D": 28, "KDM6A": 26, "ARID1A": 25, "PIK3CA": 22, "RB1": 17, "EP300": 15, "FGFR3": 14, "STAG2": 14, "ELF3": 12,
               "CDKN1A": 11, "ERCC2": 9, "TSC1": 8, "RXRA": 9, "KMT2C": 18, "CREBBP": 12, "ERBB2": 12, "ERBB3": 11, "HRAS": 5, "NFE2L2": 8, "FAT1": 12, "ZFP36L1": None},
    "BRCA":   {"PIK3CA": 32, "TP53": 34, "GATA3": 11, "MAP3K1": 9, "CDH1": 13, "KMT2C": 8, "PTEN": 5, "AKT1": 3, "RUNX1": 3, "NF1": 4,
               "TBX3": 3, "ESR1": 2, "ARID1A": 4, "MAP2K4": 4, "FOXA1": 3, "SF3B1": 2, "CBFB": 2, "NCOR1": 4},
    "CESC":   {"PIK3CA": 26, "EP300": 12, "FBXW7": 11, "PTEN": 8, "TP53": 6, "ERBB2": 5, "ERBB3": 6, "MAPK1": 5, "HLA-A": 5, "HLA-B": 6,
               "NFE2L2": 6, "ARID1A": 9, "KMT2C": 9, "KMT2D": 10, "CASP8": 6, "TGFBR2": 5, "KRAS": 6, "STK11": 3, "SHKBP1": 4},
    "COAD":   {"APC": 76, "TP53": 55, "KRAS": 43, "PIK3CA": 18, "FBXW7": 12, "SMAD4": 11, "TCF7L2": 9, "BRAF": 10, "SOX9": 8, "ARID1A": 9,
               "NRAS": 6, "ATM": 8, "AMER1": 8, "CTNNB1": 5, "ACVR2A": 8, "POLE": 6, "RNF43": 8, "SMAD2": 4, "PTEN": 5, "ERBB3": 5},
    "DLBC":   {"KMT2D": 27, "CREBBP": 22, "TP53": 18, "MYD88": 18, "CD79B": 12, "CARD11": 10, "EZH2": 10, "BCL2": 20, "BCL6": 10,
               "TNFAIP3": 12, "PIM1": 22, "BTG1": 12, "BTG2": 15, "SOCS1": 16, "B2M": 15, "CD58": 8, "TNFRSF14": 12, "HLA-A": 8, "HLA-B": 10, "EP300": 10, "MEF2B": 8},
    "GBMLGG": {"IDH1": 45, "TP53": 38, "ATRX": 25, "EGFR": 15, "PTEN": 15, "NF1": 8, "PIK3CA": 8, "PIK3R1": 7, "RB1": 5, "CIC": 12, "FUBP1": 5,
               "IDH2": 2, "NOTCH1": 5, "PDGFRA": 3, "SMARCA4": 3, "TERT": 40},
    "HNSC":   {"TP53": 72, "CDKN2A": 22, "PIK3CA": 21, "NOTCH1": 19, "FAT1": 23, "CASP8": 9, "NSD1": 12, "KMT2D": 18, "HRAS": 5, "AJUBA": 7,
               "NFE2L2": 6, "EP300": 7, "FBXW7": 5, "TGFBR2": 4, "CUL3": 6, "EPHA2": 4, "ZNF750": 5, "HLA-A": 5, "PTEN": 4, "RB1": 4, "KMT2C": 7},
    "KIPAN":  {"VHL": 35, "PBRM1": 30, "SETD2": 12, "BAP1": 10, "KDM5C": 6, "MTOR": 6, "TP53": 5, "PTEN": 4, "MET": 6, "NF2": 3, "KDM6A": 3,
               "SMARCB1": 2, "FH": 2, "PIK3CA": 3, "TSC1": 3, "STAG2": 3, "CDKN2A": 3},
    "KIRC":   {"VHL": 52, "PBRM1": 33, "SETD2": 12, "BAP1": 10, "KDM5C": 7, "MTOR": 6, "TP53": 3, "PTEN": 4, "PIK3CA": 3, "TCEB1": 2, "ARID1A": 3, "ATM": 3},
    "LAML":   {"NPM1": 27, "FLT3": 28, "DNMT3A": 26, "IDH2": 10, "IDH1": 9, "TET2": 8, "RUNX1": 10, "CEBPA": 7, "TP53": 8, "NRAS": 12, "KRAS": 4,
               "KIT": 4, "WT1": 6, "U2AF1": 4, "ASXL1": 3, "SRSF2": 3, "STAG2": 3, "PTPN11": 5, "GATA2": 3, "EZH2": 2, "SF3B1": 2, "SMC1A": 3},
    "LGG":    {"IDH1": 77, "TP53": 47, "ATRX": 37, "CIC": 20, "FUBP1": 9, "NOTCH1": 8, "PIK3CA": 7, "PIK3R1": 4, "IDH2": 4, "EGFR": 5, "PTEN": 4,
               "SMARCA4": 4, "ZBTB20": 3, "ARID1A": 3, "NF1": 4, "TERT": 20},
    "LIHC":   {"TP53": 31, "CTNNB1": 27, "AXIN1": 8, "ALB": 13, "ARID1A": 7, "ARID2": 7, "RB1": 5, "NFE2L2": 3, "KEAP1": 5, "BAP1": 5,
               "APOB": 10, "TERT": 44, "RPS6KA3": 5, "CDKN2A": 3, "ACVR2A": 3, "LZTR1": 3, "TSC2": 4, "PIK3CA": 3},
    "LUAD":   {"KRAS": 33, "EGFR": 14, "TP53": 46, "STK11": 17, "KEAP1": 17, "BRAF": 7, "NF1": 11, "MET": 7, "RBM10": 7, "SMARCA4": 6,
               "ARID1A": 7, "PIK3CA": 7, "RB1": 4, "CDKN2A": 4, "ERBB2": 3, "SETD2": 9, "U2AF1": 3, "RIT1": 2, "CTNNB1": 4, "KMT2D": 9, "ATM": 9},
    "LUSC":   {"TP53": 81, "CDKN2A": 15, "PIK3CA": 16, "NFE2L2": 15, "KEAP1": 12, "PTEN": 8, "KMT2D": 20, "NOTCH1": 8, "RB1": 7, "HLA-A": 3,
               "FAT1": 12, "CUL3": 7, "RASA1": 4, "FBXW7": 5, "NF1": 11, "ARID1A": 8, "HRAS": 3, "FGFR3": 2, "EP300": 7, "CDKN1A": 2},
    "OV":     {"TP53": 95, "BRCA1": 3, "BRCA2": 3, "NF1": 4, "RB1": 2, "CDK12": 3, "CSMD3": 6, "FAT3": 6, "GABRA6": 6, "PIK3CA": 1, "KRAS": 1},
    "PAAD":   {"KRAS": 90, "TP53": 70, "CDKN2A": 25, "SMAD4": 22, "ARID1A": 8, "RNF43": 7, "GNAS": 6, "TGFBR2": 5, "KDM6A": 5, "BRCA2": 3,
               "ATM": 4, "RBM10": 4, "KMT2C": 6, "KMT2D": 4, "PIK3CA": 2, "BRAF": 2},
    "PCPG":   {"HRAS": 10, "NF1": 8, "RET": 6, "VHL": 4, "EPAS1": 5, "SDHB": 3, "SDHD": 2, "TMEM127": 2, "MAX": 2, "CSDE1": 3, "ATRX": 3, "SETD2": 3, "FGFR1": 2, "TP53": 2},
    "PRAD":   {"SPOP": 11, "TP53": 8, "FOXA1": 4, "PTEN": 4, "KMT2D": 4, "KMT2C": 4, "CDK12": 3, "ATM": 4, "MED12": 3, "ZFHX3": 3, "BRAF": 2,
               "IDH1": 1, "PIK3CA": 3, "CTNNB1": 2, "APC": 2, "AR": 1, "CHD1": 2},
    "SARC":   {"TP53": 35, "ATRX": 20, "RB1": 12, "NF1": 8, "PTEN": 5, "MDM2": None, "CDK4": None, "PIK3CA": 3, "DNMT3A": 2, "KIT": None, "ARID1A": 4},
    "SKCM":   {"BRAF": 50, "NRAS": 28, "NF1": 14, "TP53": 15, "CDKN2A": 13, "PTEN": 8, "RAC1": 5, "ARID2": 15, "PPP6C": 8, "IDH1": 6,
               "MAP2K1": 6, "KIT": 3, "DDX3X": 5, "RB1": 5, "TERT": 70, "KMT2D": 10, "RASA2": 5, "CTNNB1": 4},
    "STES":   {"TP53": 60, "ARID1A": 20, "PIK3CA": 12, "CDH1": 8, "RHOA": 5, "KRAS": 8, "SMAD4": 6, "APC": 8, "ERBB2": 5, "CTNNB1": 4, "MUC6": 8,
               "RNF43": 6, "BCOR": 4, "KMT2D": 14, "NFE2L2": 8, "NOTCH1": 8, "CDKN2A": 8, "ZNF750": 5, "FAT1": 8, "PTEN": 4, "ELF3": 3, "EGFR": 3, "XYLT2": None, "ERBB3": 5, "KMT2C": 8, "FBXW7": 5},
    "TGCT":   {"KIT": 17, "KRAS": 12, "NRAS": 4, "TP53": 2, "CDC27": None, "NCOR2": None, "BMP2K": None},
    "THCA":   {"BRAF": 60, "NRAS": 8, "HRAS": 4, "KRAS": 1, "EIF1AX": 2, "TERT": 9, "PPM1D": 2, "CHEK2": 2, "RET": None, "TSHR": 1, "TG": None},
    "THYM":   {"GTF2I": 39, "HRAS": 7, "NRAS": 4, "TP53": 3, "KIT": 1, "KRAS": 1},
    "UCEC":   {"PTEN": 64, "PIK3CA": 53, "ARID1A": 46, "PIK3R1": 33, "CTNNB1": 30, "KRAS": 21, "TP53": 28, "POLE": 10, "FBXW7": 16, "PPP2R1A": 10,
               "CTCF": 25, "KMT2D": 16, "ZFHX3": 20, "RPL22": 13, "FGFR2": 12, "ARID5B": 15, "CHD4": 12, "SPOP": 10, "JAK1": 9, "MSH6": 8, "CCND1": 4},
}

PATHWAYS = sorted({v["pathway"] for v in GENE_INFO.values()})


def hotspot_match(gene: str, tok: str) -> str | None:
    """변이 문자열이 문헌 hotspot 위치와 일치하면 그 위치를 반환 (예: 'R132H' → 'R132')."""
    info = GENE_INFO.get(gene)
    if not info:
        return None
    for h in info["hotspots"]:
        if tok.startswith(h) and (len(tok) == len(h) or not tok[len(h)].isdigit()):
            return h
    return None
