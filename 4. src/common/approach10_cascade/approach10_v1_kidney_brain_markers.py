"""approach10 v1: KIRC(신장 투명세포암)/KIPAN(신장암 전체), GBMLGG(교모세포종+저등급 신경교종)/LGG(저등급 신경교종)
구분용 계단식(cascade) 마커 피처. 04_todo_day2.md STEP 3-2 참고.

⚠️ 2026-09-10 수정: 원래 PBRM1/BAP1/SETD2/CIC/FUBP1을 참조했으나, 이 데이터 패널(4,384개 유전자)에
해당 유전자가 없어서 항상 0으로 죽어있었음(train.csv 직접 확인). VHL/MET/IDH1/EGFR/ATRX만 실제 존재.

가설(train.csv 직접 집계, 클래스별 변이 보유율):
- VHL: KIRC 45.2% / KIPAN 29.9% / GBMLGG 0.4% / LGG 0.4% — 신장 계열에서만 흔하고, KIRC가 KIPAN보다 높음
- MET: KIRC 1.5% / KIPAN 3.7% — KIPAN(KIRP형)에서 상대적으로 더 흔함
- IDH1: GBMLGG 41.0% / LGG 78.6% — LGG에서 압도적으로 흔함
- EGFR: GBMLGG 14.5% / LGG 3.5% — GBMLGG(GBM형)에서 훨씬 흔함
- ATRX: GBMLGG 21.5% / LGG 36.2% — LGG 쪽이 더 흔함(보조 신호)
"""
import pandas as pd


class CascadeFeatures:
    """상태 없음(고정 마커 유전자 목록) — FeatureMaker 인터페이스 맞춤용."""

    def fit(self, df: pd.DataFrame) -> "CascadeFeatures":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def has(gene: str):
            return (df[gene] != "WT").astype(int) if gene in df.columns else 0

        out = pd.DataFrame(index=df.index)
        out["kidney_kirc_signal"] = has("VHL") & (1 - has("MET"))
        out["kidney_kipan_signal"] = has("MET") & (1 - has("VHL"))
        out["brain_lgg_signal"] = has("IDH1") & (1 - has("EGFR"))
        out["brain_gbmlgg_signal"] = has("EGFR")
        return out.astype(int)
