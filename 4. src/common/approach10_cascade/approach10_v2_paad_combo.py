"""approach10 v2: PAAD(췌장암)이 OV(난소암)·BRCA(유방암)로 오분류되는 문제 검증용 (04_todo_day2.md).

KRAS(PAAD 핵심 driver, 90%↑)가 패널에 없어서 TP53만으로는 OV/BRCA와 구분 안 됨.
train.csv 직접 집계: TP53+CDKN2A 동반 변이는 OV(0/253), BRCA(0/786)에는 전혀 없고
HNSC(40)·LUSC(18)·PAAD(14)·STES(13)에 몰려있음 — OV/BRCA를 배제하는 신호로 시도.
"""
import pandas as pd


class PaadComboFeature:
    def fit(self, df: pd.DataFrame) -> "PaadComboFeature":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        has_tp53 = (df["TP53"] != "WT").astype(int)
        has_cdkn2a = (df["CDKN2A"] != "WT").astype(int)
        return pd.DataFrame({"tp53_cdkn2a_combo": has_tp53 & has_cdkn2a}, index=df.index).astype(int)
