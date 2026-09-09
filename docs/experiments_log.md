# 실험 기록

| 날짜 | 실험명 | 모델/피처 요약 | CV 점수 | LB 점수 | 비고 |
|---|---|---|---|---|---|
| 2026-09-09 | baseline_lgbm_v1 | 유전자 이진화 4,230 + 변이수/유형 카운트 10, LightGBM lr0.05 ff0.3, StratifiedKFold5 | MacroF1 0.3720 / Acc 0.3862 | - | early stop ~100 iter, KIRC/LGG/SARC/CESC F1<0.15 |
