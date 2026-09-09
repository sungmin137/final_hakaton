# 실험 기록

| 날짜 | 실험명 | 모델/피처 요약 | CV 점수 | LB 점수 | 비고 |
|---|---|---|---|---|---|
| 2026-09-09 | baseline_lgbm_v1 | 유전자 이진화 4,230 + 변이수/유형 카운트 10, LightGBM lr0.05 ff0.3, StratifiedKFold5 | MacroF1 0.3720 / Acc 0.3862 | - | early stop ~100 iter, KIRC/LGG/SARC/CESC F1<0.15 |
| 2026-09-09 | official_xgb (공식 베이스라인 재현) | OrdinalEncoder 문자열→정수 4,384, XGB 100트리 depth6 lr0.1 | MacroF1 0.3048 / Acc 0.3288 | - | 공식 노트북은 검증 없음 → 우리 5-Fold로 측정 |
| 2026-09-09 | v1_xgb (메인 파이프라인) | 유전자 이진화 4,230 + 카운트 10, XGB 공식 파라미터 동일 | MacroF1 0.3845 / Acc 0.3917 | - | 공식 인코딩 대비 +0.08. LGBM v1(0.372)보다 높음 |
