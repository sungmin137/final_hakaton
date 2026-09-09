# 실험 기록

| 날짜 | 실험명 | 모델/피처 요약 | CV 점수 | LB 점수 | 비고 |
|---|---|---|---|---|---|
| 2026-09-09 | baseline_lgbm_v1 | 유전자 이진화 4,230 + 변이수/유형 카운트 10, LightGBM lr0.05 ff0.3, StratifiedKFold5 | MacroF1 0.3720 / Acc 0.3862 | - | early stop ~100 iter, KIRC/LGG/SARC/CESC F1<0.15 |
| 2026-09-09 | official_xgb (공식 베이스라인 재현) | OrdinalEncoder 문자열→정수 4,384, XGB 100트리 depth6 lr0.1 | MacroF1 0.3048 / Acc 0.3288 | - | 공식 노트북은 검증 없음 → 우리 5-Fold로 측정 |
| 2026-09-09 | v1_xgb (메인 파이프라인) | 유전자 이진화 4,230 + 카운트 10, XGB 공식 파라미터 동일 | MacroF1 0.3845 / Acc 0.3917 | - | 공식 인코딩 대비 +0.08. LGBM v1(0.372)보다 높음 |
| 2026-09-09 | approach1 단독 (variant nb) | 클래스별 변이 개수→NB 로그우도비 합산 | MacroF1 0.2765 / Acc 0.3079 | - | 단독 분류기, 개수 그대로는 0.23 |
| 2026-09-09 | v2_xgb (v1 + 접근1 점수 140개) | 내부 5-Fold OOF 점수 피처, XGB 공식 파라미터 | MacroF1 0.4942 / Acc 0.5265 | - | v1 대비 +0.11. sungmin 브랜치 |
| 2026-09-09 | v2_xgb 1차 제출 (test 최초 사용) | 전체 train 학습 → test 추론. twin 규칙 버전 별도 | CV 0.4942 | (제출 후 기록) | test 결측 237셀→WT, test 초과변이 5.6%로 train 3배, STES 21.5% 과잉 예측. docs/08 |
