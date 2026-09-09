# 실험 기록

| 날짜 | 실험명 | 모델/피처 요약 | CV 점수 | LB 점수 | 비고 |
|---|---|---|---|---|---|
| 2026-09-09 | baseline_lgbm_v1 | 유전자 이진화 4,230 + 변이수/유형 카운트 10, LightGBM lr0.05 ff0.3, StratifiedKFold5 | MacroF1 0.3720 / Acc 0.3862 | - | early stop ~100 iter, KIRC/LGG/SARC/CESC F1<0.15 |
| 2026-09-09 | official_xgb (공식 베이스라인 재현) | OrdinalEncoder 문자열→정수 4,384, XGB 100트리 depth6 lr0.1 | MacroF1 0.3048 / Acc 0.3288 | - | 공식 노트북은 검증 없음 → 우리 5-Fold로 측정 |
| 2026-09-09 | v1_xgb (메인 파이프라인) | 유전자 이진화 4,230 + 카운트 10, XGB 공식 파라미터 동일 | MacroF1 0.3845 / Acc 0.3917 | - | 공식 인코딩 대비 +0.08. LGBM v1(0.372)보다 높음 |
| 2026-09-09 | approach1 단독 (variant nb) | 클래스별 변이 개수→NB 로그우도비 합산 | MacroF1 0.2765 / Acc 0.3079 | - | 단독 분류기, 개수 그대로는 0.23 |
| 2026-09-09 | v2_xgb (v1 + 접근1 점수 140개) | 내부 5-Fold OOF 점수 피처, XGB 공식 파라미터 | MacroF1 0.4942 / Acc 0.5265 | - | v1 대비 +0.11. sungmin 브랜치 |
| 2026-09-09 | v2_xgb 1차 제출 (test 최초 사용) | 전체 train 학습 → test 추론. twin 규칙 버전 별도 | CV 0.4942 | **0.41** | test 결측 237셀→WT, test 초과변이 5.6%로 train 3배, STES 21.5% 과잉 예측. docs/08 |
| 2026-09-09 | v1_xgb (정직 CV, 쌍둥이 그룹) | 동일 피처, StratifiedGroupKFold(쌍둥이 같은 fold) | MacroF1 0.3979 / Acc 0.4301 | - | 쌍둥이 효과 제거 기준선 |
| 2026-09-09 | v2_xgb (정직 CV, 쌍둥이 그룹) | 동일 피처, 쌍둥이 같은 fold | MacroF1 0.4486 / Acc 0.4622 | 0.41 (제출본) | 일반 CV 0.4942 → 0.4486. LB 격차 0.08 → 0.04로 축소 |
| 2026-09-09 | v2_xgb + 클래스가중치 (정직 CV) | sqrt 역빈도 sample_weight | MacroF1 0.4478 / Acc 0.4594 | - | 효과 없음 (기준 0.4486) |
| 2026-09-09 | v2_xgb tuned + 클래스가중치 (정직 CV) | 600트리 lr0.05 colsample0.3 subsample0.8 | MacroF1 0.4535 / Acc 0.4682 | - | 기준 0.4486 대비 +0.005 |
| 2026-09-09 | v3_xgb (정직 CV) | v2 + hotspot 위치 one-hot·LoF 유전자·조합·특수그룹 플래그, 공식 파라미터 | MacroF1 0.4663 / Acc 0.4685 | - | v2 0.4486 대비 +0.018 |
| 2026-09-09 | v3_xgb tuned (정직 CV) | v3 + 600트리 lr0.05 colsample0.3 | MacroF1 0.4576 / Acc 0.4717 | - | v3 공식(0.4663)보다 낮음 → 공식 파라미터 채택 |
| 2026-09-09 | v3_xgb 2차 제출 | make_submission.py --features v3, 공식 파라미터 | 정직 CV 0.4663 | (제출 후 기록) | submission.csv 갱신. STES 21% 과잉 예측은 여전 |
| 2026-09-09 | v3_lgbm (정직 CV) | LightGBM 300트리 lr0.05 ff0.3 | MacroF1 0.4516 / Acc 0.4720 | - | XGB v3 0.4663보다 낮음. 앙상블 재료 |
| 2026-09-09 | v3_mlp (정직 CV, GPU/MPS) | MLP 512-256, dropout0.4, AdamW, label smoothing | MacroF1 0.3416 / Acc 0.3899 | - | 부스팅보다 낮음. 앙상블 재료 후보 |
| 2026-09-09 | v3 앙상블 (OOF 가중평균) | xgb 1.0 + mlp 0.2 | MacroF1 0.4699 | - | +0.004. lgbm 섞으면 하락. 모델 앙상블은 지렛대 아님 |
| 2026-09-09 | v3_xgb + 클래스 배율 후처리 | OOF 좌표상승으로 배율 탐색, 절반 교차확인 | 홀드아웃 +0.012 (0.4663 → 약 0.479) | - | src/postprocess/class_scale.py, 규칙 준수(train OOF만) |
| 2026-09-09 | v3_cat (정직 CV) | CatBoost 300 iter rsm0.3 | 중단 | - | fold0에 11분+ 소요, 앙상블 효과 미미(lgbm 기준)라 중단 |
| 2026-09-09 | v4_xgb (정직 CV) | v3 + BLOSUM62·아미노산 특성 지식 피처 68개 | MacroF1 0.4691 / Acc 0.4698 | - | v3 0.4663 대비 +0.003 (미미) |
| 2026-09-09 | 시각화 | train 전용 연관성 대시보드 8종 (docs/viz/mutation_associations.html, 아티팩트 공유) | - | - | 암종×표지유전자, 암종 유사도, 변이수, 유형구성, 동시변이 lift, hotspot→암종, driver 조합, 동일 프로필 쌍 |
| 2026-09-09 | v4_xgb + 클래스 배율 3차 제출 | make_submission.py --features v4 --class-scale (train OOF 기반) | 정직 CV 0.4691 (+후처리 약 0.012) | (제출 후 기록) | submission.csv 갱신. STES 과잉 예측 21%→15% (train OOF 배율의 부수 효과) |
