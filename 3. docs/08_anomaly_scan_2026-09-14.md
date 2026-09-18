# Anomaly-driven diagnostic scan (2026-09-14)

학습/CV/제출 없이 기존 결과(result.json, oof_proba.npy, confusion_matrix.csv, test_proba*.npy)만으로 진행한 2~3단계 진단. 코드: `4. src/common/approach23_synthetic_aug/approach23_v4_anomaly_scan.py`(전체 스캔), `approach23_v5_priority_narrowing.py`(우선순위 좁히기). 원본 출력: `6. experiments/2026-09-14_approach23_anomaly_scan/`.

## 조사 범위
- result.json 51개 스캔 → group_twins=True + oof_proba.npy 존재 + train 행수(6,201) 일치 = **29개** 직접 비교 가능(동일 fold 분할).
- `2026-09-10_a7_xgb_grp_khs_tp53_local`(oof_macro_f1=0.156, 전 fold 균일 붕괴)은 깨진 실행으로 판단해 클래스 통계에서 제외 → **28개**로 최종 분석.
- approach 문서 전체, decisions.md/STATUS.md의 제출 이력(21회), test 분포 스캔(approach23_v3) 결과 재사용.

## 발견 1 — CV→LB 전이의 경험적 패턴 (강한 법칙이 아니라 정황적 decision signal)
21회 제출 중 CV 개선이 기록된 변경 6건 중 4건(GBMLGG/LGG 라우팅, 라우팅 통합, 이진유전자 LR 서드모델, a7 서드모델)은 LB가 반전되거나 악화됐고, 2건(approach14 v3의 31-100 구간 겨냥 블렌드, hypermut_rule의 396+ 구간 겨냥 후처리)만 LB 개선으로 이어졌다. **"지금까지 확인된 제출 사례에서는, CV 개선만을 근거로 한 접근보다 사전에 확인된 train/test distribution shift를 직접 겨냥한 접근이 LB 전이에서 훨씬 강한 신호를 보였다"** — n=6으로 법칙화하기엔 표본이 작아 정황적 신호로만 취급한다.

## 발견 2 — BRCA/KIPAN 싱크 (OOF에서 강하나 test 관련성 약함 → 기록만 유지)
29/29 실험에서 SARC/OV/PRAD/THYM/PCPG/CESC/HNSC/PAAD/UCEC/STES → BRCA 로 새는 edge가 반복 관찰(OOF 예측 비중이 실제 대비 +6.9%p 과다). 그러나 실제 9차 모델의 test 예측에서는 BRCA 과다예측이 +1.6%p에 그치고, test의 지배적 편향은 STES(+16.5%p)다. **BRCA 싱크는 test 레버리지가 약해 개입하지 않고 기록만 유지한다.**

## 발견 3 — burden=0이 1-3보다 강한 단절점 (표본 작아 정황적)
28개 실험 평균 macro F1: burden 0 = 0.015(std 0.009, 최댓값도 0.032) vs 1-3 = 0.204(std 0.019). 0 vs 1-3의 격차(0.19)가 1-3 vs 4-7 격차(0.03)보다 6배 크다. 다만 train 94행/test 31행으로 절대 레버리지는 작다.

## 발견 4 — cw_plus 신규 토큰 4종 중 3종이 반복적으로 LGG를 깎음
v4p→{pos, prop, band} 전부 LGG F1이 최대 악화 클래스 중 하나(−0.038~−0.044). approach23의 GBMLGG/LGG 관찰을 독립적으로 재확인.

## 발견 5 (3단계 추가 분석) — KIRC/KIPAN: 강한 anomaly이나 GBMLGG/LGG식 burden 메커니즘은 적용 안 됨
- test NN(자카드) 불일치 121행(KIRC↔KIPAN)으로 전체 클래스쌍 중 최대. edge KIRC↔KIPAN은 29/29 실험에서 45~48% 공유율로 반복. KIRC 자체 F1은 28개 실험 평균 0.27, std 0.027로 사실상 고정(어떤 feature 개입도 안 움직임).
- **그러나 OOF에서 KIRC 정답(n=126, burden mean 13.5)과 오답(n=208, mean 13.2)의 burden 분포가 거의 동일** — GBMLGG/LGG(정답 14.8 vs 오답 10.9)에서 보였던 뚜렷한 burden 경계가 KIRC에는 없다. **burden 기반 synthetic/도너 제한 로직을 KIRC/KIPAN에 그대로 이식할 근거는 없다.**
- KIPAN은 정답(n=187, mean 19.1) vs 오답(n=328, mean 14.3, std 15.5)에 차이가 있어 KIRC와 비대칭.
- KIRC/KIPAN을 가장 잘 가르는 유전자(VHL 등, 데이터로 직접 산출)의 test-train 빈도 shift는 대부분 앞서 발견한 "확산형 burden 인플레이션 유전자 클러스터"(TP53·CACNA1A 등)와 겹쳐, KIRC/KIPAN 고유의 신호라기보다 전반적 test burden 인플레이션의 부산물일 가능성이 높다.

## 우선순위 재평가 (요청받은 5축 기준)

| 우선순위 | 후보 | 지금 할 일 |
|---|---|---|
| 1 | Test distribution shift 잔여 영역 | KIRC/KIPAN twin_rule 적용 실태 감사(아래 추천 실험) |
| 2 | burden=0 구조 | THYM 제외 후 나머지 클래스 구분력 재확인(추천 실험) |
| 3 | 1-3/저변이 | 샘플 제거·가중치 실험 안 함 |
| 4 | BRCA sink | 기록만, 개입 안 함 |
| 5 | GBMLGG/LGG synthetic | 잠정 보류(burden 8-15 donor 가설 폐기) |
| 6 | cw_plus 신규 token | 추가 탐색 중단 |

## 추천 실험 (둘 다 학습 없음, 기존 결과 재계산만)
1. **KIRC/KIPAN twin_rule 감사**: test에서 KIRC/KIPAN으로 예측된 행 중 몇 %가 train과 완전 동일(twin_rule 적용 대상)인지, 그리고 21차 제출본에서 twin_rule이 실제로 몇 행을 바꿨는지 대조. approach11 v10d가 이미 "KIRC 82.6%가 쌍둥이"라고 밝혔으므로, 이 쌍둥이 처리가 실제로 최적으로 작동하는지 순수 카운팅으로 확인 가능.
2. **burden=0 94행에서 THYM 제외 재확인**: 기존 OOF로 THYM을 제외한 나머지 클래스만의 pairwise 구분력을 다시 계산 — "무변이=THYM 사전확률 지배" 가설을 재학습 없이 검증.
