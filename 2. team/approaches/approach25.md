# 접근 25 — v19 class_scale 오류 메커니즘 진단 (confusion boundary + attribution audit, OV-SARC counterfactual)

## 개요
새 모델·새 피처 없음. 현재 팀 최고 파이프라인(v19)이 이미 만들어낸 OOF 확률만 재분석하는 순수 진단 작업임. 접근14/16과 마찬가지로 독립된 파이프라인이 아니라 **v19에서 파생한 오류구조 분석**임. 작업일: 2026-09-16, 다른 세션(스크래치 워크스페이스)에서 진행 후 이 문서로 복구·정리함. 원본 코드는 세션 내 일회성 bash 스크립트라 별도 파일로 안 남아있고, 수치·표는 세션 기록에서 그대로 옮김. train.csv OOF만 사용, test.csv 미사용(진단 대상이 train 정답과 raw/scaled OOF 확률 비교라 test 접근 자체가 불필요).

## Stage 0 — confusion boundary audit
쌍둥이 구조로 이미 상한이 결정된 GBMLGG↔LGG, KIPAN↔KIRC는 제외하고, 나머지 혼동 쌍들의 margin(정답과 오답 클래스 확률 차이) 분포를 확인함. 아래 3개 쌍이 후보로 추려짐.

| 방향 | 건수 | margin Q25 | 비고 |
|---|---|---|---|
| THYM → PRAD | 35 | 0.249(중앙값) | THYM recall 45.9%, PRAD recall 60.5% |
| PRAD → THYM | 14 | **0.055** | 경계 근처에서 THYM으로 넘어감 |
| STES → LUSC | 30 | 0.139 | 오류 건수는 많으나 경계형 아님 |
| LUSC → STES | 3 | **0.051** | 건수는 적으나 경계형 |
| OV → SARC | 61 | 0.18~0.19 | SARC hub 현상의 대표 edge |
| SARC → OV | 18 | 0.18~0.19 | |

BRCA→SARC, BRCA→PRAD는 방향성이 너무 극단적이라 우선순위에서 뒤로 미룸(미조사).

**함정 하나 미리 정리함**: "margin이 작다 = class_scale이 만든 오류"가 아님. margin은 최종 확률의 스냅샷일 뿐이라, 원인(class_scale이 실제로 뒤집었는지)을 보려면 scale 적용 전(raw)·후(scaled) argmax를 직접 비교해야 함.

## Stage 1 — class_scale attribution audit
각 오답 샘플을 raw→scaled argmax 변화로 3종 분류함: **scale-induced**(raw는 정답, scale 후 오답으로 뒤집힘) / **scale-amplified**(raw부터 오답이었으나 scale이 더 심화) / **scale-independent**(raw부터 이미 오답, scale과 무관).

| 방향(건수) | scale-induced | scale 전부터 오답 | raw에서 제3클래스였음 | 판정 |
|---|---|---|---|---|
| PRAD→THYM (14) | 1건 | 1건 | **12건(86%)** | THYM 배율의 광범위한 흡인 현상 부산물, edge correction 대상 아님 → 종료 |
| LUSC→STES (3) | 0건 | **3건(100%)** | 0건 | class_scale과 완전 무관, 순수 모델 오류 → 종료 |
| OV→SARC (61) | **17건(28%)** | 3건 | 41건(67%) | 유일하게 진짜 신호 있음 → 심화 조사 |

PRAD→THYM, LUSC→STES 두 쌍은 여기서 종료함. OV→SARC만 다음 단계로 넘어감.

## OV→SARC counterfactual audit (핵심 발견)
### 1. 17건의 raw 확률 구조
17건 전부 raw에서 **P(OV) > P(SARC)**였음(가장 근소한 것도 0.222 vs 0.127). 하지만 "OV vs SARC 2위 경쟁"이었는지 확인하기 위해 raw SARC 순위를 직접 셈.

### 2. SARC로 가는 전체 flip(590건, 26클래스 전체 기준)의 raw SARC 순위 분포
| raw SARC 순위 | 건수 |
|---|---|
| 2위 | 205 (34.7%) |
| 3위 이하 | 385 (65.3%) |

**raw 1위 → SARC flip의 source 클래스**: BRCA 380건(64%), GBMLGG 73, OV 45, KIPAN 43, PRAD 17 등. **SARC 배율은 압도적으로 BRCA를 노림** — OV는 7.6%뿐.

### 3. OV→SARC 17건 세부 — 진짜 경쟁자는 BRCA였음
17건 중 raw SARC 순위: 2위 5건 / 3위 10건 / 4위 2건 → **71%가 3위 이하**. 그리고 17건 중 **11건은 진짜 raw 2위가 SARC가 아니라 BRCA**였음(예: `OV .337 > BRCA .328 > SARC .178`). 즉 이건 "OV vs SARC" 대결이 아니라 **"OV vs BRCA" 대결에서 SARC가 배율 힘으로 3위에서 끼어들어 둘 다 제친 것**.

### 4. Group A(피해자) vs Group B(보호됨) 대조
| | n | P(OV) 평균 | P(SARC) 평균 | OV/SARC 비율 |
|---|---|---|---|---|
| A — SARC로 flip됨 | 17 | 0.386 | 0.154 | 2.57 |
| B — OV 유지됨 | 85 | 0.503 | 0.034 | 27.72 |

두 그룹의 OV/SARC 비율 차이(2.57 vs 27.72)는 명확한 구조적 차이임 — SARC가 A그룹엔 실제로 가까이 있고 B그룹엔 멀리 있음. 다만 이 차이가 "OV-SARC 2클래스 경계"를 의미하진 않음(3번 참고).

## 판정 (미리 정한 GO/NO-GO 기준 적용)

| 기준 | 충족 여부 |
|---|---|
| GO: SARC가 대부분 raw 2위 | ❌ (71%가 3위 이하) |
| GO: 대조군과 구별되는 일관된 구조 | ✅ (비율 2.57 vs 27.72) |
| NO-GO: 전체 SARC flip이 여러 클래스 걸친 동일 hub 패턴 | ✅ 해당 (BRCA가 64% 차지) |

기준이 엇갈려서 원래의 단순 이분법보다 더 정확한 결론이 나옴 — **"OV-SARC 2클래스 edge가 아니라, OV-BRCA-SARC 3자 구도에서 SARC가 배율로 끼어드는 것"**. 순수 pairwise routing(OV vs SARC만 비교)은 BRCA를 고려 안 하는 구조라 이 현상을 애초에 못 잡음.

- **OV-SARC pairwise correction: NO-GO**
- **OV-BRCA-SARC 3-way correction: 아이디어로는 타당하나 근거 부족(직접 피해 17건, 그중 11건은 BRCA 개입까지 겹쳐 구조 복잡, SARC가 이미 전역 hub라 3-class correction이 다른 오류에 개입할 위험) — NO-GO**
- **SARC class_scale 메커니즘 자체에 대한 새로운 이해: GO** (지식 자산으로 채택)

## 이 진단이 왜 유용한가 — 기존 결론의 "이유"를 설명함
approach14.md에 이미 있던 결론 "어떤 배율 재적합도 고정 벡터를 못 이긴다 … macro F1 최적해"가 **왜 그런지**를 메커니즘 수준으로 설명해줌:
- SARC는 표본이 작은 클래스(198명)라 raw 모델에서 자연스럽게 top-2에 드는 경우가 드묾(전체 SARC flip 590건 중 34.7%뿐)
- macro F1은 작은 클래스도 큰 클래스와 동일 비중으로 채점하므로, SARC 재현율을 올리려면 배율이 "정직하게 2위인 것만" 밀어주는 걸로는 부족하고 **3~5위까지 파고들어야 함**
- 그래서 배율을 약하게 재적합하면(팀이 과거 여러 번 시도) SARC 재현율이 확 떨어졌고, 지금의 "공격적인" 고정 배율이 오히려 macro F1엔 최적이었던 것 — 이번 진단이 그 이유를 처음으로 밝힘

## 결론
OV-SARC를 포함한 3개 edge 모두 pairwise/3-way correction으로는 닫혔음. 다만 **"class_scale 오류를 pairwise routing으로 보면 안 된다"는 것을 실제 OOF flip 구조로 확인한 것 자체가 이번 진단의 성과**임 — 실패한 라우팅 실험(approach11의 GBMLGG/LGG routing, approach24의 pairwise referee 8종)이 왜 전부 막혔는지에 대한 공통 설명이 됨. 이 축은 여기서 종료함.
