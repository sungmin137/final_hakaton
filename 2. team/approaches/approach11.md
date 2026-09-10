# 접근 11 — 쌍둥이 행 처리 + KIRC/KIPAN·GBMLGG/LGG 전용 서브모델

## 개요
KIRC(신장 투명세포암)/KIPAN(신장암 전체), GBMLGG(교모세포종+저등급 신경교종)/LGG(저등급 신경교종) 문제를 approach10(새 피처 추가)과 다른 방식으로 풀어보려는 시도임 — 피처를 더 얹는 대신, 데이터 정제와 모델 구조 자체를 바꿔봄. 배경은 `04_todo_day2.md` STEP 4 참고.

코드 폴더: `4. src/common/approach11_twin_handling/`


## 버전
| 버전 | 파일 (approach11_twin_handling/) | 정직 CV Macro F1 | 결과 | LB |
|---|---|---|---|---|
| v1 | approach11_v1_twin_removal.py | v4 0.4683 → 쌍둥이 행(1,005행) 학습 제거 0.4517 | 실패(-0.0166) | 미제출 — CV가 이미 크게 나빠져서 제출 근거 없음 |
| v2 | approach11_v2_specialist_submodel.py | 별도 측정(GBMLGG 비쌍둥이 92.0%, LGG 32.7%) | GBMLGG만 대폭 개선, LGG는 그대로 | 미제출 — 26클래스 전체 파이프라인이 아니라 2클래스 비교용 실험이라 test.csv 추론 단계 자체가 없음 |
| v3 | approach11_v3_specialist_submodel_balanced.py | 별도 측정(KIRC 20.7%, LGG 32.7%) | KIRC 소폭 개선, LGG는 여전히 그대로 | 미제출 — v2와 같은 이유(비교용 실험, 추론 단계 없음) |
| **v4** | approach11_v4_specialist_ensemble.py | v4 0.4683 → GBMLGG/LGG 라우팅 앙상블 **0.4695** | **+0.0012, 2회 재현 완전 일치** | 미제출 — 개선폭이 작고, 팀이 CV 개선이 LB를 예측 못 한다는 걸 이미 여러 번 확인함(`5. submissions/README.md` "CV ↔ LB 대응"). 실험 기록도 아직 정식으로 안 남아있어서 팀 논의 후 판단 예정 |

## 상세

### v1 — 쌍둥이 행 학습 제거 (`approach11_v1_twin_removal.py`)
기반은 approach1 v2+approach2 v1·v2(코드명 `v4`), 새 피처 없음. train에서 KIPAN/KIRC·GBMLGG/LGG 완전동일 프로필 행(1,005행)을 학습 데이터에서만 제거하고 학습함(검증은 원본 그대로 유지). fold당 학습 행이 4,960→4,130대로 급감, 정보 손실이 잡음 제거 이득보다 커서 4개 타겟 클래스·전체 전부 악화됨.

### v2 — KIRC/KIPAN, GBMLGG/LGG 전용 서브모델, 가중치 없음 (`approach11_v2_specialist_submodel.py`)
기반은 v4와 동일, 데이터 정제 없음. 26클래스 전체가 아니라 KIRC/KIPAN 두 클래스만, 또는 GBMLGG/LGG 두 클래스만 골라서 별도의 2-클래스 전용 모델을 새로 학습함(메인 모델과 무관, 비교용).

### v3 — 위 + 클래스 가중치 (`approach11_v3_specialist_submodel_balanced.py`)
v2와 동일 구성 + 전용모델 학습 시 `class_weights()`(`4. src/common/main.py`에 이미 있는 함수 재사용)를 `sample_weight`로 추가함. 다수 클래스(GBMLGG, KIPAN) 쪽으로 쏠리는 편향이 2클래스로 좁히니 더 심해지는 부작용이 있어서, v2·v3 둘 다 단독으로는 안 쓰고 v4의 재료로만 사용함.

### v4 — 라우팅 앙상블 (`approach11_v4_specialist_ensemble.py`)
기반 v4(메인 26클래스 모델) + v3의 GBMLGG/LGG 전용 서브모델(가중치 적용). 앙상블 규칙: 메인 모델의 1차 예측이 GBMLGG 또는 LGG일 때만, 그 최종 답을 전용 서브모델의 판단으로 덮어씀(그 외 24개 클래스는 메인 모델 그대로 둠). GBMLGG 정밀도 일부 희생(0.5052→0.4848), LGG는 크게 회복(0.4348→0.4870). 재실행 시각이 다른 두 번의 실행에서 소수점까지 완전히 동일한 결과 확인됨.

## 결론 — v4만 유의미한 신호, 나머지는 폐기
- v1(쌍둥이 행 제거)은 명확한 역효과라 폐기함.
- v2·v3(전용 서브모델 단독)은 단독으로 쓰지 않고 v4의 재료로만 사용함.
- v4(라우팅 앙상블)는 오늘 시도한 것 중 유일하게 재현 가능한 양(+)의 결과임. 다만 개선폭(+0.0012)이 작고, 팀이 이미 CV 개선이 LB 순위를 예측 못 한다는 걸 여러 번 확인한 상태라(`5. submissions/README.md` "CV ↔ LB 대응"), 실제 리더보드 제출은 실험 기록 보완과 팀 논의 후 판단할 예정임.

## 공용 코드 영향
`4. src/common/main.py`는 건드리지 않음 — v1~v4 전부 독립 스크립트로만 돌렸고, 메인 파이프라인 공용 파일에는 아무 분기도 추가하지 않음.
