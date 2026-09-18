# 접근 23 — GBMLGG/LGG synthetic data augmentation (train만 사용)

## 개요
train만으로 GBMLGG/LGG synthetic 샘플을 만들어 소규모(+50) 통제 실험으로 효과를 검증. 외부 데이터·biological prior 없음. 26클래스 전체가 아니라 GBMLGG/LGG 두 클래스만 대상. 코드 폴더: `4. src/common/approach23_synthetic_aug/`.

기존 approach11(문헌 마커 저차원 서브모델 라우팅)과는 완전히 독립된 축이며, 그 코드/구조를 건드리지 않았다.

## Synthetic 생성 방법 — 2-도너 crossover 교란
후보 비교(구현 전 검토):

| 방법 | burden 보존 | co-mutation 보존 | 난이도 | 비고 |
|---|---|---|---|---|
| 유전자별 독립 확률 샘플링 | ✗ | ✗ | 낮음 | 사용 금지(요구사항) |
| 실제 행 단순 bootstrap | ✓ | ✓ | 매우 낮음 | 완전 중복 발생 → 사용 불가 |
| **2-도너 crossover(채택)** | ✓ | ✓(대부분 유지) | 낮음 | 아래 설명 |
| SMOTE류 연속값 보간 후 재이산화 | △ | △ | 중간 | 임계값 왜곡 위험 |

**채택안**: 같은 클래스의 실제 행 A(베이스)·B를 무작위로 뽑아, A·B가 다른 유전자 중 20%만 B의 실제 토큰 값으로 치환. A의 co-mutation 블록 대부분이 그대로 유지되고 burden은 A·B 사이 값으로 자연스럽게 떨어진다. 실제 train과 완전동일하면 재추첨(최대 20회). 구현: `synthetic_gen.py::generate_synthetic()`.

## 데이터 누수 방지
- synthetic은 **그 fold의 학습 파티션(`tr_part`)의 GBMLGG/LGG 행에서만** 생성. 검증 파티션(`va_part`)·test는 생성 과정에 전혀 관여하지 않음.
- `FeatureMaker`는 **실제 `tr_part`만으로 fit**(synthetic 제외) → 4개 조건(baseline/GBMLGG+50/LGG+50/both+50)이 동일한 피처 공간·동일한 count-weight 통계를 공유. synthetic 행은 그 fm으로 `transform`만 해서 학습 행으로 추가 — 통계 fit에는 영향을 주지 않음.
- 정직 CV(`--group-twins`, `StratifiedGroupKFold`)를 그대로 사용해 쌍둥이 정보 누수 없음.

## v1 — +50 통제 실험

**설정**: features `v4` + params `mild_col`(XGB, colsample_bytree 0.7) 단일 모델. seed 3개(42/7/123) × 5-fold 정직 CV. 4조건: baseline / GBMLGG+50 / LGG+50 / both(GBMLGG+50+LGG+50). 실행: `approach23_v1_run_experiment.py`.

이 baseline은 approach14 v3(LB 0.4564122, 리더보드 최종 후보)의 CV가 아니라, approach11 v11 통합테스트와 같은 조건의 **단일 모델** 기준선이다(0.4595 근사, 시드에 따라 0.4536~0.4669). v4+mild_col 단독은 LB 미제출 — 가장 가까운 실제 제출은 v7(+클래스 배율, LB 0.4377).

### 결과 (3 seed 평균 ± 표준편차)

| Experiment | GBMLGG F1 | LGG F1 | Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.5098 ± 0.0177 | 0.3894 ± 0.0189 | 0.4605 ± 0.0067 |
| GBMLGG +50 | 0.5625 ± 0.0071 | 0.1212 ± 0.0513 | 0.4510 ± 0.0089 |
| LGG +50 | 0.3954 ± 0.0211 | 0.5738 ± 0.0108 | 0.4596 ± 0.0064 |
| Both +50 | 0.4780 ± 0.0155 | 0.4878 ± 0.0099 | 0.4639 ± 0.0034 |

원본 결과: `6. experiments/2026-09-14_approach23_synthetic_gbmlgg_lgg/{per_seed_results.csv, summary.csv}`.

### 해석 — 세 가지를 구분

1. **GBMLGG 성능**: GBMLGG+50 단독일 때 +0.053(3 seed 전부 같은 방향, 표준편차 0.007로 매우 안정) — 명확한 개선. Both일 때는 −0.032(2배 이상 표준편차, 트레이드오프로 소폭 손해).
2. **LGG 성능**: GBMLGG+50 단독은 LGG를 0.39→0.12로 **붕괴**시킴(3 seed 전부, −0.27 이상). LGG+50 단독은 +0.184로 크게 개선. Both일 때는 +0.098로 GBMLGG+50의 부작용 없이 개선 유지.
3. **전체 Macro F1**: GBMLGG+50 단독은 −0.0095(명확한 악화). LGG+50 단독은 −0.0009(표준편차 안, 사실상 변화 없음). **Both+50만 +0.0034로 개선**이나, 이 값 자체는 baseline 표준편차(0.0067)의 절반 수준이라 "명확한 개선"이라 단정하기는 이르다.

**결론**: "synthetic이 도움이 됐다"고 단순화할 수 없음. 실제로 일어난 일은 **GBMLGG를 늘리면 모델이 애매한 행을 GBMLGG 쪽으로 더 강하게 밀어붙여 LGG를 거의 못 맞히게 되고, LGG를 늘리면 그 반대가 일어나는 뚜렷한 트레이드오프**다. Both(균형 augmentation)일 때만 macro가 개선되지만, 그 개선폭(+0.0034)은 GBMLGG 손해(−0.032)와 LGG 이득(+0.098)이 상쇄된 결과이지 "모두 좋아진" 게 아니다. 이 패턴은 이미 approach11에서 확인된 사실(GBMLGG/LGG가 마커 공간에서 잘 구분 안 되는 행이 존재)과 일치한다.

**다른 24개 클래스 영향(근사 확인)**: macro F1 = 26개 클래스 F1 평균이므로, GBMLGG/LGG 두 클래스의 F1 변화만으로 기대되는 macro 변화를 역산해 실제 관측치와 비교함 — both_synth50 기준 기대 +0.00256 vs 실제 +0.0034(차이 0.0008), gbmlgg_synth50 기대 −0.0083 vs 실제 −0.0095(차이 0.0012), lgg_synth50 기대 +0.0027 vs 실제 −0.0009(차이 0.0036). 차이가 전부 한 자릿수 클래스 노이즈 수준이라 **다른 24개 클래스에 유의미한 부작용은 없어 보임**(다만 26클래스 전체 F1 테이블을 직접 저장하지는 않았음 — 필요하면 재실행해서 확인 가능).

**CV↔LB 무상관 재확인 필요**: 팀이 approach11에서 이미 여러 번 확인했듯(v4/v5/v6/v11), 정직 CV의 개선이 LB로 이어지지 않는 경우가 많았음. Both+50의 +0.0034도 표준편차 절반 수준의 작은 신호라 **LB로 검증하기 전까지는 결론을 유보**해야 함.

### Sanity check 결과 (전체 train GBMLGG/LGG 풀 기반, seed=42, `approach23_v1_sanity_check.py`)

- **A. duplicate**: GBMLGG/LGG 각 50개 중 실제 train과 완전동일 0개, synthetic 내부 중복 0개.
- **B. mutation burden**: GBMLGG real mean 12.54(std 8.32) → synthetic mean 14.18(std 14.04, 다소 확대). LGG real mean 8.69(std 8.52) → synthetic mean 8.98(std 4.98, 오히려 축소). 두 클래스 모두 median·min/max가 real 분포 범위 안에 있음.
- **C. gene frequency**: 상위 15개 유전자 기준 diff가 GBMLGG는 최대 0.089(HNF4A), LGG는 최대 0.078(ATRX) — 표본 50개 대비 자연스러운 변동 범위.
- **D. class identity**: synthetic GBMLGG는 자기 centroid과 코사인 유사도 0.838(반대 클래스 0.766)로 자기 클래스에 더 가까움. synthetic LGG도 0.941 vs 0.857로 정상. 이상 없음.
- **E. sparsity**: GBMLGG 밀도 1.13배, LGG 밀도 1.03배 — 과도하게 dense해지지 않음.

## 결론(v1 시점)
- GBMLGG만 늘리거나 LGG만 늘리는 편향된 augmentation은 **명확히 해롭다**(반대 클래스 붕괴).
- 균형 잡힌 augmentation(둘 다 +50)만 macro F1 개선 방향이나, 개선폭이 노이즈 경계에 걸쳐 있어 **"성공"이라 단정할 수 없음**.

## v2-B — burden 8~15로 도너 풀 제한 (`approach23_v2b_run_experiment_burden.py`)
approach14 v3 OOF 오답 분석(`approach23_v2_error_analysis.py`)에서 GBMLGG/LGG 오답이 burden 8~15 경계에 몰린다는 관찰을 근거로, synthetic 도너를 그 fold의 tr_part 안에서 burden 8~15인 행으로만 제한해 v1과 동일한 4조건(baseline/GBMLGG+50/LGG+50/both+50, seed 3개, features v4+mild_col)을 재실행.

| Experiment | v1(클래스 전체 도너) | v2-B(burden 8~15 도너) | 판단 |
|---|---:|---:|---|
| GBMLGG+50 macro F1 | 0.4510 | 0.4525 | 소폭 개선 |
| LGG+50 macro F1 | 0.4596 | 0.4588 | 사실상 동일 |
| **Both+50 macro F1** | **0.4639** | **0.4604** | **명확히 후퇴** |
| Both+50 LGG F1 (mean±std) | 0.4878±0.0099 | 0.4566±0.0320 | seed 표준편차가 3배 이상 커짐 |

**결론: burden 8~15 donor 제한 가설은 폐기한다.** 다만 이것은 "synthetic augmentation 전체의 실패"가 아니다 — 실제로 관찰된 것은:
1. **한쪽만 augment할 때의 부작용 완화는 방향성 자체는 맞았다**: GBMLGG+50의 LGG 붕괴(0.1212→0.1736), LGG+50의 GBMLGG 손해(0.3954→0.4202) 둘 다 burden 제한으로 완화됨.
2. **그런데 우리가 실제로 원했던 both+50에서는 그 이득이 사라지고 오히려 후퇴했다.** 원인으로 추정되는 것은 도너 풀 축소(LGG 도너가 fold당 ~183개→~86개로 절반)로 인한 **synthetic 다양성 손실** — LGG F1의 seed 간 표준편차가 0.0099→0.0320으로 3배 이상 커진 것이 이 추정을 뒷받침한다. 즉 "도너를 좁혀 purity를 얻는 대신 diversity를 희생했을 가능성"이 있다는 trade-off 신호로 기록한다.
3. v1(클래스 전체 도너)은 기존 후보로 남겨두되, 새로운 근거 없이 burden 경계(6-12, 7-15, 8-20 등)를 계속 탐색하는 hyperparameter sweep은 하지 않는다.

## 현재 상태(2026-09-14)
**GBMLGG/LGG synthetic 축은 잠정 보류.** 추가 실험(donor burden 재튜닝, +100/+200 규모 확대, swap_frac 스윕 등)은 새로운 근거가 나오기 전까지 중단한다. 대신 전체 접근에 대한 anomaly-driven diagnostic scan(`approach23_v4_anomaly_scan.py`, `3. docs/08_anomaly_scan_2026-09-14.md`)으로 우선순위를 재검토했다.

## approach23/anomaly-scan 축 — 완전 종료 (2026-09-14)

synthetic augmentation(v1/v2-B)에서 시작해 anomaly-driven diagnostic scan(class/burden/confusion/CV-LB 메타패턴), test population 구조 분석(4유전자 조합, SVD 클러스터링, CACNA1A/CPEB2 축), 9차→v2 conditional routing, 그리고 hypermut_rule 템플릿을 전체 burden 범위에 적용한 재탐색(COAD 61-75 후보)까지 순서대로 진행했다. 최종 정리:

| 축 | 결론 |
|---|---|
| GBMLGG/LGG synthetic (v1, v2-B) | 보류 — burden 8-15 donor 제한 가설 폐기, v1은 후보로 남김 |
| KIRC/KIPAN | 보류 — twin_rule이 이미 대부분 처리(판정 A), burden 메커니즘은 GBMLGG/LGG와 달리 적용 안 됨 |
| burden=0/1-3 | 폐기 — THYM 제외해도 나머지 클래스 구분력 0, 정보 자체가 없음 |
| BRCA/KIPAN sink | 기록만 — OOF에서는 강하나 test 전이 근거 약함 |
| TP53&APC, CACNA1A/CPEB2 | 폐기 — 각각 B(근거 약함)·B(test-only 인플레이션, 암종 무관) 판정 |
| 31-100 NN 분류기 | 폐기 — 기존 모델보다 훨씬 약함, 모든 alpha에서 손해 |
| 9차→v2 conditional routing | 폐기 — 라우터 정확도 54~60%(노이즈 수준), OOF 개선 +0.0007에 불과 |
| hypermut_rule 템플릿 전체 burden 재탐색(COAD 61-75) | 폐기 — OOF에서 0회 재현, 유일한 신규 후보였으나 검증 불가 |

**최종 결론**: 이 프로젝트의 train 데이터 안에서 추가로 발굴 가능한 LB 레버는 이번 탐색 범위 안에서 찾지 못했다. 유일하게 실증된 성공 메커니즘은 성민님의 21차(hypermut_rule)가 보여준 3조건(test-only 분포 이동 + train 근거가 절대적(0에 가까움) + train에 자연스러운 대체 클래스 존재)이며, 이번 재탐색으로 그 조건을 만족하는 두 번째 사례가 없음을 확인했다. **approach23/anomaly-scan 축은 여기서 종료한다.**
