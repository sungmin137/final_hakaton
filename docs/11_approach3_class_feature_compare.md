# 접근 3 — 각 클래스 특징별 비교 (Approach 3: Class Feature Compare) — 암종마다 특징에 맞는 접근을 쓰는 방법

> 아이디어(성민, 2026-09-09): 시각화에서 본 대로 암종마다 신호의 성격이 다르다.
> driver 표지가 강한 암종은 driver로, 변이가 많은 암종은 변이 개수로 접근하자.

## 버전
| 버전 | 구성 | 정직 CV Macro F1 | 비고 |
|---|---|---|---|
| v1 | 전문가 3개(driver / burden / full) + 로지스틱 회귀 스태킹 | (실행 중) | `src/approach3_class_feature_compare/approach3_class_feature_compare_v1.py`, 산출물 `experiments/approach3_class_feature_compare_v1/` |

## v1 설계
| 전문가 | 피처 | 역할 |
|---|---|---|
| driver | `g_*` 유전자 변이 유무, `hs_*` hotspot 위치, `lof_*` 기능상실 유전자, `combo_*` 동반 조합 | driver 비율이 높은 암종 |
| burden | `n_*` 변이 개수, 유형 비율, 개수 구간, `is_*` 무변이·초과변이 플래그, `kf_*` 치환 심각도 집계 | 변이 개수로 갈리는 암종 |
| full | v4 전체 | 기준선 |

- 각 전문가는 같은 정직 5-Fold(쌍둥이 같은 fold)로 OOF 확률을 만든다.
- 메타 모델은 세 전문가의 로그 확률(26×3=78)을 입력으로 다항 로지스틱 회귀. 같은 fold로 교차 예측.
- 하드 라우팅(그룹 먼저 맞히고 그룹별 모델) 대신 스태킹을 쓴 이유: 라우팅 실수가 그대로 오답이 되는 것을 피하기 위해.
- 산출물 `per_class_f1.csv`: 암종별로 어느 전문가가 가장 잘 맞히는지 → "이 암종은 driver, 이 암종은 개수" 표.

## 다음 버전 후보
- v2: 전문가 추가 — hotspot 전용(변이 위치만), 접근1 개수가중치 점수 전용(`cw_*`)
- v3: 메타 모델을 XGB로 교체, 또는 암종별 가중치를 직접 최적화(Macro F1 목표)
- v4: 표지가 약한 암종(SARC, PRAD, CESC, PCPG, THYM) 전용 이진 전문가
