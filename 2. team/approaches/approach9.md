# 접근 9 — 유전자 역할 오버라이드 (TP53 등 6개, missense만)

##  번호 정정
리더보드 제출한 원래 파일명은 `approach4_gene_role_override_v3_20260909-1827.csv`였음.

- 실제 코드(`4. src/common/approach9_gene_override/features_v35_override.py` 원본 docstring)에도 명시돼 있음: `approach2 v1(sungmin 브랜치, features_v3_reference.py) + 유전자 역할 오버라이드`.
- 지금 시점(2026-09-10)에는 approach4~8까지 전부 다른 내용으로 채워져 있어서, 이 제출을 **approach9**로 재배정함.

## 개요
mutation_catalog.py의 role() 규칙(hotspot_ratio≥0.15 / lof_ratio≥0.30)이 TP53·NOTCH1·CTNNB1·KIT·EGFR·MET 6개 유전자를 전부 `passenger_like`로 잘못 분류하는 걸 발견함(`4. src/common/approach9_gene_override/known_gene_roles.py`의 `diff_with_catalog()`). 이 6유전자에 역할(dual/oncogene) 기반 오버라이드 피처를 얹어서 성민님의 approach2 v1(approach2_v1_20260909_1553) 위에 추가함.

| 구성 | 내용 |
|---|---|
| 기반 피처 | 성민님의 approach2 v1(성민님 코드 원본 그대로 복사: approach1 v2의 개수가중치 피처 + approach2 v1의 hotspot 위치·LoF 유전자·동반 조합·특수 그룹 피처) |
| 오버라이드 대상 | TP53(dual), NOTCH1(dual), CTNNB1(oncogene), KIT(oncogene), EGFR(oncogene), MET(oncogene) |
| 오버라이드 피처 | 유전자마다 `ovr_<gene>_missense`(missense 변이 존재 여부) 1개씩, 총 6개 |

**피처 중요도 실측(2026-09-10, `Day1/check_importance.py`)**: 원래는 `ovr_<gene>_any`, `ovr_<gene>_lof`도 만들었으나 기존 `g_<gene>`(그 유전자 변이 유무)과 완전히 중복돼 XGB가 거의 안 씀(순위 3,000위대/6,566개, 중요도≈0) — 삭제함. `ovr_<gene>_missense`만 실제로 채택됨(EGFR 19위, CTNNB1 17위, TP53 172위) — `g_<gene>`에는 없던 "missense 여부"라는 새 정보라서 살아남음.

## 버전
| 버전 | 파일 (src / submissions) | 구성 | 정직 CV Macro F1 | LB |
|---|---|---|---|---|
| v1 | approach9_v1_20260909_1827 (원래 이름: approach4_gene_role_override_v3_20260909-1827) | approach2 v1(sungmin) + 오버라이드 6개, 후처리 없음 | **0.4663~0.4681**(2회 재실행, 1차 0.4681 / 2차 0.4663) | **0.4031845152** (2026-09-09 18:35:37 제출) |
| (미제출) | — | approach9 v1 + class_scale 후처리 | base 0.4663 + 절반교차 평균 개선 +0.0076 → **0.4739 추정** | 제출 안 함 |

class_scale 재검증 상세(`Day1/train_v35.py`가 OOF 확률 저장 → `Day1/postprocess_reference.py`, 팀 `postprocess.py` 미수정 사본, 의 `cross_check()`): OOF를 절반씩 나눠 한쪽으로 배율 학습 → 다른 쪽으로 평가함. 결과 +0.0102, +0.0051(평균 +0.0076, 두 번 다 양수라 실제 일반화 개선으로 판단). twin_rule은 정직 CV(그룹 fold) 구조상 애초에 검증 불가 — 쌍둥이가 항상 같은 fold에 있어 실제 제출로만 효과 확인 가능.

## 결론 — CV는 높은데 LB는 더 낮았음

**아래 표는 "오버라이드 얹기 전/후" 비교가 아님.** approach2 v1(오버라이드를 얹기 전, 기반 피처)은 미제출이라 LB 점수 자체가 없어서 직접 비교가 불가능함. 대신 그날(9/9) 실제 LB가 있던 다른 제출(approach1 v2)을 기준점 삼아 "이날 CV가 LB를 얼마나 못 맞혔는지"를 보여주는 것뿐임.

| | 정직 CV | 실제 LB |
|---|---|---|
| approach1 v2(개수가중치, `submission.csv`, 후처리 없음) | 0.4486 | 0.41498 |
| **approach9(후처리 없음)** | **0.4681**(approach1 v2보다 높음) | **0.40318**(approach1 v2보다 낮음!) |

CV는 approach1 v2보다 높은데 LB는 approach1 v2보다 낮음(격차 -0.065, 제출 당일인 9/9 기준 가장 컸던 역전). 참고로 다음날(9/10) approach8에서 더 큰 격차(-0.084)가 나옴 — approach8.md 참고.

추정 원인 두 가지:
1. **후처리 누락**: 팀 다른 제출(approach2, approach3)은 모두 `twin_rule`·`class_scale` 적용 상태였고, 이 후처리들이 정확히 test에서 알려진 문제(STES 과잉예측, 쌍둥이 행)를 겨냥한 보정 장치라 이게 빠지면 그대로 점수에 반영됨.
2. **train 특유 패턴 과적합 가능성**: hotspot/오버라이드 피처는 train 안의 특정 위치·유전자를 촘촘히 기억하는 방식인데 test는 이미 분포가 다름(초과변이 샘플 3배) — 촘촘한 피처일수록 분포 변화에 더 취약할 수 있음.

**판단**: "오버라이드 피처 자체가 나쁘다"보다는 "후처리 없이 성급하게 제출했다"에 더 가까움 — 같은 피처에 twin_rule·class_scale까지 적용한 재비교가 안 된 상태라 원인이 완전히 분리되지 않았음. 채택하지 않음. 제출 기준선은 approach2 v2(approach1 v2 + approach2 v1·v2 통합 피처 + 클래스 배율 + 쌍둥이 규칙, LB 0.4369) 유지.

- 코드: `4. src/common/approach9_gene_override/`(features_v3_reference.py, features_v35_override.py, count_weights.py, known_gene_roles.py — 원본과 diff 0)
- 진입 스크립트: `4. src/submissions_source/approach9_v1_20260909_1827.py` (경로만 수정)

**⚠️ 이 코드를 다시 돌려도 `5. submissions/approach9_v1_20260909_1827.csv`가 똑같이 나온다고 보장 못함.** 실제로 재실행해보니 원본과 92.5%만 일치(2,546건 중 191건 예측 다름) — XGBoost `tree_method="hist"` + `n_jobs=8` 조합이 `random_state` 고정에도 완전히 결정론적이지 않은 것으로 보임(회원님이 원래 두 번 돌렸을 때도 CV 0.4681→0.4663으로 갈렸던 것과 같은 현상, 표 위 CV 값 참고). 그래서 이 csv는 **재실행 결과가 아니라 2026-09-09 18:35:37에 실제 제출했던 원본 파일을 그대로** 넣은 것이고(byte 단위 diff로 원본과 동일함 확인), 코드는 "무엇을 했는지" 보여주는 참고용으로만 씀.

**⚠️ `6. experiments/submissions/approach9_v1_20260909_1827/`(test_proba.npy)는 의도적으로 없음.** 다른 approach들은 팀 공용 `make_submission.py`가 제출 시 test 확률을 자동 저장하지만, approach9는 이미 점수를 받은 원본 코드(`Day1/predict_v35.py`)를 그대로 보존한 것이라 그 파이프라인을 거치지 않음. 지금 다시 돌려서 만들어도 위에서 확인했듯 원본과 92.5%만 일치하는 확률이라 원본 제출을 대표하지 못해서 일부러 만들지 않음.
