# 대회 규칙 및 준수 체크리스트 (2026-09-09 규칙 수령)

## 규칙 원문 요약
| 항목 | 내용 |
|---|---|
| 평가 | **Macro F1**. Public = test 100% (리더보드 점수가 곧 최종) |
| 참여 | 개인/팀 |
| 외부 데이터 | **사용 불가** |
| 사전 학습 모델 | 사용 가능 |
| 제출 | 1일 최대 4회 |
| Data Leakage (수상 제외) | test를 학습에 활용. 예: test로 label/one-hot 인코딩 fit, test로 스케일링, test에 get_dummies, **test 통계값으로 test 결측 처리**, 그 외 test가 학습에 쓰이는 모든 경우 |

## 코드 준수 점검 (sungmin 브랜치, 2026-09-09 기준)
| 규칙 | 우리 코드 | 근거 위치 | 판정 |
|---|---|---|---|
| test를 학습에 사용 금지 | test.csv는 `load_test()` 한 곳에서만 읽고, 호출은 추론 함수 안에서만 | `src/main.py` load_test / fit_full_and_submit, `src/make_submission.py` | ✅ |
| 인코딩·스케일링 fit에 test 금지 | 라벨 인코더·컬럼 목록·hotspot 목록·가중치 전부 `fit(train)`에서만 결정. test는 `transform`만 | `FeatureMaker.fit/transform`, `InsightFeatures.fit`, `CountWeightModel.fit`, `variant_matrix(vocab=…)` | ✅ |
| test 결측을 test 통계로 처리 금지 | test 결측 237셀을 **상수 "WT"** 로 채움 (test 통계 미사용) | `main.load_test`의 `fillna("WT")` | ✅ |
| get_dummies를 test에 적용 금지 | 사용 안 함. 이진화는 `!= "WT"` 행 단위 연산 | `features.build_features` | ✅ |
| 외부 데이터 금지 | 사용 안 함. 유전자 역할·hotspot·조합은 전부 train에서 계산 | `docs/03`, `docs/04`, `src/analysis/mutation_catalog.py` | ✅ |
| CV/피처 선택에 test 미사용 | 정직 CV·내부 OOF 모두 train만 | `cross_validate`, `CountWeightFeatures.fit` | ✅ |
| 추론 후 test 관찰(분포·결측 수) | 관찰만 기록, **모델·피처·후처리 조정에 사용하지 않음** | `docs/08` | ✅ (유지 필요) |
| 쌍둥이 규칙(`--twin-rule`) | train 행과 test 행의 완전 일치를 찾아 라벨 뒤집기. test로 학습하지 않는 1-NN 성격의 추론 규칙 | `src/postprocess/twin_rule.py` | ⚠️ 규칙 위반은 아니나 회색지대. 팀 판단·설명 가능해야 함. 기본 꺼짐 |

| 클래스 배율 후처리(`--class-scale`) | train의 정직 CV OOF 확률 + train 라벨로만 배율 결정. test 확률·분포 미사용 | `src/postprocess/class_scale.py` | ✅ |

## 앞으로 지켜야 할 것
1. test 분포를 보고 임계값·클래스 비율·후처리를 조정하지 않는다 (예: "STES 21%가 많으니 줄이자" ❌).
2. 의사라벨링(pseudo-labeling) 금지 — test 예측을 다시 학습에 넣는 것은 명백한 leakage.
3. 외부 데이터 금지 → docs/09의 상위권 방법(TCGA 병합)은 **사용 불가**.
4. 하루 4회 제출 → 제출 전 정직 CV로 선별. 제출 이력은 `docs/experiments_log.md`에 기록.
5. 사전 학습 모델은 허용 → 단백질 언어모델(ESM 등) 임베딩은 규칙상 가능. 후순위 검토.

## "외부 데이터" 해석 (2026-09-09 팀 합의안)
| 구분 | 예시 | 판단 |
|---|---|---|
| ❌ 외부 데이터셋 | TCGA MC3/cBioPortal 환자 변이 기록, COSMIC 변이 DB, 다른 대회 데이터 | 금지. 환자 단위 데이터로 test 정답 조회 가능 |
| ⚠️ 큐레이션 목록 | COSMIC Cancer Gene Census, OncoKB 유전자 목록 | 회색. 데이터베이스 다운로드 형태라 피하는 쪽으로 |
| ✅ 도메인 지식 | 논문·교과서의 원리, BLOSUM62 치환 행렬, 아미노산 물리화학 특성(소수성·전하·부피), 유전자 기능 분류(oncogene/TSG) 상식 | 허용으로 해석. 피처 설계에 반영. DACON 게시판에도 BLOSUM62 활용 코드가 공개 공유됨 |

원칙: **환자·샘플 단위 데이터는 어떤 형태로도 들여오지 않는다.** 지식 기반 피처는 출처(논문·행렬 이름)를 문서에 남겨 발표에서 설명 가능하게 한다.
구현: `src/approach2_knowledge/knowledge_features.py` (BLOSUM62 + 아미노산 특성 변화량). 출처: Henikoff & Henikoff 1992 (BLOSUM62), Kyte & Doolittle 1982 (소수성), Zamyatnin 1972 (잔기 부피).
