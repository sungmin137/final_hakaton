# 대회 규칙 준수 체크리스트

규칙 원문은 `1. info/README.md`, 팀 결정은 `decisions.md`.

## 코드 준수 점검 (sungmin 브랜치, 2026-09-09 기준)
| 규칙 | 우리 코드 | 근거 위치 | 판정 |
|---|---|---|---|
| test를 학습에 사용 금지 | test.csv는 `load_test()` 한 곳에서만 읽고, 호출은 추론 함수 안에서만 | `4. src/common/main.py` load_test / fit_full_and_submit, `4. src/common/make_submission.py` | ✅ |
| 인코딩·스케일링 fit에 test 금지 | 라벨 인코더·컬럼 목록·hotspot 목록·가중치 전부 `fit(train)`에서만 결정. test는 `transform`만 | `FeatureMaker.fit/transform`, `InsightFeatures.fit`, `CountWeightModel.fit`, `variant_matrix(vocab=…)` | ✅ |
| test 결측을 test 통계로 처리 금지 | test 결측 237셀을 **상수 "WT"** 로 채움 (test 통계 미사용) | `main.load_test`의 `fillna("WT")` | ✅ |
| get_dummies를 test에 적용 금지 | 사용 안 함. 이진화는 `!= "WT"` 행 단위 연산 | `features.build_features` | ✅ |
| 외부 데이터 금지 | 사용 안 함. 유전자 역할·hotspot·조합은 전부 train에서 계산 | `3. docs/03`, `3. docs/04`, `4. src/common/analysis/mutation_catalog.py` | ✅ |
| CV/피처 선택에 test 미사용 | 정직 CV·내부 OOF 모두 train만 | `cross_validate`, `CountWeightFeatures.fit` | ✅ |
| 추론 후 test 관찰(분포·결측 수) | 관찰만 기록, **모델·피처·후처리 조정에 사용하지 않음** | `3. docs/05_test_inference.md` | ✅ (유지 필요) |
| 쌍둥이 규칙(`--twin-rule`) | train 행과 test 행의 완전 일치를 찾아 라벨 뒤집기. test로 학습하지 않는 1-NN 성격의 추론 규칙 | `4. src/common/postprocess/twin_rule.py` | ⚠️ 규칙 위반은 아니나 회색지대. 팀 판단·설명 가능해야 함. 기본 꺼짐 |

| 클래스 배율 후처리(`--class-scale`) | train의 정직 CV OOF 확률 + train 라벨로만 배율 결정. test 확률·분포 미사용 | `4. src/common/postprocess/class_scale.py` | ✅ |

## "외부 데이터" 해석 (2026-09-09 팀 합의안)
| 구분 | 예시 | 판단 |
|---|---|---|
| ❌ 외부 데이터셋 | TCGA MC3/cBioPortal 환자 변이 기록, COSMIC 변이 DB, 다른 대회 데이터 | 금지. 환자 단위 데이터로 test 정답 조회 가능 |
| ⚠️ 큐레이션 목록 | COSMIC Cancer Gene Census, OncoKB 유전자 목록 | 회색. 데이터베이스 다운로드 형태라 피하는 쪽으로 |
| ✅ 도메인 지식 | 논문·교과서의 원리, BLOSUM62 치환 행렬, 아미노산 물리화학 특성(소수성·전하·부피), 유전자 기능 분류(oncogene/TSG) 상식 | 허용으로 해석. 피처 설계에 반영. DACON 게시판에도 BLOSUM62 활용 코드가 공개 공유됨 |

원칙: **환자·샘플 단위 데이터는 어떤 형태로도 들여오지 않는다.** 지식 기반 피처는 출처(논문·행렬 이름)를 문서에 남겨 발표에서 설명 가능하게 한다.
구현: `4. src/common/approach2_knowledge/knowledge_features.py` (BLOSUM62 + 아미노산 특성 변화량). 출처: Henikoff & Henikoff 1992 (BLOSUM62), Kyte & Doolittle 1982 (소수성), Zamyatnin 1972 (잔기 부피).
