# 제공 데이터 (1. info/data/, csv는 git 제외)

| 파일 | 행 | 열 | 내용 |
|---|---|---|---|
| train.csv | 6,201 | 4,386 | `ID`, `SUBCLASS`(정답, 26종), 유전자 4,384개 |
| test.csv | 2,546 | 4,385 | `ID`, 유전자 4,384개 (train과 동일 컬럼). **최종 추론에서만 읽는다** |
| sample_submission.csv | 2,546 | 2 | `ID`, `SUBCLASS` — 제출 형식. ID 순서는 test와 동일 |

## 값의 의미
- 각 행 = 환자 1명의 종양 시퀀싱 결과 요약. 각 열 = 유전자 하나.
- `WT` = wild type(변이 없음). 그 외는 단백질 변이 표기(HGVS): `V600E`(missense), `D623D`(동의), `R213*`(종결), `K16fs`(프레임시프트). 여러 변이는 공백으로 나열.
- SUBCLASS = TCGA 암종 코드 26개 (BRCA 유방암, LGG 저등급 신경교종 …). 상위·하위 집합이 공존: GBMLGG ⊃ LGG, KIPAN ⊃ KIRC.

## train에서 확인된 특징 (3. docs/01~04)
- WT 비율 99.2%. 전부 WT인 유전자 154개. 결측 없음.
- 클래스 불균형: BRCA 786 ~ DLBC 38.
- **완전 동일 프로필 중복**: KIPAN↔KIRC 276쌍, GBMLGG↔LGG 173쌍 (같은 환자가 두 라벨). 검증은 반드시 쌍둥이를 같은 fold에.
- 변이 0개 행 94개(THYM·THCA·LAML), 300개 초과 초과변이 101명(SKCM·STES·UCEC·COAD).


## test 관찰
추론 단계에서 관찰한 test 특징(결측, 동일 행, 분포 차이)은 `3. docs/05_test_inference.md`. 모델 조정에는 쓰지 않는다.
