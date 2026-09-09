# 프로젝트: 유전자 변이 기반 암 아형(SUBCLASS) 예측 해커톤

## 과제
- 26클래스 다중 분류. 입력 4,384개 유전자 변이 컬럼(WT/변이 문자열), train 6,201행, test 2,546행.
- 해커톤 정보: `1. info/README.md`, `1. info/background.md`, 데이터: `1. info/data/README.md` (csv는 git 제외)

## 대회 규칙 (2. team/rules_compliance.md)
- 평가 Macro F1, Public = test 100%. 외부 데이터 **금지**. 사전학습 모델 허용. 1일 4회 제출.
- test 누수 = 수상 제외: 인코딩/스케일링 fit, get_dummies, test 통계로 결측 처리, 의사라벨링, test 분포 보고 후처리 조정 전부 금지.

## 작업 규칙
- **[최우선] test.csv는 최종 제출 추론 단계에서만 읽는다.** EDA, 전처리 통계, 인코더/스케일러 fit, 피처 선택, 검증, 의사라벨링, 분포 비교 등 어디에도 test 데이터를 사용하지 않는다. 사용자가 명시한 핵심 요구사항.
- 작업 흐름: 로컬 VSCode → git commit → GitHub push. 커밋은 작은 단위로 한글 메시지.
- **브랜치: 사용자(성민) 작업은 `sungmin` 브랜치에서 한다.** main에 직접 커밋하지 않는다. main 반영은 PR로. 팀원은 각자 이름 브랜치.
- 데이터(`1. info/data/*.csv`)와 모델(`models/`)은 git에 올리지 않는다. 제출 csv는 `5. submissions/`에 보관.
- 실험은 `6. experiments/`에 날짜_이름 형식으로 기록하고, 결과 요약은 `3. docs/experiments_log.md`에 누적.
- 검증은 반드시 **`--group-twins` 정직 CV**. 일반 CV는 쌍둥이 때문에 +0.05 부풀려짐(3. docs/04_duplicate_twins.md). CV 개선이 LB로 거의 안 옮겨지므로(3. docs/05_test_inference.md) 제출 기준선은 접근2 v2(LB 0.43). 팀 결정은 `2. team/decisions.md`.
- Python: `uv` 사용. 시스템 python3.14 + 기존 설치 패키지 사용 가능(pandas/sklearn/lgbm/xgb/catboost/optuna 확인됨).

## 메인 파이프라인
- 공식 베이스라인(`1. info/baseline.py`, `1. info/baseline.py`)의 5단계 구조를 따른다: Load → Preprocessing → Train → Inference → Submission.
- 구현체는 `4. src/common/main.py` .
- **파일 규칙은 `2. team/file_rules.md`가 기준.** 제출 재현 스크립트 `4. src/submissions/approachN_vK_YYYYMMDD_HHMM.py` = 같은 이름의 `5. submissions/….csv`. 공용 로직은 `4. src/common/`. 접근법 설명은 `2. team/approaches/approachN.md`에 버전별로 상세 기록.
- 피처는 `4. src/common/features/features.py` + `FeatureMaker(kind)`. import는 `PYTHONPATH="4. src/common"` 기준(`features.features`, `approach1_count_weight.count_weights` …).
- 실행: `PYTHONPATH="4. src/common" python3 "4. src/common/main.py" --features v4 --cv --group-twins`. test.csv는 `make_submission.py`(진입 스크립트 경유)에서만 읽는다.

## 스타일
- 응답은 한국어. 코드 주석은 간결하게.
- 결과 보고 시 CV 점수 표와 다음 실험 후보를 함께 제시.
