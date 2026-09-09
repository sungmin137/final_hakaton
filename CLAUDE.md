# 프로젝트: 유전자 변이 기반 암 아형(SUBCLASS) 예측 해커톤

## 과제
- 26클래스 다중 분류. 입력 4,384개 유전자 변이 컬럼(WT/변이 문자열), train 6,201행, test 2,546행.
- 배경/데이터 상세: `docs/00_background.md`

## 대회 규칙 (docs/10_rules_compliance.md)
- 평가 Macro F1, Public = test 100%. 외부 데이터 **금지**. 사전학습 모델 허용. 1일 4회 제출.
- test 누수 = 수상 제외: 인코딩/스케일링 fit, get_dummies, test 통계로 결측 처리, 의사라벨링, test 분포 보고 후처리 조정 전부 금지.

## 작업 규칙
- **[최우선] test.csv는 최종 제출 추론 단계에서만 읽는다.** EDA, 전처리 통계, 인코더/스케일러 fit, 피처 선택, 검증, 의사라벨링, 분포 비교 등 어디에도 test 데이터를 사용하지 않는다. 사용자가 명시한 핵심 요구사항.
- 작업 흐름: 로컬 VSCode → git commit → GitHub push. 커밋은 작은 단위로 한글 메시지.
- **브랜치: 사용자(성민) 작업은 `sungmin` 브랜치에서 한다.** main에 직접 커밋하지 않는다. main 반영은 PR로. 팀원은 각자 이름 브랜치.
- 데이터(`data/`)와 모델(`models/`)은 git에 올리지 않는다. 제출 csv만 `submissions/`에 보관.
- 실험은 `experiments/`에 날짜_이름 형식으로 기록하고, 결과 요약은 `docs/experiments_log.md`에 누적.
- 검증은 반드시 **`--group-twins` 정직 CV**(StratifiedGroupKFold, 완전 동일 행을 같은 fold에). 일반 CV는 쌍둥이 때문에 +0.05 부풀려짐(docs/07). 리더보드 점수와 CV 점수를 함께 기록.
- 재사용 코드는 `src/`에, 탐색은 `notebooks/`에. 노트북 → src 승격 시 함수화.
- Python: `uv` 사용. 시스템 python3.14 + 기존 설치 패키지 사용 가능(pandas/sklearn/lgbm/xgb/catboost/optuna 확인됨).

## 메인 파이프라인
- 공식 베이스라인(`notebooks/00_official_baseline_xgb.ipynb`, XGBoost)의 5단계 구조를 따른다: Load → Preprocessing → Train → Inference → Submission.
- 구현체는 `src/main.py` (스크립트) 와 `notebooks/main.ipynb` (동일 로직을 셀로 나눈 것). 로직은 src에만 두고 노트북은 호출만 한다.
- 피처는 `src/features.py` + `FeatureMaker(kind)`. 새 피처 버전은 kind를 추가(v2, v3…)하고 `--features`로 선택.
- 실행: `PYTHONPATH=src python3 src/main.py --features v1 --cv [--submit]`. `--submit`이 test.csv를 읽는 유일한 경로.

## 스타일
- 응답은 한국어. 코드 주석은 간결하게.
- 결과 보고 시 CV 점수 표와 다음 실험 후보를 함께 제시.
