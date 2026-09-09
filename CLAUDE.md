# 프로젝트: 유전자 변이 기반 암 아형(SUBCLASS) 예측 해커톤

## 과제
- 26클래스 다중 분류. 입력 4,384개 유전자 변이 컬럼(WT/변이 문자열), train 6,201행, test 2,546행.
- 배경/데이터 상세: `docs/00_background.md`

## 작업 규칙
- 작업 흐름: 로컬 VSCode → git commit → GitHub push. 커밋은 작은 단위로 한글 메시지.
- 데이터(`data/`)와 모델(`models/`)은 git에 올리지 않는다. 제출 csv만 `submissions/`에 보관.
- 실험은 `experiments/`에 날짜_이름 형식으로 기록하고, 결과 요약은 `docs/experiments_log.md`에 누적.
- 검증은 반드시 Stratified K-Fold(클래스 26개, 불균형 가능성 큼). 리더보드 점수와 CV 점수를 함께 기록.
- 재사용 코드는 `src/`에, 탐색은 `notebooks/`에. 노트북 → src 승격 시 함수화.
- Python: `uv` 사용. 시스템 python3.14 + 기존 설치 패키지 사용 가능(pandas/sklearn/lgbm/xgb/catboost/optuna 확인됨).

## 스타일
- 응답은 한국어. 코드 주석은 간결하게.
- 결과 보고 시 CV 점수 표와 다음 실험 후보를 함께 제시.
