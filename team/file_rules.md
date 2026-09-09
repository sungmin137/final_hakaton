# 파일 생성 규칙

## 1. 접근법 설명 문서 (md)
- 위치: `team/approaches/approachN.md` (N = 접근법 번호)
- **새 접근법을 시도할 때마다 새 md 파일**을 만든다. 파일 하나 = 접근법 하나.
- 내용: 접근법 개요(아이디어, 왜 시도하는지) + **버전별 상세 설명**. 버전이 올라갈 때마다 그 버전 섹션을 추가한다.
- 각 버전 섹션에 반드시 적을 것: 무엇을 바꿨는지(피처·모델·후처리·파라미터), 정직 CV 점수, 리더보드 점수(제출했다면), 만든 py·csv 파일명, 결론(채택/폐기).
- 예시 — `approach1.md`
  > 접근 1: 클래스별 변이 개수 가중치. v1: 개수 가중치 합산 단독 분류기(CV 0.2765). v2: v1 피처 + 가중치 점수 140개를 XGB에 투입, 정직 CV 0.4486, LB 0.41 (`approach1_v2_20260909_1442`).

## 2. 파일 이름
```
approachN_vK_YYYYMMDD_HHMM
```
- `approachN` = 접근법 번호, `vK` = 그 접근법의 버전, 뒤는 **제작 일자와 시각**.
- py 파일(`src/`)과 제출 csv(`submissions/`)는 **같은 이름**을 쓴다 → 어떤 코드가 어떤 제출을 만들었는지 1:1 대응.
- 후처리 변형본은 접미사로 구분: `_twin_rule` (쌍둥이 규칙 적용본).
- 버전이 올라가면 새 파일을 만들고, 바뀐 내용은 `team/approaches/approachN.md`의 해당 버전 섹션에 상세히 적는다. 기존 파일은 지우지 않는다.

## 3. 코드 (py)
- `src/approachN_vK_YYYYMMDD_HHMM.py` — 그 제출을 **재현하는 진입 스크립트**. 실행하면 같은 이름의 csv가 `submissions/`에 생긴다.
- 공용 로직(피처 생성, CV, 후처리)은 `src/common/` 아래 모듈에 두고 진입 스크립트에서 import 한다. `src/common/`은 이름 규칙의 예외(공용 라이브러리).
  - `src/common/main.py` 파이프라인(피처 조립, 정직 CV) · `make_submission.py` 제출 생성 · `features/` `approach1_count_weight/` `approach2_knowledge/` `approach3_class_feature_compare/` `postprocess/` `models/` `analysis/`
- 실행: `python3 src/approachN_vK_YYYYMMDD_HHMM.py` (스크립트가 `src/common`을 path에 추가한다)

## 4. 제출 파일 (csv)
- `submissions/approachN_vK_YYYYMMDD_HHMM.csv` — py와 동일 이름. `submissions/submission.csv`는 가장 최근 파일 복사본.
- `submissions/README.md`에 제출 순서·설정·정직 CV·리더보드 점수를 기록한다.

## 5. 검증·기록
- 모든 실험은 `docs/experiments_log.md`에 한 줄 추가 (날짜, 실험명, 설정, 정직 CV, LB, 비고).
- 검증은 반드시 정직 CV(`--group-twins`, 쌍둥이 같은 fold). test.csv는 제출 스크립트의 추론 단계에서만 읽는다.
- 분석 문서(EDA, 도메인, 프로필, 중복 발견, test 관찰)는 `docs/`에 둔다.
