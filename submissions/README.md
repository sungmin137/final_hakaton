# 제출 파일 목록

파일명 규칙 (team/file_rules.md): `approachN_vK_YYYYMMDD_HHMM.csv` — **같은 이름의 `src/submissions/….py`가 이 파일을 만든다.**
- 쌍둥이 규칙 적용본(`_twin_rule`)은 `experiments/submissions/twin_rule/`에 보관 · `submission.csv`: 가장 최근 파일 복사본
- 설정 상세는 `team/approaches/approachN.md`의 해당 버전 섹션

| 순서 | 파일 | 만든 모듈 / 설정 | 정직 CV Macro F1 | 리더보드 |
|---|---|---|---|---|
| 1차 | approach1_v2_20260909_1442.csv | 접근1 v2: 개수 가중치 점수 + XGB | 0.4486 | 0.41 |
| 2차 | approach2_v1_20260909_1553.csv | 접근2 v1: 인사이트 피처 + XGB | 0.4663 | 미제출 |
| 3차 | approach2_v2_20260909_1653.csv | 접근2 v2: 지식 피처 + 클래스 배율 | 0.4691 (+0.012) | **0.43 (최고)** |
| 4차 | approach3_v2_20260909_1715.csv | 접근3 v2: driver/burden 블렌딩 + 배율 | 약 0.49 | 미제출 |
| 5차 | approach3_v3_20260909_1807.csv | 접근3 v3: + CatBoost 앙상블 + 배율 | 약 0.49 | 0.417 |
| (후보) | approach2_v2_20260909_1943.csv | 3차와 같은 설정의 재현 가능 버전 (재현성 수정 후) | 0.4683 (+배율) | 미제출 |

쌍둥이 규칙(docs/04_duplicate_twins.md) 적용본은 팀 판단 전까지 제출하지 않는다.
