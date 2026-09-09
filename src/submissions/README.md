# src/submissions — 제출 재현 스크립트

`submissions/<이름>.csv` 하나마다 같은 이름의 `src/submissions/<이름>.py`가 있다. 스크립트를 실행하면 그 csv가 다시 만들어진다.

| 스크립트 | 접근법 문서 | 만든 제출 |
|---|---|---|
| approach1_v2_20260909_1442.py | team/approaches/approach1.md | submissions/approach1_v2_20260909_1442.csv (LB 0.41) |
| approach2_v1_20260909_1553.py | team/approaches/approach2.md | submissions/approach2_v1_20260909_1553.csv |
| approach2_v2_20260909_1653.py | team/approaches/approach2.md | submissions/approach2_v2_20260909_1653.csv (LB 0.43, 재현성 수정 전 파일 — 보존용) |
| approach2_v2_20260909_1943.py | team/approaches/approach2.md | submissions/approach2_v2_20260909_1943.csv (같은 설정의 재현 가능 버전, 미제출) |
| approach3_v2_20260909_1715.py | team/approaches/approach3.md | submissions/approach3_v2_20260909_1715.csv |
| approach3_v3_20260909_1807.py | team/approaches/approach3.md | submissions/approach3_v3_20260909_1807.csv (LB 0.417) |

실행: `python3 src/submissions/approach2_v2_20260909_1653.py` (루트에서). 작성 규칙은 `team/file_rules.md` 3절.
