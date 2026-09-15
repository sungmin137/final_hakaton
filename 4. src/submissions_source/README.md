# 4. src/submissions — 제출 재현 스크립트

`5. submissions/<이름>.csv` 하나마다 같은 이름의 `4. src/submissions_source/<이름>.py`가 있다. 스크립트를 실행하면 그 csv가 다시 만들어진다.

| 스크립트 | 접근법 문서 | 만든 제출 |
|---|---|---|
| approach1_v2_20260909_1442.py | 2. team/approaches/approach1.md | 5. submissions/approach1_v2_20260909_1442.csv (LB 0.41) |
| approach2_v1_20260909_1553.py | 2. team/approaches/approach2.md | 5. submissions/approach2_v1_20260909_1553.csv |
| approach2_v2_20260909_1653.py | 2. team/approaches/approach2.md | 5. submissions/approach2_v2_20260909_1653.csv (LB 0.43, 재현성 수정 전 파일 — 보존용) |
| approach2_v2_20260909_1943.py | 2. team/approaches/approach2.md | 5. submissions/approach2_v2_20260909_1943.csv (같은 설정의 재현 가능 버전, 미제출) |
| approach3_v2_20260909_1715.py | 2. team/approaches/approach3.md | 5. submissions/approach3_v2_20260909_1715.csv |
| approach3_v3_20260909_1807.py | 2. team/approaches/approach3.md | 5. submissions/approach3_v3_20260909_1807.csv (LB 0.417) |

실행: `python3 "4. src/submissions_source/approach2_v2_20260909_1653.py"` (루트에서). 작성 규칙은 `2. team/file_rules.md` 3절.
| approach4_v1_20260909_2135.py | 2. team/approaches/approach4.md | 5. submissions/approach4_v1_20260909_2135.csv |
| approach2_v3_20260909_2135.py | 2. team/approaches/approach2.md | 5. submissions/approach2_v3_20260909_2135.csv |
| approach5_v1_20260909_2135.py | 2. team/approaches/approach5.md | 5. submissions/approach5_v1_20260909_2135.csv |
| approach5_v2_20260909_2322.py | 2. team/approaches/approach5.md | 5. submissions/approach5_v2_20260909_2322.csv |
| approach8_v1_20260910_1029.py | 2. team/approaches/approach8.md | 5. submissions/approach8_v1_20260910_1029.csv (LB 0.3972671492) |
| approach9_v1_20260909_1827.py | 2. team/approaches/approach9.md | 5. submissions/approach9_v1_20260909_1827.csv (LB 0.4031845152, 코드는 참고용 — 재실행 결과 재현 안 됨) |
| approach12_v1_20260910_1451.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v1_20260910_1451.csv (fold 배깅) |
| approach12_v2_20260910_1453.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v2_20260910_1453.csv (16모델 배깅 + 배율 안정화) |
| approach12_v3_20260910_1505.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v3_20260910_1505.csv (후처리 전용: v5 원시 확률 × 3차 복원 배율) |
| approach12_v4_20260910_1651.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v4_20260910_1651.csv (mild colsample 0.7) |
| approach12_v5_20260910_1651.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v5_20260910_1651.csv (mild cw_variant 제거) |
| approach12_v6_20260910_1651.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v6_20260910_1651.csv (mild mcw3·lambda3) |
| approach12_v7_20260911_1014.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v7_20260911_1014.csv (colsample 0.5) |
| approach12_v8_20260911_1014.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v8_20260911_1014.csv (colsample 0.7 + mcw3·lambda3) |
| approach12_v9_20260911_1129.py | 2. team/approaches/approach2.md | 5. submissions/approach12_v9_20260911_1129.csv (colsample 0.5 + mcw3·lambda3) |
| approach13_v1_20260911_1602.py | 2. team/approaches/approach13.md | 5. submissions/approach13_v1_20260911_1602.csv (개선판 라우팅) |
| approach14_v3_20260913_2132.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v3_20260913_2132.csv (두 모델 로그평균) |
| approach14_v4_20260913_2150.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v4_20260913_2150.csv (v3 + HNSC/STES 라우터) |
| approach14_v5_20260913_2150.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v5_20260913_2150.csv (w=0.3) |
| approach14_v6_20260913_2150.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v6_20260913_2150.csv (w=0.7) |
| approach14_v9_20260914_0010.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v9_20260914_0010.csv (16차 + 이진 유전자 LR w3=0.2) |
| approach14_v10_20260914_0110.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v10_20260914_0110.csv (9차+v2+a7 세 모델) |
| approach14_v11_20260914_1627.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v11_20260914_1627.csv (v4s 스펙트럼 + v2) |
| approach14_v12_20260914_1627.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v12_20260914_1627.csv (4모델) |
| approach14_v13_20260915_0900.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v13_20260915_0900.csv (v4s .6 / v2 .4) |
| approach14_v14_20260915_1030.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v14_20260915_1030.csv (v4s .5/v2 .3/v4sp .2) |
| approach14_v15_20260915_1030.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v15_20260915_1030.csv (v4s .7/v2 .3) |
| approach14_v16_20260915_1130.py | 2. team/approaches/approach14.md | 5. submissions/approach14_v16_20260915_1130.csv (v14 + spec .15) |
| approach14_v17_20260915_1345.py | approach14.md | 5. submissions/approach14_v17_20260915_1345.csv (혜림 NB + 스펙트럼 모델군) |
| approach14_v18_20260915_1400.py | approach14.md | 5. submissions/approach14_v18_20260915_1400.csv (5모델, v4sn 포함) |
