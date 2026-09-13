# 현재 상태 한 장 (2026-09-13 밤) — 새 세션·팀원 공용

- **팀 최고 LB 0.44235** = 13차 `approach19_hnsc_stes_soft_20260912_twin_rule.csv` (혜림). 구성: v7(v4 피처, XGB colsample 0.7) + 3차 복원 배율 26개 + HNSC/STES 소프트 라우터 + 쌍둥이 규칙.
- 상승분 분해: 쌍둥이 규칙 8행 ≈ +0.0048, 라우터 3행 ≈ 0. 배율 재계산(approach22)은 −0.0035로 폐기.
- **다음 제출 후보**: `6. experiments/submissions/twin_rule/approach14_v3_20260913_2132_twin_rule.csv` — 9차 모델 + 접근14 v2 모델(유전자별 변이 유형 분리 점수) 로그평균 + 복원 배율 + 규칙. 정직 CV 0.4941(+0.0094), 변이 31~100구간 0.403→0.457. 9차와 377행 차이.
- 고정 원칙: 배율은 복원 벡터 고정, 규칙 ON, 제출은 "13차 + 한 요소", 판단은 LB(편차 ±0.015), CV는 버그 감지용.
- 역할: 혜림=재판기 쌍 확장(입력에 새 정보 추가), 혜성=신장·뇌 저차원 서브모델(v11 통합 확인), 성민=접근 14.
- 문서 지도: 방향 `2. team/direction_2026-09-13.md` · 쌍둥이 논리 `2. team/twin_rule_rationale.md` · 분석 `3. docs/07_top_submissions_analysis.md` · 제출 이력 `5. submissions/README.md` · 접근별 `2. team/approaches/approachN.md`.
- 정리 필요: hyerim·hyeseong 브랜치 main 병합(접근 10 번호 충돌, hyerim의 옛 폴더명 `4. src/submissions`).
