# 제출 파일 목록

파일명 규칙: `<src 모듈>_<버전>[_class_scale][_twin_rule]_<YYYYMMDD-HHMM>.csv`
- `<src 모듈>`: 결과를 만든 접근법 폴더 이름 (`src/` 하위 폴더와 동일)
- `_class_scale`: `src/postprocess/class_scale.py` 후처리 적용 · `_twin_rule`: `src/postprocess/twin_rule.py` 적용본
- `submission.csv`: 가장 최근에 만든 파일의 복사본 (제출용)

| 순서 | 파일 | 만든 모듈 / 설정 | 정직 CV Macro F1 | 리더보드 |
|---|---|---|---|---|
| 1차 | approach1_count_weight_v2_20260909-1442.csv | v2 피처(접근1 개수 가중치 점수) + XGB | 0.4486 | **0.41** |
| 2차 | features_v3_20260909-1553.csv | v3 인사이트 피처 + XGB | 0.4663 | - |
| 3차 | approach2_knowledge_v4_class_scale_20260909-1653.csv | v4 지식 피처 + 클래스 배율 | 0.4691 (+0.012) | **0.43 (최고)** |
| 4차 | approach3_class_feature_compare_v2_class_scale_20260909-1715.csv | v4 + driver/burden 블렌딩(0.5,0.2) + 배율 | 약 0.49 | - |
| 5차 | approach3_class_feature_compare_v3_class_scale_20260909-1807.csv | v4 + driver/burden/CatBoost 앙상블(0.7,0.4,0.8) + 배율 | 약 0.49~0.50 | **0.417** |

각 파일의 `_twin_rule_` 버전은 쌍둥이 규칙(docs/07)을 추가로 적용한 것. 팀 판단 전까지는 규칙 미적용본을 기본으로.
