# 접근 5 — 통합 (접근 1 + 2 + 3 + 4 모두 적용)

## 개요
지금까지의 접근법을 한 모델에 전부 적용한 케이스. 각 접근법은 단독 파일(approach1~4)로 따로 평가하고, 이 문서의 케이스만 합친다.

| 구성 요소 | 출처 | 적용 방식 |
|---|---|---|
| 기본 피처 (유전자 이진화 4,230 + 변이 개수·유형) | 베이스라인 | `features/features.py` |
| 접근 1 클래스별 개수 가중치 점수 140 | approach1 | `--features v5`에 포함 |
| 접근 2 인사이트 피처(hotspot 위치·LoF·조합·특수그룹) + BLOSUM62 지식 피처 | approach2 | `--features v5`에 포함 |
| 접근 4 문헌 driver·경로·역할 피처 131 | approach4 | `--features v5`에 포함 |
| 접근 3 driver/burden 전문가 로그 블렌딩 (0.5, 0.2) | approach3 v2 | `--approach3-blend 0.5,0.2` (CatBoost 전문가는 LB 효과 미확인이라 제외) |
| 클래스 배율 후처리 (train OOF 기반) | postprocess | `--class-scale "6. experiments/approach5_all_v1"` |

## 버전
| 버전 | 파일 (src / submissions) | 정직 CV Macro F1 | LB |
|---|---|---|---|
| v1 | approach5_v1_<시각> | (기입 예정: v5 단독 0.4761, +블렌딩 OOF 기입) | 미제출 |

## 주의
- 접근 3 블렌딩은 CV에서 +0.013이었지만 LB에서는 역효과(5차 0.417 < 3차 0.43)였다. 통합 케이스의 LB가 접근 2 v2(0.43)보다 낮으면 블렌딩을 빼는 v2를 만든다.
