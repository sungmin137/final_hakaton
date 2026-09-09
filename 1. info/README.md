# 해커톤 정보 (final_hakaton info)

| 항목 | 내용 |
|---|---|
| 주제 | 유전자 변이 정보 기반 암 아형(SUBCLASS) 예측 — 26클래스 다중 분류 |
| 평가 | **Macro F1**, Public = test 100% (리더보드 = 최종) |
| 제출 | 1일 최대 4회, `ID,SUBCLASS` csv |
| 외부 데이터 | **불가** (논문·교과서 지식은 가능으로 해석 → `2. team/rules_compliance.md`) |
| 사전학습 모델 | 가능 |
| Data leakage | test를 학습·인코딩·스케일링·결측 통계에 쓰면 수상 제외 |
| 참여 | 팀 (성민 외), GitHub `sungmin137/final_hakaton` |
| 마감 | (미확인) |

## 문서
- `background.md` — 대회 배경·설명·데이터 설명 원문 정리
- `data/README.md` — 제공 데이터 3종(train / test / sample_submission) 구조와 특징
- `reference_dacon_2024.md` — 같은 데이터로 열린 DACON 2024 대회 리더보드·상위팀 방법 조사
- `baseline.py` — 주최측 공식 베이스라인(XGB)을 그대로 옮긴 스크립트

## 규칙 원문 요약
1. 평가산식 Macro F1 Score, Public에서 Test 100% 활용
2. 개인/팀 참여 가능
3. 외부 데이터 사용 불가, 사전 학습 모델 사용 가능
4. 1일 최대 제출 4회. 모델 학습에 평가 데이터셋 활용(Data Leakage) 시 수상 제외 — label/one-hot 인코딩·스케일링·get_dummies·결측 처리에 test 활용 포함
