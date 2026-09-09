# 암 아형(SUBCLASS) 예측 — 유전자 변이 기반 다중 분류

환자별 유전자 변이 프로파일(4,384개 컬럼)로 26개 암 아형을 예측하는 해커톤 프로젝트.

## 구조
```
final_hakaton/
├── data/raw/        # train.csv, test.csv, sample_submission.csv (git 제외)
├── docs/            # 배경, 계획, 실험 기록, 발표자료
├── notebooks/       # EDA 및 실험 노트북
├── src/             # 재사용 코드 (전처리, 피처, 학습, 추론)
├── experiments/     # 실험 설정/결과 로그
├── models/          # 학습된 모델 (git 제외)
└── submissions/     # 제출 파일
```

## 메인 파이프라인 (공식 베이스라인 구조)
```bash
PYTHONPATH=src python3 src/main.py --features v2 --cv        # Stratified 5-Fold 평가
PYTHONPATH=src python3 src/main.py --features v1 --cv --submit  # + 전체 학습 → test 추론 → submissions/
```
노트북 버전: `notebooks/main.ipynb` (커널: Python 3.14 (final_hakaton))

| 실험 | OOF Macro F1 | OOF Acc |
|---|---|---|
| 공식 베이스라인 인코딩 + XGB | 0.3048 | 0.3288 |
| v1 피처 + XGB | 0.3845 | 0.3917 |
| v2 = v1 + 접근1 개수 가중치 점수 (sungmin) | 0.4942 | 0.5265 |

## 시작
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```
