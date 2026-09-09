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

## 시작
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```
