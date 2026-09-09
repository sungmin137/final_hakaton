# 암 아형(SUBCLASS) 예측 — 유전자 변이 기반 다중 분류

환자별 유전자 변이 프로파일(4,384개 컬럼)로 26개 암 아형을 예측하는 해커톤 프로젝트.

## 구조
```
final_hakaton/
├── data/raw/        # train.csv, test.csv, sample_submission.csv (git 제외)
├── docs/            # 배경, 계획, 실험 기록, 발표자료
├── notebooks/       # EDA 및 실험 노트북
├── src/
│   ├── main.py                    # 베이스라인 파이프라인 (공식 XGB 구조 + 정직 CV) — 최상위 유지
│   ├── make_submission.py         # 제출 파일 생성 (test.csv를 읽는 유일한 진입점)
│   ├── baseline/                  # 초기 LightGBM 베이스라인
│   ├── analysis/                  # EDA, 변이 카탈로그, 암종별 프로필
│   ├── features/                  # v1 이진화·카운트, v3 인사이트 피처
│   ├── approach1_count_weight/    # 접근 1: 클래스별 변이 개수 가중치 (v2 점수 피처)
│   ├── approach2_knowledge/       # 접근 2: 지식 기반 변이 특성 (BLOSUM62, v4)
│   ├── approach3_class_feature_compare/  # 접근 3: 각 클래스 특징별 비교 v1 (전문가 + 스태킹)
│   ├── postprocess/               # 쌍둥이 규칙, 클래스 배율 후처리
│   └── models/                    # GPU MLP
├── experiments/     # 실험 설정/결과 로그
├── models/          # 학습된 모델 (git 제외)
└── submissions/     # 제출 파일
```

## 메인 파이프라인 (공식 베이스라인 구조)
```bash
PYTHONPATH=src python3 src/main.py --features v2 --cv        # Stratified 5-Fold 평가
PYTHONPATH=src python3 src/main.py --features v1 --cv --submit  # + 전체 학습 → test 추론 → submissions/
```
제출 파일 생성 (test.csv는 여기서만 읽힘):
```bash
PYTHONPATH=src python3 src/make_submission.py --features v3   # → submissions/submission.csv (+ _twin.csv)
```
피처 버전: `v1` 이진화+카운트 · `v2` = v1 + 접근1 개수가중치 점수 · `v3` = v2 + hotspot 위치·LoF 유전자·조합·특수그룹

노트북 버전: `notebooks/main.ipynb` (커널: Python 3.14 (final_hakaton))

| 실험 | OOF Macro F1 | OOF Acc |
|---|---|---|
| 공식 베이스라인 인코딩 + XGB | 0.3048 | 0.3288 |
| v1 피처 + XGB | 0.3845 | 0.3917 |
| v2 = v1 + 접근1 개수 가중치 점수 (sungmin) | 0.4942 | 0.5265 |

**정직 CV (쌍둥이를 같은 fold에, `--group-twins`) — 이 기준으로 비교**

| 실험 | 정직 CV Macro F1 | 리더보드 |
|---|---|---|
| v1 | 0.3979 | - |
| v2 | 0.4486 | 0.41 |
| v2 + 클래스가중치 | 0.4478 | - |
| v2 + 튜닝 파라미터 | 0.4535 | - |
| **v3 (현재 submission.csv)** | **0.4663** | (제출 대기) |
| v3 + 튜닝 파라미터 | 0.4576 | - |

## 시작
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```
