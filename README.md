# 암 아형(SUBCLASS) 예측 — 유전자 변이 기반 다중 분류

환자별 유전자 변이 프로파일(4,384개 컬럼)로 26개 암 아형을 예측하는 해커톤 프로젝트. 평가 Macro F1, 외부 데이터 금지.

## 구조
```
final_hakaton/
├── info/                 # 해커톤 정보: 배경·규칙(README), 제공 데이터(data/, csv는 git 제외), 공식 baseline.py, DACON 참고
├── team/                 # 팀 규칙·회의·결정: file_rules.md(파일 규칙), decisions.md, meetings/, approaches/approachN.md
├── src/
│   ├── approachN_vK_YYYYMMDD_HHMM.py   # 제출 재현 진입 스크립트 (csv와 같은 이름)
│   └── common/                          # 공용 라이브러리: main.py(파이프라인·정직 CV), make_submission.py, features/, approach*/, postprocess/, models/, analysis/
├── submissions/          # approachN_vK_YYYYMMDD_HHMM.csv (+ _twin_rule), README.md(제출 색인), submission.csv(최신)
├── docs/                 # 분석 문서: EDA, 도메인 지식, 암종별 프로필, 중복 발견, test 관찰, 실험 로그, 시각화
├── experiments/          # 실험 산출물(OOF 확률, 결과 json, 로그)
└── notebooks/            # 공식 베이스라인 원본, main.ipynb
```

## 시작
```bash
pip3 install -r requirements.txt
python3 info/baseline.py                                   # 공식 베이스라인 그대로
PYTHONPATH=src/common python3 src/common/main.py --features v4 --cv --group-twins   # 정직 CV
python3 src/approach2_v2_20260909_1653.py                  # 제출 재현 → submissions/approach2_v2_20260909_1653.csv
```

## 현재 성적 (정직 CV = 쌍둥이를 같은 fold에 묶은 5-Fold)
| 제출 | 파일 | 정직 CV Macro F1 | 리더보드 |
|---|---|---|---|
| 접근1 v2 | approach1_v2_20260909_1442 | 0.4486 | 0.41 |
| **접근2 v2** | approach2_v2_20260909_1653 | 0.4691 (+배율 0.012) | **0.43** |
| 접근3 v3 | approach3_v3_20260909_1807 | 약 0.49 | 0.417 |

CV 개선이 리더보드로 거의 옮겨지지 않는다 — test는 변이 31~100개 구간이 train의 2.4배(docs/08). 상세는 `team/approaches/`.
