# 암 아형(SUBCLASS) 예측 — 유전자 변이 기반 다중 분류

환자별 유전자 변이 프로파일(4,384개 컬럼)로 26개 암 아형을 예측하는 해커톤 프로젝트. 평가 Macro F1, 외부 데이터 금지.

## 구조
```
final_hakaton/
├── 1. info/                 # 해커톤 정보: 배경·규칙(README), 제공 데이터(data/, csv는 git 제외), 공식 baseline.py, DACON 참고
├── 2. team/                 # 팀 규칙·회의·결정: file_rules.md(파일 규칙), decisions.md, meetings/, approaches/approachN.md
├── 4. src/
│   ├── submissions_source/approachN_vK_YYYYMMDD_HHMM.py   # 제출 재현 스크립트 (5. submissions/ csv와 같은 이름)
│   └── common/                          # 공용 라이브러리: main.py(파이프라인·정직 CV), make_submission.py, features/, approach*/, postprocess/, models/, analysis/
├── 5. submissions/          # approachN_vK_YYYYMMDD_HHMM.csv (+ _twin_rule), README.md(제출 색인), submission.csv(최신)
├── 3. docs/                 # 분석 문서: EDA, 도메인 지식, 암종별 프로필, 중복 발견, test 관찰, 실험 로그, 시각화
├── 6. experiments/          # 실험 산출물(OOF 확률, 결과 json, 로그)
```

## 시작
```bash
pip3 install -r requirements.txt
python3 "1. info/baseline.py"                                   # 공식 베이스라인 그대로
PYTHONPATH="4. src/common" python3 "4. src/common/main.py" --features v4 --cv --group-twins   # 정직 CV
python3 "4. src/submissions_source/approach2_v2_20260909_1653.py"                  # 제출 재현 → 5. submissions/approach2_v2_20260909_1653.csv
```

## 성적·제출 기록
**전체 진행 정리(접근법·방법·이력·교훈): `2. team/PROGRESS_2026-09-14.md`** · 현재 상태 한 장 `2. team/STATUS.md` · 제출 색인과 리더보드 점수 `5. submissions/README.md` · 실험 기록 `3. docs/experiments_log.md` · 접근법별 상세 `2. team/approaches/`.

팀 최고 LB **0.4689** (21차, 2026-09-14): 9차 모델 + 변이 유형 분리 점수 모델 로그평균 + 3차 복원 배율 + 쌍둥이 규칙 + 초과변이 규칙.
