"""접근 2 v2: + BLOSUM62·아미노산 특성 지식 피처, 클래스 배율 후처리 (정직 CV 0.4691+0.012, LB 0.43 최고)

⚠️ 이 csv는 재현성 수정(접근1 점수 float64+반올림, 2026-09-09 저녁) 이전에 만들어져 비트 단위 재현이 안 된다.
   같은 설정의 재현 가능 버전: approach2_v2_20260909_1943.py / .csv (원본과 522행 차이). 원본 csv는 리더보드 기록으로 보존.

재현: python3 "4. src/submissions_source/approach2_v2_20260909_1653.py"  →  5. submissions/approach2_v2_20260909_1653.csv (+ _twin_rule 변형본)
설정: --features v4 --class-scale 6. experiments/2026-09-09_v4_xgb_grp
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4"] + ['--class-scale', '6. experiments/2026-09-09_v4_xgb_grp'] + ["--tag", "approach2_v2_20260909_1653"]
make_submission.main()
