"""접근 2 v2: + BLOSUM62·아미노산 특성 지식 피처, 클래스 배율 후처리 (정직 CV 0.4691+0.012, LB 0.43 최고)

재현: python3 src/submissions/approach2_v2_20260909_1653.py  →  submissions/approach2_v2_20260909_1653.csv (+ _twin_rule 변형본)
설정: --features v4 --class-scale experiments/2026-09-09_v4_xgb_grp
설명 문서: team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4"] + ['--class-scale', 'experiments/2026-09-09_v4_xgb_grp'] + ["--tag", "approach2_v2_20260909_1653"]
make_submission.main()
