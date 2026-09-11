"""접근 2 v2: + BLOSUM62·아미노산 특성 지식 피처, 클래스 배율 후처리 (재현 가능 버전. 정직 CV 0.4683+배율, LB 미제출)

재현: python3 "4. src/submissions_source/approach2_v2_20260909_1943.py"  →  5. submissions/approach2_v2_20260909_1943.csv (+ _twin_rule 변형본)
설정: --features v4 --class-scale 6. experiments/2026-09-09_v4_xgb_grp
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4"] + ['--class-scale', '6. experiments/2026-09-09_v4_xgb_grp'] + ["--tag", "approach2_v2_20260909_1943"]
make_submission.main()
