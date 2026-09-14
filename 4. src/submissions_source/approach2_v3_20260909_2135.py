"""접근 2 v3 (단독): 기본 피처 + 인사이트 피처 + BLOSUM62 지식 피처 → XGB. 접근 1 점수 제외.

재현: python3 "4. src/submissions_source/approach2_v3_20260909_2135.py"  →  5. submissions/approach2_v3_20260909_2135.csv
설정: --features a2
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ['--features', 'a2'] + ["--tag", "approach2_v3_20260909_2135"]
make_submission.main()
