"""접근 4 v1 (단독): 기본 피처(유전자 이진화·변이 개수) + 문헌 driver·경로·역할 피처 131개 → XGB. 다른 접근법 피처 없음.

재현: python3 "4. src/submissions_source/approach4_v1_20260909_2135.py"  →  5. submissions/approach4_v1_20260909_2135.csv
설정: --features a4
설명 문서: 2. team/approaches/approach4.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ['--features', 'a4'] + ["--tag", "approach4_v1_20260909_2135"]
make_submission.main()
