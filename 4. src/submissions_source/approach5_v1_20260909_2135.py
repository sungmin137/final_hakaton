"""접근 5 v1 (통합): 접근 1 점수 + 접근 2 인사이트·지식 + 접근 4 문헌 피처(v5) → XGB, 접근 3 driver/burden 전문가 블렌딩(0.5,0.2), 클래스 배율 후처리.

재현: python3 "4. src/submissions_source/approach5_v1_20260909_2135.py"  →  5. submissions/approach5_v1_20260909_2135.csv
설정: --features v5 --approach3-blend 0.5,0.2 --class-scale "6. experiments/approach5_all_v1"
설명 문서: 2. team/approaches/approach5.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ['--features', 'v5', '--approach3-blend', '0.5,0.2', '--class-scale', '6. experiments/approach5_all_v1'] + ["--tag", "approach5_v1_20260909_2135"]
make_submission.main()
