"""접근 8 v1: v5 피처를 중요도 상위 500개로 압축, 후처리 없음. 정직 CV 0.4809, LB 0.3972671492(2026-09-10 11:07:17 제출).

재현: python3 "4. src/submissions_source/approach8_v1_20260910_1029.py"  →  5. submissions/approach8_v1_20260910_1029.csv
설정: --features a8
설명 문서: 2. team/approaches/approach8.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ["--features", "a8"] + ["--tag", "approach8_v1_20260910_1029"]
make_submission.main()
