"""접근 1 v2: v1 이진화·카운트 + 클래스별 개수 가중치 점수 140개 → XGB (정직 CV 0.4486, LB 0.41)

재현: python3 src/submissions/approach1_v2_20260909_1442.py  →  submissions/approach1_v2_20260909_1442.csv (+ _twin_rule 변형본)
설정: --features v2 
설명 문서: team/approaches/approach1.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v2"] + [] + ["--tag", "approach1_v2_20260909_1442"]
make_submission.main()
