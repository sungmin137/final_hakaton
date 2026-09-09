"""접근 2 v1: 인사이트 피처(hotspot 위치·LoF 유전자·동반 조합·특수 그룹) 추가 → XGB (정직 CV 0.4663, 미제출)

재현: python3 src/approach2_v1_20260909_1553.py  →  submissions/approach2_v1_20260909_1553.csv (+ _twin_rule 변형본)
설정: --features v3 
설명 문서: team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v3"] + [] + ["--tag", "approach2_v1_20260909_1553"]
make_submission.main()
