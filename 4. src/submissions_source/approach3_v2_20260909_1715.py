"""접근 3 v2: driver/burden 전문가 로그 블렌딩(0.5,0.2) + 클래스 배율 (정직 CV 약 0.49, 미제출)

재현: python3 "4. src/submissions_source/approach3_v2_20260909_1715.py"  →  5. submissions/approach3_v2_20260909_1715.csv (+ _twin_rule 변형본)
설정: --features v4 --approach3-blend 0.5,0.2 --class-scale 6. experiments/approach3_class_feature_compare_v2
설명 문서: 2. team/approaches/approach3.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4"] + ['--approach3-blend', '0.5,0.2', '--class-scale', '6. experiments/approach3_class_feature_compare_v2'] + ["--tag", "approach3_v2_20260909_1715"]
make_submission.main()
