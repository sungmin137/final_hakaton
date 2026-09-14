"""접근 3 v3: + CatBoost 전문가 4-모델 앙상블 + 클래스 배율 (정직 CV 약 0.49, LB 0.417)

재현: python3 "4. src/submissions_source/approach3_v3_20260909_1807.py"  →  5. submissions/approach3_v3_20260909_1807.csv (+ _twin_rule 변형본)
설정: --features v4 --approach3-blend 0.7,0.4,0.8 --class-scale 6. experiments/approach3_class_feature_compare_v3
설명 문서: 2. team/approaches/approach3.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4"] + ['--approach3-blend', '0.7,0.4,0.8', '--class-scale', '6. experiments/approach3_class_feature_compare_v3'] + ["--tag", "approach3_v3_20260909_1807"]
make_submission.main()
