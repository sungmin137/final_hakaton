"""접근 2 v4: 3차(LB 0.4369)와 같은 피처(v4)·클래스 배율·쌍둥이 규칙에, 정직 CV fold 모델 5개 + 전체 모델의 확률 평균(fold 배깅)을 적용.
피처를 더하지 않고 분산만 줄인다. 하루 1회 제출 상황에서 가장 안전한 개선 후보.

재현: python3 "4. src/submissions_source/approach2_v4_20260910_1451.py"  →  5. submissions/approach2_v4_20260910_1451.csv (+ _twin_rule)
설정: --features v4 --fold-bag --class-scale "6. experiments/2026-09-09_v4_xgb_grp"
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4", "--fold-bag", "--class-scale", "6. experiments/2026-09-09_v4_xgb_grp", "--tag", "approach2_v4_20260910_1451"]
make_submission.main()
