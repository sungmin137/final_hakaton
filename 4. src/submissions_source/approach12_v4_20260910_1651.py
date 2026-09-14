"""접근 12 (구 접근 2 v7) (mild): 3차 구성(v4 + 복원 배율 + 쌍둥이 규칙)에서 XGB colsample_bytree만 0.7로.

재현: python3 "4. src/submissions_source/approach12_v4_20260910_1651.py"  →  5. submissions/approach12_v4_20260910_1651.csv
설정: --features v4 --params mild_col --class-scale-file <복원 배율 json>
설명 문서: 2. team/approaches/approach12.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ['--features', 'v4', '--params', 'mild_col', '--class-scale-file', '6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json'] + ["--tag", "approach12_v4_20260910_1651"]
make_submission.main()
