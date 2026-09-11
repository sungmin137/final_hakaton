"""접근 2 v10 (규제 축 2단계): v7(colsample 0.7, LB 0.4377)에서 colsample_bytree를 0.5로. 복원 배율 적용.

재현: python3 "4. src/submissions_source/approach2_v10_20260911_1014.py"  →  5. submissions/approach2_v10_20260911_1014.csv
설정: --features v4 --params mild_col5 --class-scale-file <복원 배율 json>
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ['--features', 'v4', '--params', 'mild_col5', '--class-scale-file', '6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json'] + ["--tag", "approach2_v10_20260911_1014"]
make_submission.main()
