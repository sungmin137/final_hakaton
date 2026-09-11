"""접근 2 v12: v11 구조(min_child_weight 3, reg_lambda 3, 복원 배율)에 colsample_bytree만 0.5로 (v11 0.7 → 0.5).

재현: python3 "4. src/submissions_source/approach2_v12_20260911_1129.py"  →  5. submissions/approach2_v12_20260911_1129.csv
설정: --features v4 --params mild_col5_reg --class-scale-file <복원 배율 json>
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4", "--params", "mild_col5_reg", "--class-scale-file", '6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json', "--tag", "approach2_v12_20260911_1129"]
make_submission.main()
