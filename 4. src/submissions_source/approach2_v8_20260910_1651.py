"""접근 2 v8 (mild): 3차 구성에서 접근 1 변이 단위 점수(cw_variant_*)만 제거.

재현: python3 "4. src/submissions_source/approach2_v8_20260910_1651.py"  →  5. submissions/approach2_v8_20260910_1651.csv
설정: --features v4 --drop-cols-prefix cw_variant --class-scale-file <복원 배율 json>
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py"] + ['--features', 'v4', '--drop-cols-prefix', 'cw_variant', '--class-scale-file', '6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json'] + ["--tag", "approach2_v8_20260910_1651"]
make_submission.main()
