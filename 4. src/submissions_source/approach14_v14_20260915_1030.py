"""접근 14 v14: v4s .5 / v2 .3 / v4sp .2 로그 결합 → 배율 → 규칙. 정직 CV 0.5200 (v13 0.5177 +0.0023, 절반 교차 +0.002/+0.002, 31~100 0.475→0.480).
v4s·v4sp test 확률은 v11·v12 실행 때 저장본 재사용(재학습 편차 없음).

재현: python3 "4. src/submissions_source/approach14_v14_20260915_1030.py"  →  5. submissions/approach14_v14_20260915_1030.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
v4s = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
v4sp = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v12_20260914_1627/test_proba_part3.npy")
run_multi("approach14_v14_20260915_1030", [(v4s, 0.5), ("p2", 0.3), (v4sp, 0.2)])
