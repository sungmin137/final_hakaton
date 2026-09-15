"""접근 14 v15: v13에서 가중치만 v4s .7 / v2 .3. 정직 CV 0.5194 (v13 +0.0017, 31~100 0.482). v4s 확률은 v11 저장본.

재현: python3 "4. src/submissions_source/approach14_v15_20260915_1030.py"  →  5. submissions/approach14_v15_20260915_1030.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
v4s = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
run_multi("approach14_v15_20260915_1030", [(v4s, 0.7), ("p2", 0.3)])
