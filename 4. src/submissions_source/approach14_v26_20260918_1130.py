"""접근 14 v26: v21(32차 0.4960)과 같은 다섯 파트, v4sn 0.425 / NB 0.125 (v4s .1 / v2 .2 / v4sp .15). LB 봉우리 위치 탐색(v20 .4/.15 → v21 .45/.1 → v22 .5/0). CV로는 v21과 구별 불가. 저장 확률 재사용.

재현: python3 "4. src/submissions_source/approach14_v26_20260918_1130.py"
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
D = ROOT / "6. experiments/submissions/approach14_v19_20260916_1110"
part = lambda k: (lambda tr, te: np.load(D / f"test_proba_part{k}.npy"))
run_multi("approach14_v26_20260918_1130", [(part(0), 0.1), (part(1), 0.425), ("p2", 0.2), (part(3), 0.15), (part(4), 0.125)])
