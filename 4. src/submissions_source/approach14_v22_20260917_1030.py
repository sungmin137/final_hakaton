"""접근 14 v22: v21(32차 0.4960, 팀 최고)에서 NB 파트너를 빼고 v4sn을 올림 — v4s .1 / v4sn .5 / v2 .25 / v4sp .15 / NB 0.
정직 CV 0.5453 (v21 0.5433 +0.0020, 절반 교차 +0.0040/+0.0004, train 바뀐 행 391). 32차에서 확인된 방향(NB 정보는 파트너보다 v4sn 피처로). 저장 확률 재사용.

재현: python3 "4. src/submissions_source/approach14_v22_20260917_1030.py"  →  5. submissions/approach14_v22_20260917_1030.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
D = ROOT / "6. experiments/submissions/approach14_v19_20260916_1110"
part = lambda k: (lambda tr, te: np.load(D / f"test_proba_part{k}.npy"))
run_multi("approach14_v22_20260917_1030", [(part(0), 0.1), (part(1), 0.5), ("p2", 0.25), (part(3), 0.15)])
