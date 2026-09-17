"""접근 14 v20: v19(30차 0.4803, 팀 최고)에서 v4sn 비중을 올린 변형 — v4s .1 / v4sn .4 / v2 .2 / v4sp .15 / NB .15.
정직 CV 0.5381 (v19 0.5352 +0.0029, 절반 교차 +0.0009/+0.0055, train 바뀐 행 346). 다섯 파트 전부 v19 실행 때 저장된 test 확률 재사용(재학습 없음).

재현: python3 "4. src/submissions_source/approach14_v20_20260916_1140.py"  →  5. submissions/approach14_v20_20260916_1140.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
D = ROOT / "6. experiments/submissions/approach14_v19_20260916_1110"
part = lambda k: (lambda tr, te: np.load(D / f"test_proba_part{k}.npy"))
run_multi("approach14_v20_20260916_1140", [(part(0), 0.1), (part(1), 0.4), ("p2", 0.2), (part(3), 0.15), (part(4), 0.15)])
