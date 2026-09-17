"""접근 14 v21: v20(31차 0.4810, 팀 최고)에서 v4sn .45 / NB .1 (v4s .1 / v2 .2 / v4sp .15). 정직 CV 0.5433 (v20 0.5381 +0.0052, 절반 교차 +0.0079/+0.0019, train 바뀐 행 164).
다섯 파트 전부 v19 실행 때 저장된 test 확률 재사용(재학습 없음).

재현: python3 "4. src/submissions_source/approach14_v21_20260917_0900.py"  →  5. submissions/approach14_v21_20260917_0900.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
D = ROOT / "6. experiments/submissions/approach14_v19_20260916_1110"
part = lambda k: (lambda tr, te: np.load(D / f"test_proba_part{k}.npy"))
run_multi("approach14_v21_20260917_0900", [(part(0), 0.1), (part(1), 0.45), ("p2", 0.2), (part(3), 0.15), (part(4), 0.1)])
