"""접근 14 v24: v21(32차 0.4960)에 v4snb(NB 점수표 피처 α2.0 버전의 v4sn)를 파트너로 추가. v4s .1 / v4sn .25 / v4snb .2 / v2 .2 / v4sp .15 / NB .1.
정직 CV 0.5446 (v21 0.5433 +0.0013, 절반 교차 +0.0007/+0.0023, train 바뀐 행 231). v4snb만 전체 train 학습, 나머지 저장본.

재현: python3 "4. src/submissions_source/approach14_v24_20260918_1000.py"  →  5. submissions/approach14_v24_20260918_1000.csv (+ twin_rule/ 변형본)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, model_proba, ROOT
D = ROOT / "6. experiments/submissions/approach14_v19_20260916_1110"
part = lambda k: (lambda tr, te: np.load(D / f"test_proba_part{k}.npy"))
run_multi("approach14_v24_20260918_1000", [(part(0), 0.1), (part(1), 0.25), (model_proba("v4snb"), 0.2), ("p2", 0.2), (part(3), 0.15), (part(4), 0.1)])
