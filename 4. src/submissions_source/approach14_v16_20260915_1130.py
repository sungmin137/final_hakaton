"""접근 14 v16: v14(v4s .5 / v2 .3 / v4sp .2)에 스펙트럼·카운트만 쓴 421열 저차원 모델(spec, XGB mild_col, 전체 train 학습)을 파트너로 추가.
가중 v4s .45 / v2 .25 / v4sp .15 / spec .15. 정직 CV 0.5234 (v14 0.5200 +0.0034, 절반 교차 +0.003/+0.004). spec 단독은 0.2959로 약하지만 정보가 달라 블렌드에 보탬.
v4s·v4sp·v2 확률은 저장본 재사용, spec만 새로 학습(약 1분).

재현: python3 "4. src/submissions_source/approach14_v16_20260915_1130.py"  →  5. submissions/approach14_v16_20260915_1130.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, model_proba, ROOT
v4s = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
v4sp = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v12_20260914_1627/test_proba_part3.npy")
run_multi("approach14_v16_20260915_1130", [(v4s, 0.45), ("p2", 0.25), (v4sp, 0.15), (model_proba("spec"), 0.15)])
