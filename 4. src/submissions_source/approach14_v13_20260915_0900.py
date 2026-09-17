"""접근 14 v13: v11과 같은 두 모델(v4s 스펙트럼 모델, v2)에서 가중치만 0.6/0.4 (정직 CV 0.5177, v11 0.5137). 한 번에 한 요소(가중치).
v4s의 test 확률은 v11 실행 때 저장된 것을 재사용(재학습 편차 없음): 6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy

재현: python3 "4. src/submissions_source/approach14_v13_20260915_0900.py"  →  5. submissions/approach14_v13_20260915_0900.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
v4s = lambda train, test: np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
run_multi("approach14_v13_20260915_0900", [(v4s, 0.6), ("p2", 0.4)])
