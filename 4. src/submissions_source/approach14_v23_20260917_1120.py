"""접근 14 v23: v21(32차 0.4960, 팀 최고)에 v4sn10(v4sn과 같되 NB 점수표 피처의 내부 OOF를 10-fold로 만든 모델)을 파트너로 추가.
v4s .1 / v4sn .25 / v4sn10 .2 / v2 .2 / v4sp .15 / NB .1. 정직 CV 0.5474 (v21 0.5433 +0.0041, 절반 교차 +0.0062/+0.0022, train 바뀐 행 224).
v4sn10만 전체 train 학습, 나머지 파트는 v19 실행 저장본 재사용. 전부 train 근거.

재현: python3 "4. src/submissions_source/approach14_v23_20260917_1120.py"  →  5. submissions/approach14_v23_20260917_1120.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, model_proba, ROOT
D = ROOT / "6. experiments/submissions/approach14_v19_20260916_1110"
part = lambda k: (lambda tr, te: np.load(D / f"test_proba_part{k}.npy"))
run_multi("approach14_v23_20260917_1120", [(part(0), 0.1), (part(1), 0.25), (model_proba("v4sn10"), 0.2), ("p2", 0.2), (part(3), 0.15), (part(4), 0.1)])
