"""접근 14 v12: 4모델 로그 결합 — 9차(저장) .25 / v2(저장) .25 / v4s(전체 학습) .25 / v4sp(v4p + 스펙트럼, 전체 학습) .25 → 배율 → 규칙.
정직 CV 0.5172 (16차 +0.0231, 절반 교차 +0.030/+0.017). v11(두 모델)보다 두 요소 더 바뀜 → v11 뒤에 제출.

재현: python3 "4. src/submissions_source/approach14_v12_20260914_1627.py"  →  5. submissions/approach14_v12_20260914_1627.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, model_proba
run_multi("approach14_v12_20260914_1627", [("p7", 0.25), ("p2", 0.25), (model_proba("v4s"), 0.25), (model_proba("v4sp"), 0.25)])
