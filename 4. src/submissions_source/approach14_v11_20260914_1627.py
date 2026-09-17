"""접근 14 v11: v4s 모델(v4 피처 + 아미노산 치환 스펙트럼, 전체 train 학습) 0.5 + 접근14 v2 모델(저장 확률) 0.5 로그 결합 → 3차 복원 배율 → 쌍둥이 규칙.
16차 대비 바뀐 요소: 9차 모델을 "9차 피처 + 스펙트럼" 모델로 교체(한 요소). 전부 train 근거(스펙트럼은 행 내부 정규화, fit 통계 없음).
정직 CV: v4s 단독 0.5035(9차 0.4847), v4s+v2 0.5137 (16차 0.4941, +0.0196), 절반 교차 +0.025/+0.015, 31~100 0.436→0.464.

재현: python3 "4. src/submissions_source/approach14_v11_20260914_1627.py"  →  5. submissions/approach14_v11_20260914_1627.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, model_proba
run_multi("approach14_v11_20260914_1627", [(model_proba("v4s"), 0.5), ("p2", 0.5)])
