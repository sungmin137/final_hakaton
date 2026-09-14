"""접근 13 v1: 9차 설정(v4 피처, colsample 0.7, 3차 복원 배율) 위에 개선판 라우팅 — 혼동 쌍(쌍둥이 쌍 2 + 재판 쌍 9)에서
top1·top2 확률차 ≤ 0.1일 때, 쌍둥이 제외로 학습한 쌍별 로지스틱(L2 C=0.1)이 확신 ≥ 0.5이면 덮어씀. train 쌍둥이가 있는 test 행은 제외.
정직 CV: 기준 0.4847 → 0.4883 (+0.0036, 교차선택 동일). 입력 확률은 9차의 저장 test 확률(git 포함).

재현: PYTHONPATH="4. src/common" python3 "4. src/common/approach13_routing/approach13_v1_routing.py" --submit approach13_v1_20260911_1602
설명 문서: 2. team/approaches/approach13.md
"""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(ROOT / "4. src/common/approach13_routing/approach13_v1_routing.py"), "--submit", "approach13_v1_20260911_1602"], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "4. src/common")}, check=True)
