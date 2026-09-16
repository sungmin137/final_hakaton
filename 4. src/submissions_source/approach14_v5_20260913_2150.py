"""접근 14 v5: 16차(v3) 구성에서 블렌드 가중치만 w=0.3 (9차 0.7 : v2 0.3). 라우터 없음. 복원 배율 + 쌍둥이 규칙.
한 번에 한 요소: 16차 대비 바뀐 요소는 w뿐. 두 모델 확률은 16차 저장본 재사용.

재현: python3 "4. src/submissions_source/approach14_v5_20260913_2150.py"  →  5. submissions/approach14_v5_20260913_2150.csv (+ twin_rule/ 변형본)
설정: w=0.3, router=False
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run
run("approach14_v5_20260913_2150", w=0.3)
