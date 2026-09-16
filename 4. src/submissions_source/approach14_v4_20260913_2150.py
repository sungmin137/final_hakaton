"""접근 14 v4: 16차(v3, w=0.5 로그평균 + 복원 배율) + 혜림 님 HNSC/STES 소프트 라우터(approach19, conf≥0.7) + 쌍둥이 규칙.
한 번에 한 요소: 16차 대비 바뀐 요소는 라우터뿐. 두 모델 확률은 16차 저장본 재사용(재학습 없음).

재현: python3 "4. src/submissions_source/approach14_v4_20260913_2150.py"  →  5. submissions/approach14_v4_20260913_2150.csv (+ twin_rule/ 변형본)
설정: w=0.5, router=True, 배율=3차 복원 벡터 고정
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run
run("approach14_v4_20260913_2150", w=0.5, router=True)
