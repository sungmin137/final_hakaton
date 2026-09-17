"""접근 14 v10: 세 모델 로그 결합 — 9차(v4, 저장 확률) 0.4 / 접근14 v2(v4p, 저장 확률) 0.3 / 접근 7 전처리 확장 모델(a7, 전체 train 학습) 0.3 → 3차 복원 배율 → 쌍둥이 규칙.
한 번에 한 요소: 16차 대비 바뀐 요소는 a7 모델 추가뿐. 정직 CV 0.4941 → 0.5049 (+0.0108), 절반 교차 +0.009/+0.013, 31~100 구간 0.436→0.449, STES 예측 292→281.

재현: python3 "4. src/submissions_source/approach14_v10_20260914_0110.py"  →  5. submissions/approach14_v10_20260914_0110.csv (+ twin_rule/ 변형본)
설정: w=3/7 (2모델 블렌드 안에서 9차:v2 = 4:3), third=a7 XGB mild_col, w3=0.3 → 최종 가중 0.4/0.3/0.3
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run, third_a7
run("approach14_v10_20260914_0110", w=3/7, third=third_a7, w3=0.3)
