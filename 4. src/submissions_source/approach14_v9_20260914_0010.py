"""접근 14 v9: 16차 블렌드(9차+v2 로그평균 w=0.5) + 세 번째 모델(이진 유전자 행렬 로지스틱 회귀 C=1.0) 로그 결합 w3=0.2 → 복원 배율 → 쌍둥이 규칙.
한 번에 한 요소: 16차 대비 바뀐 요소는 세 번째 모델뿐(두 모델 확률은 16차 저장본). 정직 CV 0.4941 → 0.5010 (+0.007, 절반 교차확인 +0.004/+0.007, 31~100 구간 0.436→0.459).

재현: python3 "4. src/submissions_source/approach14_v9_20260914_0010.py"  →  5. submissions/approach14_v9_20260914_0010.csv (+ twin_rule/ 변형본)
설정: w=0.5, third=LR(C=1.0, 이진 유전자), w3=0.2, 배율=3차 복원 벡터 고정
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run, third_lr_binary
run("approach14_v9_20260914_0010", w=0.5, third=lambda tr, te: third_lr_binary(tr, te, C=1.0), w3=0.2)
