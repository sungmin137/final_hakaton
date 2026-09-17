"""approach11 v5: v7(colsample_bytree 0.7, 복원 배율, LB 0.4377 최고) + GBMLGG/LGG 라우팅 앙상블 결합.

재현: python3 "4. src/submissions_source/approach11_v5_20260911_1420.py"  →  5. submissions/approach11_v5_20260911_1420.csv
설정: features v4, XGB mild_col(colsample_bytree 0.7) + 복원 배율, GBMLGG/LGG 전용 서브모델 라우팅(approach11_v4 로직)
정직 CV: mild_col 메인 단독 0.4669, 라우팅 적용 시 0.4693 (+0.0024)
LB: 0.4331 (v7 단독 0.4377 대비 −0.0046, 라우팅 역효과 — CV↔LB 무상관 재확인)
설명 문서: 2. team/approaches/approach11.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach11_twin_handling.approach11_v5_v7_routing_submission import main

main("approach11_v5_20260911_1420")
