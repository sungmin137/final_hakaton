"""접근 5 v2 (통합, 블렌딩 제외): 접근 1 점수 + 접근 2 인사이트·지식 + 접근 4 문헌 피처(v5) → XGB, 클래스 배율 후처리. 접근 3 전문가 블렌딩은 뺐다.
v1(블렌딩 포함)과 LB를 비교해 블렌딩의 실제 효과를 확정하기 위한 파일. 정직 CV: v5 0.4761 + 배율(heldout 약 +0.01).

재현: python3 "4. src/submissions_source/approach5_v2_20260909_2322.py"  →  5. submissions/approach5_v2_20260909_2322.csv
설정: --features v5 --class-scale "6. experiments/2026-09-09_v5_xgb_grp"
설명 문서: 2. team/approaches/approach5.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v5", "--class-scale", "6. experiments/2026-09-09_v5_xgb_grp", "--tag", "approach5_v2_20260909_2322"]
make_submission.main()
