"""접근 2 v5: 3차 구성(v4 피처 + 클래스 배율 + 쌍둥이 규칙)의 극대화. fold 배깅을 시드 3개(15 fold 모델 + 전체 모델 = 16개 평균)로 확장하고,
클래스 배율은 OOF 절반 분할 6회에서 맞춘 값의 기하평균으로 안정화. 피처·모델 종류는 3차와 동일.

재현: python3 "4. src/submissions_source/approach2_v5_20260910_1453.py"  →  5. submissions/approach2_v5_20260910_1453.csv (+ _twin_rule)
설정: --features v4 --fold-bag --bag-seeds 42,7,123 --scale-avg --class-scale "6. experiments/2026-09-09_v4_xgb_grp"
설명 문서: 2. team/approaches/approach2.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4", "--fold-bag", "--bag-seeds", "42,7,123", "--scale-avg",
            "--class-scale", "6. experiments/2026-09-09_v4_xgb_grp", "--tag", "approach2_v5_20260910_1453"]
make_submission.main()
