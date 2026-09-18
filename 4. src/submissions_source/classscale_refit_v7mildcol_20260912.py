"""v7(features v4, colsample_bytree=0.7) 전용 class_scale 재계산본.

⚠️ 파일명 잠정: "approach" 번호는 팀과 상의 후 확정 필요(기존 번호 체계와 충돌 방지, 특히 approach7은 사용 금지).

배경: v7 이후 모든 제출(v7~v12, 혜림님 v19)이 2026-09-09에 다른(v4_xgb_grp) 모델로 한 번 계산된
class_scale을 그대로 재사용해왔음(`6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json`).
v7 자신의 정직 CV OOF(성민님이 만든 공식 파일 `6. experiments/2026-09-11_v4_xgb_mild_col_grp/oof_proba.npy`,
oof_macro_f1=0.4669)로 배율을 새로 맞추면 CV가 크게(+0.014) 오름을 확인 — 그리드는 확장하지 않고
기존 9개 후보 그대로 사용(과적합 위험을 낮추기 위해 이 버전만 우선 시도).

재현: python3 "4. src/submissions_source/classscale_refit_v7mildcol_20260912.py"
설정: --features v4 --params mild_col --class-scale "6. experiments/2026-09-11_v4_xgb_mild_col_grp" (기존 그리드)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import make_submission

sys.argv = ["make_submission.py", "--features", "v4", "--params", "mild_col",
            "--class-scale", "6. experiments/2026-09-11_v4_xgb_mild_col_grp",
            "--tag", "classscale_refit_v7mildcol_20260912"]
make_submission.main()
