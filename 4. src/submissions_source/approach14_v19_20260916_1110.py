"""접근 14 v19: v18과 같은 5모델 구성이지만 **버그 수정된 v4sn**(v4s + 혜림 NB 점수표 26열 피처, cw_·인사이트·지식 골격 포함) 사용.
v4s .25 / v4sn .2 / v2 .25 / v4sp .15 / NB .15. 정직 CV: v4sn 단독 0.5324(단일 모델 최고), 블렌드 0.5352 (접근16 v1 0.5309 +0.0043, 절반 교차 +0.003/+0.006, train 바뀐 행 359).
v18(27차 0.4639)은 골격이 빠진 v4sn(단독 0.4666)이었음. 전부 train 근거.

재현: python3 "4. src/submissions_source/approach14_v19_20260916_1110.py"  →  5. submissions/approach14_v19_20260916_1110.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, model_proba, ROOT
from features.spectrum_nb import spectrum_counts
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
v4s = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
v4sp = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v12_20260914_1627/test_proba_part3.npy")
def nb(tr, te):
    genes = [c for c in tr.columns if c not in ("ID", "SUBCLASS")]; y = LabelEncoder().fit_transform(tr["SUBCLASS"])
    return MultinomialNB(alpha=0.5).fit(spectrum_counts(tr[genes].to_numpy()), y).predict_proba(spectrum_counts(te[genes].to_numpy()))
run_multi("approach14_v19_20260916_1110", [(v4s, 0.25), (model_proba("v4sn"), 0.2), ("p2", 0.25), (v4sp, 0.15), (nb, 0.15)])
