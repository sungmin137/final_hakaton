"""접근 14 v17: 혜림 접근16(치환 프로필 MultinomialNB α0.5 파트너 .15) + 성민 스펙트럼 모델군(v4s / v2 / v4sp)을 한 파일에. 가중 v4s .4 / v2 .25 / v4sp .2 / NB .15.
정직 OOF 0.5317 (접근16 v1 0.5309 +0.0008; 절반 교차 +0.0017/−0.0003 → 사실상 동급의 변형). 전부 train 근거.
v4s·v4sp·v2 확률은 저장본, NB는 train 전체 fit(초 단위) → test predict_proba.

재현: python3 "4. src/submissions_source/approach14_v17_20260915_1345.py"  →  5. submissions/approach14_v17_20260915_1345.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md, approach16.md
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from approach14_cw_plus.blend import run_multi, ROOT
from features.spectrum_nb import spectrum_counts
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
v4s = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
v4sp = lambda tr, te: np.load(ROOT / "6. experiments/submissions/approach14_v12_20260914_1627/test_proba_part3.npy")
def nb(tr, te):
    genes = [c for c in tr.columns if c not in ("ID", "SUBCLASS")]; y = LabelEncoder().fit_transform(tr["SUBCLASS"])
    return MultinomialNB(alpha=0.5).fit(spectrum_counts(tr[genes].to_numpy()), y).predict_proba(spectrum_counts(te[genes].to_numpy()))
run_multi("approach14_v17_20260915_1345", [(v4s, 0.4), ("p2", 0.25), (v4sp, 0.2), (nb, 0.15)])
