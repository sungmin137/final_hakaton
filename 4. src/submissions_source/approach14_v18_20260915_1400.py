"""접근 14 v18: 두 방법을 가장 깊게 결합한 5모델. 성민 v4s(v4+스펙트럼) .25 / v4sn(v4s + 혜림 NB 점수표 26열을 피처로, 내부 OOF) .2 / v2 .25 / v4sp .15 / 혜림 NB 파트너 .15.
정직 OOF 0.5348 (접근16 v1 0.5309 +0.0039, 절반 교차 +0.0017/+0.0056, 31~100 0.507). v4sn 단독은 0.4666으로 약하지만 파트너로는 보탬. 전부 train 근거.
v4s·v4sp·v2·NB는 저장본/초 단위 fit, v4sn만 전체 train 학습(약 4분).

재현: python3 "4. src/submissions_source/approach14_v18_20260915_1400.py"  →  5. submissions/approach14_v18_20260915_1400.csv (+ twin_rule/ 변형본)
설명 문서: 2. team/approaches/approach14.md, approach16.md
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
run_multi("approach14_v18_20260915_1400", [(v4s, 0.25), (model_proba("v4sn"), 0.2), ("p2", 0.25), (v4sp, 0.15), (nb, 0.15)])
