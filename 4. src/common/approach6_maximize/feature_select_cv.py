"""접근 6 — 차원 축소: 학습 fold 안에서 카이제곱으로 유전자 이진 열을 상위 K개만 남기고 v5 XGB 학습 (다른 피처는 유지).
실행: PYTHONPATH="4. src/common" python3 ".../feature_select_cv.py" K
"""
import sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.feature_selection import chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, load_train, twin_groups

K = int(sys.argv[1]) if len(sys.argv) > 1 else 800
OUT = ROOT / "6. experiments" / f"approach6_fs{K}"; OUT.mkdir(parents=True, exist_ok=True)
tr = load_train(); le = LabelEncoder(); y = le.fit_transform(tr[TARGET]); C = len(le.classes_)
oof = np.zeros((len(tr), C)); t0 = time.time()
for k, (tri, vai) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, y, twin_groups(tr))):
    fm = FeatureMaker("v5").fit(tr.iloc[tri]); Xtr, Xva = fm.transform(tr.iloc[tri]), fm.transform(tr.iloc[vai])
    gcols = [c for c in Xtr.columns if c.startswith("g_")]; other = [c for c in Xtr.columns if not c.startswith("g_")]
    sc, _ = chi2(Xtr[gcols], y[tri]); keep = [gcols[i] for i in np.argsort(-np.nan_to_num(sc))[:K]]
    cols = keep + other
    m = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr[cols], y[tri]); oof[vai] = m.predict_proba(Xva[cols])
    print(f"fold{k} genes {len(gcols)}→{K}, total {len(cols)} f1={f1_score(y[vai], oof[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
np.save(OUT / "oof_proba.npy", oof); r = round(f1_score(y, oof.argmax(1), average="macro"), 4)
(OUT / "result.json").write_text(json.dumps({"K": K, "oof_macro_f1": r})); print(f"== fs{K} OOF macroF1={r} (v5 전체 0.4761)")
