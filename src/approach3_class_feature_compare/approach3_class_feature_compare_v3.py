"""접근 3 — 각 클래스 특징별 비교 v3: CatBoost 전문가 추가 + 4-모델 로그 가중 앙상블.

전문가:
  driver, burden : v1 OOF 재사용
  full           : v4 XGB OOF 재사용
  cat            : CatBoost, 요약 피처만 사용 (cw_*, n_*, kf_*, hs_*, combo_*, lof_*, is_*, 구간) → 전체 유전자 4,230개 제외로 속도 확보
앙상블: log p = log p_full + w_d·log p_driver + w_b·log p_burden + w_c·log p_cat, 격자 탐색 + 절반 교차확인.

실행: PYTHONPATH=src python3 src/approach3_class_feature_compare/approach3_class_feature_compare_v3.py
산출: experiments/approach3_class_feature_compare_v3/{oof_cat.npy, oof_proba.npy(앙상블), result.json, per_class_f1.csv}
"""
from __future__ import annotations

import itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, TARGET, FeatureMaker, load_train, twin_groups

V1 = ROOT / "experiments" / "approach3_class_feature_compare_v1"
OUT = ROOT / "experiments" / "approach3_class_feature_compare_v3"
CAT_COLS = lambda c: not c.startswith("g_")           # 요약 피처만 (유전자 이진화 제외)
CAT_PARAMS = dict(iterations=500, learning_rate=0.08, depth=6, l2_leaf_reg=3, random_seed=SEED,
                  loss_function="MultiClass", thread_count=8, verbose=0)
GRID = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


def blend4(P, w):
    z = np.log(P["full"] + 1e-6) + w[0] * np.log(P["driver"] + 1e-6) + w[1] * np.log(P["burden"] + 1e-6) + w[2] * np.log(P["cat"] + 1e-6)
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    train = load_train(); le = LabelEncoder(); y = le.fit_transform(train[TARGET]); K = len(le.classes_)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, y, twin_groups(train)))
    P = {"full": np.load(ROOT / "experiments/2026-09-09_v4_xgb_grp/oof_proba.npy"),
         "driver": np.load(V1 / "oof_driver.npy"), "burden": np.load(V1 / "oof_burden.npy")}
    cat = np.zeros((len(train), K))
    for k, (tri, vai) in enumerate(folds):
        fm = FeatureMaker("v4").fit(train.iloc[tri]); Xtr, Xva = fm.transform(train.iloc[tri]), fm.transform(train.iloc[vai])
        cols = [c for c in Xtr.columns if CAT_COLS(c)]
        m = CatBoostClassifier(**CAT_PARAMS).fit(Xtr[cols], y[tri]); cat[vai] = m.predict_proba(Xva[cols])
        print(f"[cat] fold{k} ncols={len(cols)} f1={f1_score(y[vai], cat[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
    P["cat"] = cat; np.save(OUT / "oof_cat.npy", cat)
    single = {n: round(f1_score(y, p.argmax(1), average="macro"), 4) for n, p in P.items()}
    f = lambda w, ii=slice(None): f1_score(y[ii], blend4({k: v[ii] for k, v in P.items()}, w).argmax(1), average="macro")
    cands = list(itertools.product(GRID, GRID, GRID))
    w_all = max(cands, key=f)
    rng = np.random.RandomState(0); idx = rng.permutation(len(y)); A, B = idx[: len(y) // 2], idx[len(y) // 2:]
    cc = []
    for fi, ei in [(A, B), (B, A)]:
        w = max(cands, key=lambda w: f(w, fi)); cc.append(dict(weights=w, heldout_gain_vs_full=round(f(w, ei) - f((0, 0, 0), ei), 4),
                                                          heldout_gain_vs_v2=round(f(w, ei) - f((0.5, 0.2, 0), ei), 4)))
    ens = blend4(P, w_all); np.save(OUT / "oof_proba.npy", ens)
    res = dict(single=single, weights=dict(driver=w_all[0], burden=w_all[1], cat=w_all[2]), ensemble_oof_f1=round(f(w_all), 4),
               ensemble_oof_acc=round(accuracy_score(y, ens.argmax(1)), 4), v2_oof_f1=round(f((0.5, 0.2, 0)), 4), cross_check=cc)
    per = pd.DataFrame({n: f1_score(y, p.argmax(1), average=None) for n, p in P.items()}, index=le.classes_)
    per["ensemble"] = f1_score(y, ens.argmax(1), average=None); per.round(3).to_csv(OUT / "per_class_f1.csv")
    (OUT / "result.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
