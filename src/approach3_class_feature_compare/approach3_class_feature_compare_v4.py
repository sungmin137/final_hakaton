"""접근 3 — 각 클래스 특징별 비교 v4: 변이 개수 구간(중간 부담 31~300개) 전용 전문가.

배경: 변이 31~100개 구간은 상피암(STES/LUSC/HNSC/LUAD/BLCA/COAD)이 뒤섞여 있고, 모델이 STES로 몰아 예측한다(train OOF에서도 정밀도 낮음).
      이 구간만 따로 학습한 전문가가 구간 안의 구분을 더 잘하는지 확인한다. 라우팅 기준은 샘플 자체의 변이 개수(피처)라 test 통계와 무관.
방법: 학습 fold에서 n_mut ∈ [31, 300] 인 행만으로 XGB(v4 피처) 학습 → 검증 fold의 같은 구간 행에 예측.
      전체 모델(v3 앙상블 OOF)과 구간 안에서 비교하고, 로그 가중 블렌딩 효과를 절반 교차확인.
산출: experiments/approach3_class_feature_compare_v4/{oof_mid.npy, result.json}
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, load_train, twin_groups, gene_columns

OUT = ROOT / "experiments" / "approach3_class_feature_compare_v4"; LO, HI = 31, 300


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    train = load_train(); le = LabelEncoder(); y = le.fit_transform(train[TARGET]); K = len(le.classes_)
    nm = (train[gene_columns(train)] != "WT").sum(axis=1).to_numpy(); mid = (nm >= LO) & (nm <= HI)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, y, twin_groups(train)))
    base = np.load(ROOT / "experiments/approach3_class_feature_compare_v3/oof_proba.npy")
    oof = base.copy()                      # 구간 밖은 기존 앙상블 그대로
    for k, (tri, vai) in enumerate(folds):
        tri_m, vai_m = tri[mid[tri]], vai[mid[vai]]
        fm = FeatureMaker("v4").fit(train.iloc[tri_m])
        present = np.unique(y[tri_m]); remap = {c: i for i, c in enumerate(present)}   # XGB는 연속 라벨 필요
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(fm.transform(train.iloc[tri_m]), np.array([remap[c] for c in y[tri_m]]))
        p = np.zeros((len(vai_m), K)); p[:, present] = m.predict_proba(fm.transform(train.iloc[vai_m]))
        oof[vai_m] = p
        print(f"[mid] fold{k} n_train={len(tri_m)} n_val={len(vai_m)} in-bucket F1 mid={f1_score(y[vai_m], p.argmax(1), average='macro'):.3f} base={f1_score(y[vai_m], base[vai_m].argmax(1), average='macro'):.3f} ({time.time()-t0:.0f}s)", flush=True)
    np.save(OUT / "oof_mid.npy", oof)
    f = lambda P, ii=slice(None): f1_score(y[ii], P[ii].argmax(1), average="macro")
    res = dict(base_all=round(f(base), 4), mid_replace_all=round(f(oof), 4),
               base_in_bucket=round(f(base, mid), 4), mid_in_bucket=round(f(oof, mid), 4))
    # 구간 안에서 base와 mid 로그 블렌딩 가중치 탐색 + 절반 교차확인
    def blendw(w):
        z = np.log(base + 1e-6); z[mid] = z[mid] + w * np.log(oof[mid] + 1e-6); return z
    grid = [0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rng = np.random.RandomState(0); idx = rng.permutation(len(y)); A, B = idx[: len(y) // 2], idx[len(y) // 2:]
    cc = []
    for fi, ei in [(A, B), (B, A)]:
        w = max(grid, key=lambda w: f(blendw(w), fi)); cc.append(dict(w=w, heldout_gain=round(f(blendw(w), ei) - f(base, ei), 4)))
    w_all = max(grid, key=lambda w: f(blendw(w))); res.update(best_w=w_all, blend_all=round(f(blendw(w_all)), 4), cross_check=cc)
    np.save(OUT / "oof_proba.npy", np.exp(blendw(w_all)) / np.exp(blendw(w_all)).sum(1, keepdims=True))
    (OUT / "result.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
