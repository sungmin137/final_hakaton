"""베이스라인: 피처 v1 + LightGBM, Stratified 5-Fold. train.csv만 사용.
실행: python3 src/train_baseline.py
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from features import build_features, gene_columns, TARGET

EXP = "2026-09-09_baseline_lgbm_v1"
OUT = Path("experiments") / EXP
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def main() -> None:
    t0 = time.time()
    tr = pd.read_csv("data/raw/train.csv")
    genes = gene_columns(tr)
    X = build_features(tr, genes)
    # 전부 WT인 유전자 제거 (train 기준, test는 이 컬럼 목록을 그대로 따른다)
    keep = [c for c in X.columns if not (c.startswith("g_") and X[c].sum() == 0)]
    X = X[keep]
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET])
    print(f"features {X.shape}, classes {len(le.classes_)}, prep {time.time()-t0:.1f}s")

    params = dict(
        objective="multiclass", num_class=len(le.classes_), learning_rate=0.05,
        num_leaves=31, min_data_in_leaf=10, feature_fraction=0.3, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=1.0, seed=SEED, verbose=-1, num_threads=8,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(le.classes_)))
    folds = []
    for k, (tri, vai) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[tri], y[tri])
        dva = lgb.Dataset(X.iloc[vai], y[vai])
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        pred = oof[vai].argmax(1)
        f1 = f1_score(y[vai], pred, average="macro"); acc = accuracy_score(y[vai], pred)
        folds.append(dict(fold=k, best_iter=m.best_iteration, macro_f1=f1, acc=acc))
        print(f"fold{k} iter={m.best_iteration} macroF1={f1:.4f} acc={acc:.4f}")

    pred = oof.argmax(1)
    res = dict(
        exp=EXP, n_features=X.shape[1], folds=folds,
        oof_macro_f1=f1_score(y, pred, average="macro"), oof_acc=accuracy_score(y, pred),
        per_class_f1=dict(zip(le.classes_, f1_score(y, pred, average=None).round(4).tolist())),
        elapsed_s=round(time.time() - t0, 1), params=params,
    )
    print(f"\nOOF macroF1={res['oof_macro_f1']:.4f} acc={res['oof_acc']:.4f} ({res['elapsed_s']}s)")
    print("per-class F1 (낮은 순):",
          sorted(res["per_class_f1"].items(), key=lambda kv: kv[1])[:8])

    cm = pd.DataFrame(confusion_matrix(y, pred), index=le.classes_, columns=le.classes_)
    cm.to_csv(OUT / "confusion_matrix.csv")
    np.save(OUT / "oof_proba.npy", oof)
    (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    pd.Series(keep).to_csv(OUT / "feature_columns.csv", index=False, header=False)


if __name__ == "__main__":
    main()
