"""접근18 — v7 주변의 마일드 규제 그리드, train-only 정직 CV.

피처는 V4/approach2 계열로 고정한다. v7(colsample=.7) 주변에서 한 번에 한 축만
아주 작게 움직여, 원본 유전자 힌트를 버리지 않고 과한 암기만 줄일 수 있는지 본다.
test.csv는 읽지 않는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import DATA, ROOT, SEED, XGB_PARAMS, FeatureMaker, twin_groups  # noqa: E402
from main import xgb  # noqa: E402


# 기준 v7 + 오직 한 축만 바꾼 후보들. v11/v12처럼 여러 축을 강하게 동시에 바꾸지 않는다.
CONFIGS = {
    "v7_ref_col0.7": {"colsample_bytree": 0.7},
    "A_col0.8": {"colsample_bytree": 0.8},
    "B_col0.6": {"colsample_bytree": 0.6},
    "C_leaf_mild": {"colsample_bytree": 0.7, "min_child_weight": 2, "reg_lambda": 2.0},
    "D_depth5": {"colsample_bytree": 0.7, "max_depth": 5},
    "E_subsample0.9": {"colsample_bytree": 0.7, "subsample": 0.9},
}


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    le = LabelEncoder()
    y = le.fit_transform(train["SUBCLASS"])
    groups = twin_groups(train)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {name: np.zeros((len(train), len(le.classes_)), dtype=np.float32) for name in CONFIGS}
    folds = {name: [] for name in CONFIGS}
    t0 = time.time()

    for fold, (tri, vai) in enumerate(cv.split(train, y, groups), 1):
        print(f"\n========== Fold {fold} ==========" , flush=True)
        # 같은 fold의 피처는 딱 한 번 만들고 모든 설정이 공유한다.
        fm = FeatureMaker("v4").fit(train.iloc[tri])
        xtr, xva = fm.transform(train.iloc[tri]), fm.transform(train.iloc[vai])
        print(f"features={xtr.shape[1]}", flush=True)
        for name, change in CONFIGS.items():
            params = {**XGB_PARAMS, **change, "random_state": SEED}
            model = xgb.XGBClassifier(**params).fit(xtr, y[tri])
            proba = model.predict_proba(xva)
            oof[name][vai] = proba
            pred = proba.argmax(1)
            f1 = float(f1_score(y[vai], pred, average="macro"))
            acc = float(accuracy_score(y[vai], pred))
            folds[name].append({"fold": fold, "macro_f1": f1, "accuracy": acc})
            print(f"  {name}: F1={f1:.4f}, acc={acc:.4f}", flush=True)
        print(f"elapsed={time.time() - t0:.0f}s", flush=True)

    summary = []
    for name in CONFIGS:
        pred = oof[name].argmax(1)
        summary.append({
            "config": name,
            "changes": CONFIGS[name],
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)),
            "delta_vs_v7": 0.0,  # 아래에서 기준 대비 계산
        })
    ref = next(r["macro_f1"] for r in summary if r["config"] == "v7_ref_col0.7")
    for row in summary:
        row["delta_vs_v7"] = row["macro_f1"] - ref
    summary.sort(key=lambda r: r["macro_f1"], reverse=True)

    result = {
        "description": "V4 피처 고정, v7 주변 단일축 마일드 규제 grid; test 미사용",
        "group_twins": True,
        "configs": CONFIGS,
        "summary": summary,
        "folds": folds,
        "elapsed_seconds": round(time.time() - t0, 1),
        "warning": "CV는 제출 후보를 거르는 용도이며 LB 점수를 예측하지 않는다.",
    }
    out = ROOT / "6. experiments" / "2026-09-11_approach18_mild_grid"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for name, p in oof.items():
        np.save(out / f"oof_{name}.npy", p)
    print("\n==============================", flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
