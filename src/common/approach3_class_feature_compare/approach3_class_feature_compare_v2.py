"""접근 3 — 각 클래스 특징별 비교 v2: 전문가 확률의 로그 가중 블렌딩 (기하평균).

v1(로지스틱 회귀 스태킹)은 full 단독보다 낮았다(0.4455 < 0.4691). 메타 모델이 과하게 자유로워 OOF에 과적합.
v2는 자유도를 2개(driver 가중치, burden 가중치)로 줄인 블렌딩:
    log p = log p_full + w_d · log p_driver + w_b · log p_burden
가중치는 train OOF에서 격자 탐색하고, 절반 교차확인으로 일반화 이득을 검증한다.

실행: PYTHONPATH=src/common python3 src/common/approach3_class_feature_compare/approach3_class_feature_compare_v2.py
입력: v1 산출물(experiments/approach3_class_feature_compare_v1/oof_*.npy) + v4 정직 CV OOF
산출: experiments/approach3_class_feature_compare_v2/result.json (최적 가중치, 교차확인 이득)
추론: make_submission.py --approach3-blend  (전문가를 train 전체로 학습해 test 확률을 블렌딩)
"""
from __future__ import annotations

import itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "experiments" / "approach3_class_feature_compare_v1"
OUT = ROOT / "experiments" / "approach3_class_feature_compare_v2"
GRID = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

# 전문가별 피처 선택 규칙 (v1과 동일)
EXPERT_COLS = {
    "driver": lambda c: c.startswith(("g_", "hs_", "lof_", "combo_")),
    "burden": lambda c: (c.startswith(("n_", "log_", "syn_ratio", "is_", "kf_", "lof_gene_ratio")) and not c.startswith("lof_")) or c == "n_lof_genes",
}


def blend(p_full: np.ndarray, p_driver: np.ndarray, p_burden: np.ndarray, w_d: float, w_b: float) -> np.ndarray:
    z = np.log(p_full + 1e-6) + w_d * np.log(p_driver + 1e-6) + w_b * np.log(p_burden + 1e-6)
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)


def search(P, y, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    best = max(itertools.product(GRID, GRID), key=lambda w: f1_score(y[idx], blend(P["full"][idx], P["driver"][idx], P["burden"][idx], *w).argmax(1), average="macro"))
    return best


def main():
    tr = pd.read_csv(ROOT / "info/data/train.csv", usecols=["SUBCLASS"]); classes = sorted(tr.SUBCLASS.unique())
    y = tr.SUBCLASS.map({c: i for i, c in enumerate(classes)}).to_numpy()
    P = {"full": np.load(ROOT / "experiments/2026-09-09_v4_xgb_grp/oof_proba.npy"),
         "driver": np.load(V1 / "oof_driver.npy"), "burden": np.load(V1 / "oof_burden.npy")}
    f = lambda w, ii=slice(None): f1_score(y[ii], blend(P["full"][ii], P["driver"][ii], P["burden"][ii], *w).argmax(1), average="macro")
    w_all = search(P, y); res = dict(weights=dict(driver=w_all[0], burden=w_all[1]), oof_f1=round(f(w_all), 4), full_f1=round(f((0, 0)), 4))
    rng = np.random.RandomState(0); idx = rng.permutation(len(y)); A, B = idx[: len(y) // 2], idx[len(y) // 2:]
    res["cross_check"] = []
    for fi, ei in [(A, B), (B, A)]:
        w = search(P, y, fi); res["cross_check"].append(dict(weights=w, heldout_gain=round(f(w, ei) - f((0, 0), ei), 4)))
    per = pd.DataFrame({"full": f1_score(y, P["full"].argmax(1), average=None),
                        "blend": f1_score(y, blend(P["full"], P["driver"], P["burden"], *w_all).argmax(1), average=None)}, index=classes).round(3)
    per["diff"] = (per.blend - per.full).round(3)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "result.json").write_text(json.dumps(res, indent=2)); per.to_csv(OUT / "per_class_f1.csv")
    print(json.dumps(res, ensure_ascii=False)); print(per.sort_values("diff", ascending=False).to_string())


if __name__ == "__main__":
    main()
