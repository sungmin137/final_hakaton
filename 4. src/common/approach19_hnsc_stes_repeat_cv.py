"""접근19 — HNSC↔STES soft router를 새 정직 CV 분할에서 반복 검증.

approach20의 seed=42에서 미리 고정된 한 가지 설정(C=.1, confidence=.7)만 재검증한다.
실행 중 새 파라미터를 고르지 않으며 test.csv는 읽지 않는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import DATA, ROOT, XGB_PARAMS, FeatureMaker, gene_columns, twin_groups  # noqa: E402
from main import xgb  # noqa: E402
from postprocess.class_scale import fit_class_scales  # noqa: E402

REPEAT_SPLIT_SEEDS = (13, 77, 202)
PAIR = ("HNSC", "STES")
C, THRESHOLD = 0.1, 0.70


def metric(y, pred):
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred))}


def one_seed(train, Xbin, y, groups, ids, split_seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    raw = np.zeros((len(y), len(ids)), dtype=np.float32)
    specialist = np.zeros(len(y), dtype=np.int16)
    eligible = np.zeros(len(y), dtype=bool)
    a, b = ids[PAIR[0]], ids[PAIR[1]]
    t0 = time.time()
    for fold, (tri, vai) in enumerate(cv.split(train, y, groups), 1):
        fm = FeatureMaker("v4").fit(train.iloc[tri])
        model = xgb.XGBClassifier(**{**XGB_PARAMS, "colsample_bytree": .7, "random_state": 42})
        model.fit(fm.transform(train.iloc[tri]), y[tri])
        raw[vai] = model.predict_proba(fm.transform(train.iloc[vai]))
        ptr = tri[np.isin(y[tri], (a, b))]
        pa = Xbin[ptr][y[ptr] == a].mean(0); pb = Xbin[ptr][y[ptr] == b].mean(0)
        top = np.argsort(np.abs(pa - pb))[-40:]
        clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000,
                                 random_state=42 + fold).fit(Xbin[ptr][:, top], y[ptr])
        p = clf.predict_proba(Xbin[vai][:, top])
        specialist[vai] = clf.classes_[p.argmax(1)]
        # 이 단계에서는 raw main prediction으로 후보군을 정하고, 뒤에 class scale을 적용한다.
        eligible[vai] = np.isin(raw[vai].argmax(1), (a, b)) & (p.max(1) >= THRESHOLD)
        print(f"seed={split_seed} fold={fold} elapsed={time.time()-t0:.0f}s", flush=True)
    scales = fit_class_scales(raw, y)
    base = (raw * scales).argmax(1)
    # 보정 후 main 답이 두 클래스인 경우에만 덮어써, 현재 제출 구조와 맞춘다.
    use = eligible & np.isin(base, (a, b))
    routed = base.copy(); routed[use] = specialist[use]
    changed = routed != base
    return {"split_seed": split_seed, "base": metric(y, base), "router": metric(y, routed),
            "delta_macro_f1": float(f1_score(y, routed, average="macro") - f1_score(y, base, average="macro")),
            "changed": int(changed.sum()), "helped": int((changed & (routed == y) & (base != y)).sum()),
            "hurt": int((changed & (routed != y) & (base == y)).sum()),
            "seconds": round(time.time()-t0, 1)}


def main():
    train = pd.read_csv(DATA / "train.csv")
    genes = gene_columns(train); Xbin = (train[genes].to_numpy() != "WT").astype(np.int8)
    le = LabelEncoder(); y = le.fit_transform(train["SUBCLASS"]); ids = {x:i for i,x in enumerate(le.classes_)}
    groups = twin_groups(train)
    rows = [one_seed(train, Xbin, y, groups, ids, seed) for seed in REPEAT_SPLIT_SEEDS]
    d = np.array([r["delta_macro_f1"] for r in rows])
    result = {"description": "HNSC/STES C=.1 threshold=.7 fixed repeat SGKF; test 미사용", "rows": rows,
              "mean_delta": float(d.mean()), "std_delta": float(d.std()), "positive_seeds": int((d > 0).sum()),
              "decision": "반복 분할 평균 개선이 표준편차보다 작거나 음수 seed가 있으면 제출 후보 탈락"}
    out = ROOT / "6. experiments" / "2026-09-12_approach19_hnsc_stes_repeat"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
