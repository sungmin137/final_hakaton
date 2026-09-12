"""main_oof_cache_build.py가 만든 캐시를 재사용해, 암종 쌍 라우터 후보를 몇 초 안에 검증.

메인모델을 다시 학습하지 않고 캐시된 OOF(raw)와 fold 배정만 그대로 쓰고,
그 쌍에 한정된 가벼운 로지스틱회귀 서브모델만 새로 학습한다. test.csv는 읽지 않는다.

사용: python3 pair_router_check.py LUSC STES
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import TARGET, load_train, twin_groups
from postprocess.class_scale import fit_class_scales

CACHE_PATH = Path(__file__).resolve().parent / "main_oof_cache.pkl"
V11_CACHE_PATH = Path("6. experiments/2026-09-12_approach11_v11_integration/oof_main_cache.pkl")
C, THRESHOLD = 0.1, 0.70


def load_v11_cache_as_generic():
    """v11 캐시(oof_main만 있음)를 재사용 — fold 배정은 재학습 없이 그대로 재현(수 초)."""
    with open(V11_CACHE_PATH, "rb") as f:
        raw_cache = pickle.load(f)
    train = load_train()
    le = LabelEncoder(); y = le.fit_transform(train[TARGET])
    genes = [c for c in train.columns if c not in ("ID", TARGET)]
    Xbin = (train[genes].to_numpy() != "WT").astype(np.int8)
    groups = twin_groups(train)
    seeds = {}
    for sd, d in raw_cache.items():
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
        fold_of = np.full(len(y), -1)
        for fold, (_, vai) in enumerate(skf.split(train, y, groups)):
            fold_of[vai] = fold
        seeds[sd] = {"raw": d["oof_main"], "fold_of": fold_of}
    return {"classes": le.classes_, "y": y, "genes": genes, "Xbin": Xbin, "seeds": seeds}


def check_pair(cache, pair, top_k=40):
    classes = list(cache["classes"]); y = cache["y"]; Xbin = cache["Xbin"]
    ids = {c: i for i, c in enumerate(classes)}
    a, b = ids[pair[0]], ids[pair[1]]

    rows = []
    for split_seed, d in cache["seeds"].items():
        raw, fold_of = d["raw"], d["fold_of"]
        specialist = np.zeros(len(y), dtype=np.int16)
        eligible = np.zeros(len(y), dtype=bool)
        for fold in np.unique(fold_of):
            tri = np.flatnonzero(fold_of != fold)
            vai = np.flatnonzero(fold_of == fold)
            ptr = tri[np.isin(y[tri], (a, b))]
            if len(ptr) < 10:
                continue
            pa = Xbin[ptr][y[ptr] == a].mean(0); pb = Xbin[ptr][y[ptr] == b].mean(0)
            top = np.argsort(np.abs(pa - pb))[-top_k:]
            clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000,
                                     random_state=42 + int(fold)).fit(Xbin[ptr][:, top], y[ptr])
            p = clf.predict_proba(Xbin[vai][:, top])
            specialist[vai] = clf.classes_[p.argmax(1)]
            eligible[vai] = np.isin(raw[vai].argmax(1), (a, b)) & (p.max(1) >= THRESHOLD)
        scales = fit_class_scales(raw, y)
        base = (raw * scales).argmax(1)
        use = eligible & np.isin(base, (a, b))
        routed = base.copy(); routed[use] = specialist[use]
        rows.append({"split_seed": int(split_seed),
                     "base_f1": float(f1_score(y, base, average="macro")),
                     "router_f1": float(f1_score(y, routed, average="macro")),
                     "delta": float(f1_score(y, routed, average="macro") - f1_score(y, base, average="macro")),
                     "changed": int((routed != base).sum())})
    d = np.array([r["delta"] for r in rows])
    return {"pair": pair, "rows": rows, "mean_delta": float(d.mean()), "std_delta": float(d.std()),
            "positive_seeds": int((d > 0).sum()), "n_seeds": len(rows)}


if __name__ == "__main__":
    cache = load_v11_cache_as_generic()
    pair = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("LUSC", "STES")
    result = check_pair(cache, pair)
    print(f"=== {pair[0]} <-> {pair[1]} ===")
    for r in result["rows"]:
        print(f"  seed={r['split_seed']}: base={r['base_f1']:.5f} router={r['router_f1']:.5f} "
              f"delta={r['delta']:+.5f} changed={r['changed']}")
    print(f"mean_delta={result['mean_delta']:+.5f} std_delta={result['std_delta']:.5f} "
          f"positive_seeds={result['positive_seeds']}/{result['n_seeds']}")
    ok = result["mean_delta"] > result["std_delta"] and result["positive_seeds"] == result["n_seeds"]
    print("판정:", "통과(제출 후보)" if ok else "탈락(노이즈 수준 또는 음수 seed 존재)")
