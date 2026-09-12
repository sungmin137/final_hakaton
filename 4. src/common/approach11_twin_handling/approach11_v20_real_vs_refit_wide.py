"""approach11 v20: v18의 실수 수정 재검증.
비교 A(기준선) = 실제 고정 배율 파일(모든 제출이 실제로 쓴 값) + 혜림님 라우팅
비교 B          = v7 캐시(진짜 v7 자신의 OOF)로 "처음부터" 넓은 그리드(0.1~5.0)로 새로 계산한 배율 + 혜림님 라우팅
→ 기존 고정 파일을 무비판적으로 재사용하는 관행 자체가, v7 전용으로 새로 맞춘 것보다 못한지 확인.
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, TARGET, twin_groups, load_train
from postprocess.class_scale import fit_class_scales

PAIR = ("HNSC", "STES")
C, THRESHOLD = 0.1, 0.70
SEEDS = [42, 7, 123, 2024, 999]
GRID_WIDE = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

cache_path = ROOT / "6. experiments" / "2026-09-12_approach11_v11_integration" / "oof_main_cache.pkl"
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

train = load_train()
genes = [c for c in train.columns if c not in ("ID", TARGET)]
Xbin = (train[genes].to_numpy() != "WT").astype(np.int8)
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
ids = {c: i for i, c in enumerate(le.classes_)}
a, b = ids[PAIR[0]], ids[PAIR[1]]

real_scales_dict = json.load(open(ROOT / "6. experiments" / "2026-09-09_v4_xgb_cs" / "class_scales_recovered.json"))
real_scales = np.array([float(real_scales_dict[c]) for c in le.classes_])

summary = []
for sd in SEEDS:
    oof_main = cache[sd]["oof_main"]
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)

    specialist = np.zeros(len(y), dtype=np.int64)
    eligible = np.zeros(len(y), dtype=bool)

    for fold, (tri, vai) in enumerate(skf.split(train, y, groups), 1):
        ptr = tri[np.isin(y[tri], (a, b))]
        pa = Xbin[ptr][y[ptr] == a].mean(0)
        pb = Xbin[ptr][y[ptr] == b].mean(0)
        top = np.argsort(np.abs(pa - pb))[-40:]
        clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000,
                                  random_state=42 + fold).fit(Xbin[ptr][:, top], y[ptr])
        p = clf.predict_proba(Xbin[vai][:, top])
        specialist[vai] = clf.classes_[p.argmax(1)]
        eligible[vai] = np.isin(oof_main[vai].argmax(1), (a, b)) & (p.max(1) >= THRESHOLD)
        print(f"  seed{sd} fold{fold} 완료", flush=True)

    def evaluate(scales):
        base = (oof_main * scales).argmax(1)
        use = eligible & np.isin(base, (a, b))
        routed = base.copy(); routed[use] = specialist[use]
        return f1_score(y, base, average="macro"), f1_score(y, routed, average="macro")

    base_real, routed_real = evaluate(real_scales)
    scales_fresh_wide = fit_class_scales(oof_main, y, grid=GRID_WIDE)
    base_fresh, routed_fresh = evaluate(scales_fresh_wide)

    row = dict(seed=sd, base_real=round(base_real, 5), routed_real=round(routed_real, 5),
               base_freshwide=round(base_fresh, 5), routed_freshwide=round(routed_fresh, 5))
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
print("\n=== 결과 ===")
print(df.to_string(index=False))
diff = df["routed_freshwide"] - df["routed_real"]
print(f"\n(v7 새로계산 넓은그리드+라우팅) - (진짜 고정배율+라우팅): mean={diff.mean():.5f} std={diff.std(ddof=1):.5f} 승={int((diff>0).sum())}/5")
