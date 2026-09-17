"""approach11 v18: 혜림님 approach19의 정확한 로직(top-40 유전자 변이율차, LogisticRegression C=0.1,
신뢰도 임계값 0.70)을 v11 캐시(같은 v7 메인모델)에 재현해서, class_scale 그리드를
"기존(0.5~3.0)" vs "확장(0.1~5.0)"으로 바꿨을 때 그녀의 라우팅 결과가 실제로 좋아지는지/나빠지는지 직접 비교.
"""
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

GRID_ORIG = np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
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

    def evaluate(grid):
        scales = fit_class_scales(oof_main, y, grid=grid)
        base = (oof_main * scales).argmax(1)
        use = eligible & np.isin(base, (a, b))
        routed = base.copy(); routed[use] = specialist[use]
        return (f1_score(y, base, average="macro"), f1_score(y, routed, average="macro"),
                scales[a], scales[b])

    base_orig, routed_orig, s_hnsc_o, s_stes_o = evaluate(GRID_ORIG)
    base_wide, routed_wide, s_hnsc_w, s_stes_w = evaluate(GRID_WIDE)

    row = dict(seed=sd,
               base_origgrid=round(base_orig, 5), routed_origgrid=round(routed_orig, 5),
               base_widegrid=round(base_wide, 5), routed_widegrid=round(routed_wide, 5),
               stes_scale_orig=s_stes_o, stes_scale_wide=s_stes_w)
    print(row, flush=True)
    summary.append(row)

df = pd.DataFrame(summary)
print("\n=== 결과 ===")
print(df.to_string(index=False))
print("\n=== 핵심 비교: 혜림님 라우팅 적용 후, 기존 그리드 vs 확장 그리드 ===")
diff = df["routed_widegrid"] - df["routed_origgrid"]
print(f"차이 평균={diff.mean():.5f}  표준편차={diff.std(ddof=1):.5f}  확장이 더 나은 시드={int((diff>0).sum())}/5")
