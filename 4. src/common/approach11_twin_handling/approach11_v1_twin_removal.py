"""쌍둥이 행(KIPAN/KIRC, GBMLGG/LGG 완전 동일 프로필)을 학습에서만 제외하고
검증은 원래(쌍둥이 포함) 그대로 하는 정직 CV 비교. 04_todo_day2.md STEP 4 참고.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, TARGET, FeatureMaker, XGB_PARAMS, SEED, twin_groups, load_train

train = load_train()
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
groups = twin_groups(train)

# 쌍둥이 행(라벨 섞인 그룹) 식별
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_mixed_twin = gid.map(label_sets).apply(lambda s: len(s) > 1)
print(f"쌍둥이(라벨 섞인) 행: {is_mixed_twin.sum()}개 / 전체 {len(train)}개")

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros((len(train), len(le.classes_)))
for k, (tri, vai) in enumerate(skf.split(train, y, groups)):
    tr_idx = train.iloc[tri].index
    keep = ~is_mixed_twin.loc[tr_idx]
    tri_clean = tri[keep.to_numpy()]
    print(f"fold{k}: 원래 학습 {len(tri)} -> 쌍둥이 제거 후 {len(tri_clean)}")

    tr_part, va_part = train.iloc[tri_clean], train.iloc[vai]
    fm = FeatureMaker("v4").fit(tr_part)
    Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
    model = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr, y[tri_clean])
    oof[vai] = model.predict_proba(Xva)
    pred = oof[vai].argmax(1)
    print(f"  fold{k} macroF1={f1_score(y[vai], pred, average='macro'):.4f}")

pred_all = oof.argmax(1)
macro = f1_score(y, pred_all, average="macro")
print(f"\n쌍둥이 학습 제거 OOF macroF1={macro:.4f}  (기존 v4 baseline=0.4683)")

pc = dict(zip(le.classes_, f1_score(y, pred_all, average=None).round(4).tolist()))
for c in ["KIRC", "KIPAN", "GBMLGG", "LGG"]:
    print(f"  {c}: {pc[c]}")

out_dir = ROOT / "6. experiments" / "2026-09-10_approach11_v1_twin_removal"
out_dir.mkdir(parents=True, exist_ok=True)
res = dict(features="v4", note="쌍둥이 행 학습 제거, 검증은 원본 유지",
           oof_macro_f1=round(macro, 4), oof_acc=round(accuracy_score(y, pred_all), 4),
           per_class_f1=pc)
(out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
np.save(out_dir / "oof_proba.npy", oof)
pd.DataFrame(confusion_matrix(y, pred_all), index=le.classes_, columns=le.classes_).to_csv(out_dir / "confusion_matrix.csv")
print(f"\nsaved: {out_dir}")
