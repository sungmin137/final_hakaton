"""암종별 접근을 달리하는 전문가 모델 + 스태킹 (사용자 제안 2026-09-09).

전문가:
  driver : 유전자 변이 유무(g_*), hotspot 위치(hs_*), LoF 유전자(lof_*), 조합(combo_*)      ← "driver 비율이 높은 암종"
  burden : 변이 개수·유형 비율·구간·플래그(n_*, log_*, syn_ratio, is_*, kf_* 집계)          ← "변이 개수로 갈리는 암종"
  full   : v4 전체 (기존 OOF 재사용: experiments/…_v4_xgb_grp)
메타   : 세 전문가의 OOF 확률(로그)을 입력으로 다항 로지스틱 회귀. 같은 정직 fold로 cross_val_predict.
산출   : 정직 CV Macro F1 (전문가별 / 스태킹), 암종별 "어느 전문가가 가장 잘 맞히나" 표.
train.csv만 사용. 실행: PYTHONPATH=src python3 src/experts.py
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, load_train, twin_groups

OUT = ROOT / "experiments" / "2026-09-09_experts_stack"; OUT.mkdir(parents=True, exist_ok=True)
EXPERTS = {
    "driver": lambda c: c.startswith(("g_", "hs_", "lof_", "combo_")),
    "burden": lambda c: c.startswith(("n_", "log_", "syn_ratio", "is_", "kf_", "lof_gene_ratio")) and not c.startswith("lof_") or c in ("n_lof_genes",),
}


def main():
    t0 = time.time()
    train = load_train(); le = LabelEncoder(); y = le.fit_transform(train[TARGET]); K = len(le.classes_)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, y, twin_groups(train)))
    oof = {"full": np.load(ROOT / "experiments/2026-09-09_v4_xgb_grp/oof_proba.npy")}
    for name, sel in EXPERTS.items():
        P = np.zeros((len(train), K))
        for k, (tri, vai) in enumerate(folds):
            fm = FeatureMaker("v4").fit(train.iloc[tri])
            Xtr, Xva = fm.transform(train.iloc[tri]), fm.transform(train.iloc[vai])
            cols = [c for c in Xtr.columns if sel(c)]
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr[cols], y[tri])
            P[vai] = m.predict_proba(Xva[cols])
            print(f"[{name}] fold{k} ncols={len(cols)} f1={f1_score(y[vai], P[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
        oof[name] = P; np.save(OUT / f"oof_{name}.npy", P)

    res = {n: dict(macro_f1=round(f1_score(y, p.argmax(1), average="macro"), 4), acc=round(accuracy_score(y, p.argmax(1)), 4)) for n, p in oof.items()}
    # 스태킹 메타 (같은 fold)
    Z = np.hstack([np.log(oof[n] + 1e-6) for n in ("driver", "burden", "full")])
    S = np.zeros((len(train), K))
    for tri, vai in folds:
        lr = LogisticRegression(C=0.3, max_iter=2000).fit(Z[tri], y[tri]); S[vai] = lr.predict_proba(Z[vai])
    res["stack"] = dict(macro_f1=round(f1_score(y, S.argmax(1), average="macro"), 4), acc=round(accuracy_score(y, S.argmax(1)), 4))
    np.save(OUT / "oof_stack.npy", S)
    # 암종별 최적 전문가
    per = {}
    for n, p in {**oof, "stack": S}.items():
        per[n] = f1_score(y, p.argmax(1), average=None)
    table = pd.DataFrame(per, index=le.classes_).round(3)
    table["best_expert"] = table[["driver", "burden", "full"]].idxmax(axis=1)
    table.to_csv(OUT / "per_class_f1.csv")
    print("\n== 정직 CV ==", json.dumps(res, ensure_ascii=False))
    print("\n== 암종별 F1 (driver / burden / full / stack) ==")
    print(table.sort_values("full").to_string())
    print("\n전문가별 '최고' 암종 수:", table.best_expert.value_counts().to_dict())
    (OUT / "result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
