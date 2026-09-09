"""v4 + 시드 배깅: XGB를 시드만 바꿔 N번 학습해 확률 평균. 분산을 줄여 LB로 옮겨지는 안정적 개선을 노린다.
정직 CV OOF(시드별)를 저장하고 평균 OOF의 Macro F1을 단일 시드와 비교한다.
실행: PYTHONPATH=src/common python3 src/common/approach2_knowledge/seed_bagging_v4.py [n_seeds]
산출: experiments/approach2_knowledge_v4_bag/{oof_seed*.npy, oof_proba.npy, result.json}
"""
import json, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, TARGET, XGB_PARAMS, FeatureMaker, load_train, twin_groups

OUT = ROOT / "experiments" / "approach2_knowledge_v4_bag"
SEEDS = [42, 7, 123, 2024, 31337]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    train = load_train(); le = LabelEncoder(); y = le.fit_transform(train[TARGET]); K = len(le.classes_)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, y, twin_groups(train)))
    oofs = []
    for s in SEEDS[:n]:
        P = np.zeros((len(train), K))
        for k, (tri, vai) in enumerate(folds):
            fm = FeatureMaker("v4").fit(train.iloc[tri])
            m = xgb.XGBClassifier(**{**XGB_PARAMS, "random_state": s}).fit(fm.transform(train.iloc[tri]), y[tri])
            P[vai] = m.predict_proba(fm.transform(train.iloc[vai]))
        np.save(OUT / f"oof_seed{s}.npy", P); oofs.append(P)
        print(f"[seed {s}] macroF1={f1_score(y, P.argmax(1), average='macro'):.4f} | 누적평균 {f1_score(y, np.mean(oofs, 0).argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
    avg = np.mean(oofs, 0); np.save(OUT / "oof_proba.npy", avg)
    res = dict(seeds=SEEDS[:n], single=[round(f1_score(y, p.argmax(1), average="macro"), 4) for p in oofs],
               bagged=round(f1_score(y, avg.argmax(1), average="macro"), 4))
    (OUT / "result.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res))


if __name__ == "__main__":
    main()
