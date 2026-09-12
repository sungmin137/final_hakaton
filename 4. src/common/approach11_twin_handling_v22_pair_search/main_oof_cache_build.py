"""메인모델(v7: features v4, colsample_bytree .7) OOF를 반복 시드별로 한 번만 계산해 캐시로 저장.

여러 암종 쌍 라우터 후보(LUSC/STES, LUAD/LUSC, UCEC/CESC, COAD/STES, THYM/LAML/THCA 묶음 …)를
검증할 때 이 메인모델 부분은 완전히 동일한데 매번 재학습하는 게 비효율적이라 분리함.
이 스크립트로 한 번만 만들어두면, 이후 쌍별 검증은 pair_router_check.py로 몇 초 안에 끝난다.
test.csv는 읽지 않는다.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import DATA, ROOT, XGB_PARAMS, FeatureMaker, gene_columns, twin_groups
from main import xgb

REPEAT_SPLIT_SEEDS = (13, 77, 202)
CACHE_PATH = Path(__file__).resolve().parent / "main_oof_cache.pkl"


def main():
    train = pd.read_csv(DATA / "train.csv")
    genes = gene_columns(train)
    Xbin = (train[genes].to_numpy() != "WT").astype(np.int8)
    le = LabelEncoder(); y = le.fit_transform(train["SUBCLASS"])
    groups = twin_groups(train)

    cache = {"classes": le.classes_, "y": y, "genes": genes, "Xbin": Xbin, "seeds": {}}
    t0 = time.time()
    for split_seed in REPEAT_SPLIT_SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        raw = np.zeros((len(y), len(le.classes_)), dtype=np.float32)
        fold_of = np.full(len(y), -1)
        for fold, (tri, vai) in enumerate(cv.split(train, y, groups)):
            fm = FeatureMaker("v4").fit(train.iloc[tri])
            model = xgb.XGBClassifier(**{**XGB_PARAMS, "colsample_bytree": .7, "random_state": 42})
            model.fit(fm.transform(train.iloc[tri]), y[tri])
            raw[vai] = model.predict_proba(fm.transform(train.iloc[vai]))
            fold_of[vai] = fold
            print(f"[cache] seed={split_seed} fold={fold} elapsed={time.time()-t0:.0f}s", flush=True)
        cache["seeds"][split_seed] = {"raw": raw, "fold_of": fold_of}

    CACHE_PATH.write_bytes(pickle.dumps(cache))
    print(f"저장 완료: {CACHE_PATH} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
