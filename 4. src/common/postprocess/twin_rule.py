"""쌍둥이(완전 동일 프로필) 규칙. 3. docs/04_duplicate_twins.md 참고.

fit(train)  : 변이가 1개 이상인 학습 행의 프로필 해시 → 라벨 목록
apply(df, pred): df 행이 학습 행과 완전히 같으면 FLIP 규칙으로 pred를 덮어쓴다.
"""
import hashlib
import numpy as np
import pandas as pd

FLIP = {"KIPAN": "KIRC", "KIRC": "KIPAN", "GBMLGG": "LGG", "LGG": "GBMLGG"}


def _hash_rows(df: pd.DataFrame, genes: list[str]) -> np.ndarray:
    G = df[genes].to_numpy()
    return np.array([hashlib.md5("|".join(r).encode()).hexdigest() for r in G])


class TwinRule:
    def __init__(self):
        self.lut: dict[str, list[str]] = {}
        self.genes: list[str] = []

    def fit(self, train: pd.DataFrame) -> "TwinRule":
        self.genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
        h = _hash_rows(train, self.genes)
        n_mut = (train[self.genes] != "WT").sum(axis=1).to_numpy()
        for hh, lab, n in zip(h, train["SUBCLASS"], n_mut):
            if n > 0:
                self.lut.setdefault(hh, []).append(lab)
        return self

    def apply(self, df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, int]:
        h = _hash_rows(df, self.genes)
        out = pred.astype(object).copy(); n_hit = 0
        for i, hh in enumerate(h):
            labs = self.lut.get(hh)
            if labs:
                twin = max(set(labs), key=labs.count)
                out[i] = FLIP.get(twin, twin); n_hit += 1
        return out, n_hit
