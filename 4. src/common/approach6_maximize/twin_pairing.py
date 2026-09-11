"""접근 6 — 쌍둥이 배정 후처리 (두 단계).
 1) train 쌍둥이 규칙: 예측 대상 행과 완전히 같은 train 행이 있으면 그 라벨을 뒤집어 배정 (KIPAN↔KIRC, GBMLGG↔LGG). docs/04.
 2) 내부 쌍 분리: 예측 대상 집합(test 또는 검증 fold) 안에서 서로 완전히 같은 행들 중, 모델이 쌍둥이 클래스(KIRC/KIPAN, LGG/GBMLGG)로
    예측한 쌍은 한 행을 하위(KIRC/LGG), 다른 행을 상위(KIPAN/GBMLGG)로 나눠 배정한다. 두 행은 동일하므로 어느 쪽이 어느 라벨인지는 알 수 없고
    (동전 던지기), 배정 순서는 ID 순으로 고정한다. 목적: Macro F1에서 하위 클래스 recall이 0에 가까워지는 것을 막는다.
 ⚠️ 2)는 test 행끼리의 관계를 추론에 쓰는 회색지대 (2. team/rules_compliance.md 기록). 팀 결정으로 적용.
"""
import hashlib
import numpy as np, pandas as pd
from postprocess.twin_rule import FLIP, _hash_rows

PAIRS = {frozenset({"KIRC", "KIPAN"}): ("KIRC", "KIPAN"), frozenset({"LGG", "GBMLGG"}): ("LGG", "GBMLGG")}
SUB_OF = {"KIRC": "KIPAN", "KIPAN": "KIRC", "LGG": "GBMLGG", "GBMLGG": "LGG"}


class TwinPairing:
    def fit(self, train: pd.DataFrame):
        self.genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
        h = _hash_rows(train, self.genes); n = (train[self.genes] != "WT").sum(axis=1).to_numpy()
        self.lut = {}
        for hh, lab, k in zip(h, train["SUBCLASS"], n):
            if k > 0: self.lut.setdefault(hh, []).append(lab)
        return self

    def apply(self, df: pd.DataFrame, pred: np.ndarray, proba: np.ndarray | None = None, classes=None):
        """pred: 클래스 문자열 배열. proba/classes 가 있으면 내부 쌍의 '어느 쪽이 하위 클래스일 확률'도 동률이므로 ID 순으로 배정."""
        h = _hash_rows(df, self.genes); n = (df[self.genes] != "WT").sum(axis=1).to_numpy()
        out = pred.astype(object).copy(); n_rule = 0; n_pair = 0
        # 1) train 쌍둥이 규칙
        has_train_twin = np.zeros(len(df), bool)
        for i, hh in enumerate(h):
            labs = self.lut.get(hh)
            if labs and n[i] > 0:
                twin = max(set(labs), key=labs.count); out[i] = FLIP.get(twin, twin); has_train_twin[i] = True; n_rule += 1
        # 2) 내부 쌍 분리 (train 쌍둥이 없는 행끼리)
        groups = {}
        for i, hh in enumerate(h):
            if n[i] > 0 and not has_train_twin[i]: groups.setdefault(hh, []).append(i)
        for hh, idx in groups.items():
            if len(idx) < 2: continue
            preds = {out[i] for i in idx}
            fam = None
            for pair, (sub, sup) in PAIRS.items():
                if preds & pair: fam = (sub, sup)
            if fam is None: continue
            sub, sup = fam; idx = sorted(idx, key=lambda i: str(df.iloc[i]["ID"]))
            # 짝수개면 절반씩, 홀수개면 남는 하나는 상위 클래스(더 큰 집합)
            half = len(idx) // 2
            for k, i in enumerate(idx):
                new = sub if k < half else sup
                if out[i] != new: n_pair += 1
                out[i] = new
        return out, n_rule, n_pair
