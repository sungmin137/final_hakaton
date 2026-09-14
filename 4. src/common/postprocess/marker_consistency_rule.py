"""마커 일관성 규칙 (2026-09-14). 초과변이 규칙(21차 +0.0125)의 일반화: "train에 없는 (마커, 클래스) 조합"을 예측하지 않는다.
- 마커 = 유전자:아미노산 위치 토큰(예: BRAF:600). train 보유자 ≥ MIN_CARRIERS인 마커만.
- 허용 클래스 = 그 마커 보유자 중 train에 ≥ MIN_ROWS명인 클래스. (예: BRAF:600 → THCA·SKCM·COAD·LUAD·GBMLGG; STES·LUSC는 0명 → 비허용)
- test 행이 그런 마커를 갖고 예측이 비허용이면, 그 행의 모든 마커 허용 집합의 교집합(비면 합집합) 안에서 배율 적용 확률 argmax로 교체.
- 마커·집합은 train으로만 결정. 쌍둥이 중복은 하나로 센다."""
import re
import numpy as np, pandas as pd
from collections import defaultdict, Counter
MIN_CARRIERS, MIN_ROWS = 30, 2
_POS = re.compile(r"^[A-Z]?(\d+)")

def _tokens(cell, gene):
    out = set()
    for t in str(cell).split(" "):
        m = _POS.match(t)
        if m and t != "WT": out.add(f"{gene}:{m.group(1)}")
    return out

class MarkerConsistency:
    def fit(self, train):
        genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
        h = pd.util.hash_pandas_object(train[genes], index=False).to_numpy(); keep = ~pd.Series(h).duplicated().to_numpy()
        d = train[keep]; G = d[genes].to_numpy(); y = d["SUBCLASS"].to_numpy(); cnt = defaultdict(Counter)
        for i in range(len(d)):
            for j in np.nonzero(G[i] != "WT")[0]:
                for t in _tokens(G[i, j], genes[j]): cnt[t][y[i]] += 1
        self.allowed = {t: {c for c, n in cc.items() if n >= MIN_ROWS} for t, cc in cnt.items() if sum(cc.values()) >= MIN_CARRIERS}
        self.genes = genes; return self
    def apply(self, df, pred, proba_scaled, classes):
        G = df[self.genes].to_numpy(); out = np.asarray(pred, dtype=object).copy(); changed = []
        for i in range(len(df)):
            sets = []
            for j in np.nonzero(G[i] != "WT")[0]:
                for t in _tokens(G[i, j], self.genes[j]):
                    if t in self.allowed: sets.append(self.allowed[t])
            if not sets or any(out[i] in s for s in sets) and all(out[i] in s for s in sets): continue
            if any(out[i] in s for s in sets): continue          # 마커 하나라도 허용하면 유지 (보수적)
            inter = set.intersection(*sets) if len(sets) > 1 else sets[0]; cand = inter if inter else set.union(*sets)
            ai = [classes.index(c) for c in cand]; out[i] = classes[ai[int(np.argmax(proba_scaled[i][ai]))]]; changed.append(i)
        return out, changed
