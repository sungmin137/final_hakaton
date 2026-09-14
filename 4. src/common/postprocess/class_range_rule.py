"""클래스별 변이 수 범위 규칙 (2026-09-14, train 근거만).
train에서 클래스 c의 변이 수 관측 범위 [min_c, max_c]를 구하고, 예측 클래스의 범위 밖에 있는 행은
범위 안에 드는 클래스 중 배율 적용 확률이 가장 높은 클래스로 바꾼다. 매개변수(범위)는 train으로만 결정하고 test는 행 단위 추론에만 쓴다.
정직 OOF(fold 학습부로 범위 fit): 변경 6행, 정답 0→2, macro F1 +0.0005. 초과변이 규칙(21차)을 test를 보지 않고 세울 수 있는 일반형.
MIN_ROWS: 범위 끝값이 단 1행에 의존하지 않도록, 상한은 상위 MIN_ROWS번째 값으로 잡을 수 있다(기본 1 = 관측 최대)."""
import numpy as np

class ClassRangeRule:
    def __init__(self, k_max=1): self.k_max = k_max   # 상한 = 상위 k번째 관측값
    def fit(self, train):
        self.genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
        n = (train[self.genes] != "WT").sum(axis=1).to_numpy(); y = train["SUBCLASS"].to_numpy()
        self.lo, self.hi = {}, {}
        for c in np.unique(y):
            v = np.sort(n[y == c]); self.lo[c] = int(v[0]); self.hi[c] = int(v[-min(self.k_max, len(v))])
        return self
    def apply(self, df, pred, proba_scaled, classes):
        n = (df[self.genes] != "WT").sum(axis=1).to_numpy(); out = np.asarray(pred, dtype=object).copy(); changed = []
        for i in range(len(df)):
            c = out[i]
            if self.lo[c] <= n[i] <= self.hi[c]: continue
            ok = [k for k, cc in enumerate(classes) if self.lo[cc] <= n[i] <= self.hi[cc]]
            if not ok: continue
            out[i] = classes[ok[int(np.argmax(proba_scaled[i][ok]))]]; changed.append(i)
        return out, changed
