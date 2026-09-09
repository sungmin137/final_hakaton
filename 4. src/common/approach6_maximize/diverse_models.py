"""접근 6 — 모델 다양성: 희소 로지스틱 회귀(변이 토큰 + 유전자) 와 자카드 최근접 이웃. 정직 CV OOF 저장.
train.csv만 사용. 실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach6_maximize/diverse_models.py"
산출: 6. experiments/approach6_maximize/{oof_lr.npy, oof_knn.npy, result.json}
"""
import json, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, TARGET, load_train, twin_groups, gene_columns
from approach1_count_weight.count_weights import gene_matrix, variant_matrix

OUT = ROOT / "6. experiments" / "approach6_maximize"


class SparseLR:
    """유전자 이진 + (유전자:변이) 토큰 이진을 IDF 가중 후 다항 로지스틱 회귀."""
    def __init__(self, C=0.5, min_df=2):
        self.C, self.min_df = C, min_df

    def fit(self, df, y):
        self.genes = gene_columns(df)
        Xg = gene_matrix(df, self.genes); Xv, self.vocab = variant_matrix(df, self.genes, None)
        keep = np.asarray(Xv.sum(0)).ravel() >= self.min_df; self.keep = np.flatnonzero(keep)
        X = sparse.hstack([Xg, Xv[:, self.keep]]).tocsr()
        n = X.shape[0]; dfreq = np.asarray(X.sum(0)).ravel(); self.idf = np.log((n + 1) / (dfreq + 1)) + 1
        X = X.multiply(self.idf).tocsr(); X = sparse.diags(1 / np.sqrt(np.maximum(np.asarray(X.multiply(X).sum(1)).ravel(), 1e-9))) @ X
        self.model = LogisticRegression(C=self.C, max_iter=3000, solver="saga", n_jobs=8).fit(X, y); self.classes_ = self.model.classes_
        return self

    def predict_proba(self, df):
        Xg = gene_matrix(df, self.genes); Xv, _ = variant_matrix(df, self.genes, self.vocab)
        X = sparse.hstack([Xg, Xv[:, self.keep]]).tocsr().multiply(self.idf).tocsr()
        X = sparse.diags(1 / np.sqrt(np.maximum(np.asarray(X.multiply(X).sum(1)).ravel(), 1e-9))) @ X
        return self.model.predict_proba(X)


class JaccardKNN:
    """유전자 집합 + 변이 토큰 집합의 자카드 유사도 상위 k 이웃 가중 투표. 완전 동일 행(자카드 1)은 강한 표. 학습 fold 정보만 사용."""
    def __init__(self, k=15, power=3.0):
        self.k, self.power = k, power

    def fit(self, df, y):
        self.genes = gene_columns(df); Xg = gene_matrix(df, self.genes); Xv, self.vocab = variant_matrix(df, self.genes, None)
        self.X = sparse.hstack([Xg, Xv]).tocsr().astype(np.float32); self.sz = np.asarray(self.X.sum(1)).ravel(); self.y = np.asarray(y); self.K = int(self.y.max()) + 1
        return self

    def predict_proba(self, df):
        Xg = gene_matrix(df, self.genes); Xv, _ = variant_matrix(df, self.genes, self.vocab)
        Q = sparse.hstack([Xg, Xv]).tocsr().astype(np.float32); qs = np.asarray(Q.sum(1)).ravel()
        inter = (Q @ self.X.T).toarray(); union = qs[:, None] + self.sz[None, :] - inter; J = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        P = np.full((Q.shape[0], self.K), 1e-3)
        idx = np.argsort(-J, axis=1)[:, : self.k]
        for i in range(Q.shape[0]):
            for j in idx[i]:
                P[i, self.y[j]] += J[i, j] ** self.power
        return P / P.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tr = load_train(); le = LabelEncoder(); y = le.fit_transform(tr[TARGET]); K = len(le.classes_)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, y, twin_groups(tr)))
    res = {}
    for name, mk in [("knn", lambda: JaccardKNN()), ("lr", lambda: SparseLR())]:
        P = np.zeros((len(tr), K))
        for k, (tri, vai) in enumerate(folds):
            m = mk().fit(tr.iloc[tri], y[tri]); p = m.predict_proba(tr.iloc[vai])
            if name == "lr": P[vai][:, :] = 0; P[np.ix_(vai, m.classes_)] = p
            else: P[vai] = p
            print(f"[{name}] fold{k} f1={f1_score(y[vai], P[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
        np.save(OUT / f"oof_{name}.npy", P); res[name] = round(f1_score(y, P.argmax(1), average="macro"), 4); print(f"== {name} OOF macroF1={res[name]}", flush=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res))


if __name__ == "__main__":
    main()
