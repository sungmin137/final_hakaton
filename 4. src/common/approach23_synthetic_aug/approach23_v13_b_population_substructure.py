"""approach23 v13: B_midshift(31-100, n=969) 내부를 4유전자 presence/absence가 아니라
전체 mutation landscape(상위 빈도 유전자 기반, SVD+KMeans)로 재분해. 규칙 제안 없음, 구조 발견만.
학습/CV/제출 금지 — unsupervised 구조 탐색은 지도학습이 아니므로 허용 범위 내.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from main import ROOT, DATA, load_train
from features.features import gene_columns, TARGET, ID

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")
genes = gene_columns(train)
is_mut_tr = (train[genes] != "WT")
is_mut_te = (test[genes] != "WT")
burden_te = is_mut_te.sum(axis=1).to_numpy()
n_te = len(test)

Bmask = (burden_te >= 31) & (burden_te <= 100)
print(f"B_midshift n={Bmask.sum()}")

# ---- 43개 제출 파일로 agreement/entropy 재사용 ----
sub_files = sorted(set(glob.glob(str(ROOT / "5. submissions" / "*.csv"))) |
                    set(glob.glob(str(ROOT / "6. experiments" / "submissions" / "**" / "*.csv"), recursive=True)))
sub_files = [f for f in sub_files if "test_proba" not in f]
pred_matrix = {}
ref_id = None
for f in sub_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if ID not in df.columns or TARGET not in df.columns:
        continue
    if ref_id is None:
        ref_id = df[ID].to_numpy()
    if len(df) != n_te or not np.array_equal(df[ID].to_numpy(), ref_id):
        continue
    pred_matrix[Path(f).name] = df[TARGET].to_numpy()
pm = pd.DataFrame(pred_matrix)
majority = pm.mode(axis=1)[0]
majority_ratio = pm.eq(majority, axis=0).mean(axis=1)
pred_entropy = pm.apply(lambda r: sp_entropy(r.value_counts(normalize=True).values, base=2), axis=1)
current_pred = pm["approach14_v3_20260913_2132.csv"] if "approach14_v3_20260913_2132.csv" in pm.columns else majority

# ---- train 최근접 유사도 (기존 방식 재사용) ----
top300_sim = (is_mut_tr.mean() + is_mut_te.mean()).sort_values(ascending=False).head(300).index.tolist()
Atr = is_mut_tr[top300_sim].to_numpy(dtype=np.float32)
Ate = is_mut_te[top300_sim].to_numpy(dtype=np.float32)
inter = Ate @ Atr.T
sum_tr = Atr.sum(axis=1); sum_te = Ate.sum(axis=1)
union = sum_te[:, None] + sum_tr[None, :] - inter
jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
best_sim = jacc.max(axis=1)

# ---- 4유전자 조합(이전 분석) 재계산: 비교용 ----
key_genes = ["TP53", "APC", "AHNAK", "TCHH"]
combo_code = np.zeros(n_te, dtype=int)
for i, g in enumerate(key_genes):
    combo_code += (is_mut_te[g].to_numpy().astype(int) << i)
combo_lab = {i: "+".join(g for j, g in enumerate(key_genes) if i & (1 << j)) or "none" for i in range(16)}
combo_str = pd.Series(combo_code).map(combo_lab)

print("\n" + "=" * 78)
print("B 내부 전체 mutation landscape로 재분해 (상위 빈도 유전자 300개, TruncatedSVD -> KMeans)")
Xb = is_mut_te.loc[Bmask, top300_sim].to_numpy(dtype=np.float32)
svd = TruncatedSVD(n_components=15, random_state=42)
Zb = svd.fit_transform(Xb)
print(f"SVD 15개 성분 설명 분산 비율 합: {svd.explained_variance_ratio_.sum():.3f}")

best_k, best_score, best_labels = None, -1, None
for k in range(3, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Zb)
    sc = silhouette_score(Zb, km.labels_)
    print(f"k={k}: silhouette={sc:.4f}, cluster 크기={np.bincount(km.labels_).tolist()}")
    if sc > best_score:
        best_score, best_k, best_labels = sc, k, km.labels_

print(f"\n선택된 k={best_k} (silhouette={best_score:.4f} 최댓값 기준)")

idxB = np.flatnonzero(Bmask)
cl = pd.Series(best_labels, index=idxB)

print("\n" + "=" * 78)
print("클러스터별 요약")
gene_freq_all_B = is_mut_te.loc[Bmask, genes].mean()
rows = []
for c in range(best_k):
    ids = cl[cl == c].index.to_numpy()
    n = len(ids)
    freq_c = is_mut_te.loc[ids, genes].mean()
    diff = (freq_c - gene_freq_all_B).sort_values(key=np.abs, ascending=False)
    top_genes_c = ", ".join(f"{g}({diff[g]:+.2f})" for g in diff.head(6).index)
    combo_dist = combo_str.iloc[ids].value_counts(normalize=True).head(3)
    combo_desc = ", ".join(f"{k_}={v:.0%}" for k_, v in combo_dist.items())
    pred_top = current_pred.iloc[ids].value_counts(normalize=True).head(3)
    pred_desc = ", ".join(f"{k_}={v:.0%}" for k_, v in pred_top.items())
    rows.append(dict(cluster=c, n=n, burden_med=np.median(burden_te[ids]),
                      agreement=majority_ratio.iloc[ids].mean(), entropy=pred_entropy.iloc[ids].mean(),
                      train_sim=np.median(best_sim[ids]), top_pred=pred_desc, dominant_combo=combo_desc,
                      distinguishing_genes=top_genes_c))
summ = pd.DataFrame(rows)
for _, r in summ.iterrows():
    print(f"\n--- Cluster {r.cluster} (n={r.n}, {r.n/Bmask.sum():.1%} of B) ---")
    print(f"burden 중앙값={r.burden_med:.0f}  agreement={r.agreement:.3f}  entropy={r.entropy:.3f}  train유사도={r.train_sim:.3f}")
    print(f"예측 분포: {r.top_pred}")
    print(f"4유전자 조합 분포: {r.dominant_combo}")
    print(f"이 cluster를 구분짓는 유전자(빈도차 상위6, B 전체 평균 대비): {r.distinguishing_genes}")

print("\n" + "=" * 78)
print("클러스터 x 기존 4유전자 조합 교차표 (구조가 겹치는지 orthogonal한지 확인)")
cross = pd.crosstab(cl.to_numpy(), combo_str.iloc[idxB].to_numpy())
print(cross.to_string())

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
summ.to_csv(out_dir / "v13_b_population_clusters.csv", index=False)
pd.DataFrame({"idx": idxB, "cluster": best_labels}).to_csv(out_dir / "v13_b_population_rowlevel.csv", index=False)
print(f"\nsaved: {out_dir}")
