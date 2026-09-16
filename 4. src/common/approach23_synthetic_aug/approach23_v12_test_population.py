"""approach23 v12: test population 구조 분석. 규칙 제안 없음, 순수 구조 발견.
학습/CV/제출 금지. train/test 원본 + 기존 제출 csv 44개(test 예측 다양성 확보용) + 기존 OOF만 사용.

주의: "28개 모델"은 group_twins OOF(train만 대상)이라 test에는 적용된 적이 없다(test는 최종 추론에서만
읽는다는 프로젝트 규칙 때문). 따라서 test 쪽 다중모델 합의는 5. submissions/ + 6. experiments/submissions/
의 실제 제출 파일(서로 다른 feature/모델/후처리 조합, 총 44개)을 재사용한다. 다만 이 중 다수가 twin_rule
적용/미적용 쌍이거나 같은 블렌드의 변형이라 완전히 독립적이지 않다는 점을 명시한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

from main import ROOT, DATA, load_train
from features.features import gene_columns, TARGET, ID

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")
genes = gene_columns(train)
is_mut_tr = (train[genes] != "WT")
is_mut_te = (test[genes] != "WT")
burden_tr = is_mut_tr.sum(axis=1).to_numpy()
burden_te = is_mut_te.sum(axis=1).to_numpy()
n_te = len(test)

print("=" * 78)
print("2. 사용 데이터 확인")
print(f"train: {train.shape}, test: {test.shape}, gene 컬럼: {len(genes)}개")

# ---- 여러 제출 파일을 test-side '모델' 앙상블로 재사용 ----
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
print(f"test 예측 파일(서로 다른 접근/후처리 조합) {len(pred_matrix)}개 확보 "
      f"(다수가 twin_rule 쌍/블렌드 변형이라 완전 독립은 아님)")

pm = pd.DataFrame(pred_matrix)  # n_te x n_models
majority = pm.mode(axis=1)[0]
majority_ratio = (pm.eq(majority, axis=0)).mean(axis=1)
n_unique = pm.nunique(axis=1)


def row_entropy(row):
    vc = row.value_counts(normalize=True)
    return sp_entropy(vc.values, base=2)


pred_entropy = pm.apply(row_entropy, axis=1)

print(f"\nagreement(다수 예측 비율) 분포: mean={majority_ratio.mean():.3f} median={majority_ratio.median():.3f} "
      f"p10={majority_ratio.quantile(.1):.3f} p90={majority_ratio.quantile(.9):.3f}")
print(f"prediction entropy 분포: mean={pred_entropy.mean():.3f} median={pred_entropy.median():.3f}")

print("\n" + "=" * 78)
print("3~4. burden 구조 기반 대분류 population (이미 알려진 경계 재사용, 재확인 아님 -> 큰 그룹화 목적)")
BANDS = [("A_low(0-30)", 0, 30), ("B_midshift(31-100)", 31, 100), ("C_normal(101-396)", 101, 396), ("D_hypermut(397+)", 397, 10**9)]
pop_of = pd.Series(index=test.index, dtype=object)
for name, lo, hi in BANDS:
    pop_of[(burden_te >= lo) & (burden_te <= hi)] = name

print(pop_of.value_counts().to_string())

print("\n" + "=" * 78)
print("4. B(31-100) 내부를 TP53/APC/AHNAK/TCHH 4-유전자 존재조합으로 세분화 (이미 아는 개별 유전자가 아니라 population 결합 구조 확인)")
sub_mask = pop_of == "B_midshift(31-100)"
key_genes = ["TP53", "APC", "AHNAK", "TCHH"]
gene_tag = {"TP53": "TP53", "APC": "APC", "AHNAK": "AHNAK", "TCHH": "TCHH"}
combo_code = np.zeros(n_te, dtype=int)
for i, g in enumerate(key_genes):
    combo_code += (is_mut_te[g].to_numpy().astype(int) << i)
combo_lab = {i: "+".join(gene_tag[g] for j, g in enumerate(key_genes) if i & (1 << j)) or "none" for i in range(16)}
sub_combo = pd.Series(combo_code[sub_mask]).map(combo_lab)
combo_counts = sub_combo.value_counts()
print(f"B(31-100, n={sub_mask.sum()}) 내부 4-유전자 존재조합 분포 (T=TP53,A=APC,H=AHNAK,C=TCHH):")
print(combo_counts.to_string())

print("\n" + "=" * 78)
print("5. Population(대역) x 현재 예측(제출 파일 다수결) x agreement/entropy")
# 대표 최신 예측: 17차 규모의 블렌드(있으면), 없으면 최다수결
if "approach14_v3_20260913_2132.csv" in pm.columns:
    current_pred = pm["approach14_v3_20260913_2132.csv"]
else:
    current_pred = majority

rows = []
for name, lo, hi in BANDS:
    mask = (pop_of == name).to_numpy()
    n = mask.sum()
    top3 = current_pred[mask].value_counts(normalize=True).head(3)
    rows.append(dict(population=name, n=n, pct=n / n_te,
                      top1=f"{top3.index[0]} {top3.iloc[0]:.1%}" if len(top3) else "-",
                      top2=f"{top3.index[1]} {top3.iloc[1]:.1%}" if len(top3) > 1 else "-",
                      top3=f"{top3.index[2]} {top3.iloc[2]:.1%}" if len(top3) > 2 else "-",
                      agreement_mean=majority_ratio[mask].mean(), entropy_mean=pred_entropy[mask].mean(),
                      burden_median=np.median(burden_te[mask])))
pop_df = pd.DataFrame(rows)
print(pop_df.round(3).to_string(index=False))

print("\n" + "=" * 78)
print("6. 4-유전자 조합별(B 내부) 예측/agreement/entropy")
rows2 = []
for code, name in combo_lab.items():
    mask = sub_mask.to_numpy() & (combo_code == code)
    n = mask.sum()
    if n < 10:
        continue
    top2 = current_pred[mask].value_counts(normalize=True).head(2)
    rows2.append(dict(combo=name, n=n, top1=f"{top2.index[0]} {top2.iloc[0]:.1%}",
                       top2=f"{top2.index[1]} {top2.iloc[1]:.1%}" if len(top2) > 1 else "-",
                       agreement=majority_ratio[mask].mean(), entropy=pred_entropy[mask].mean(),
                       burden_med=np.median(burden_te[mask])))
combo_df = pd.DataFrame(rows2).sort_values("n", ascending=False)
print(combo_df.round(3).to_string(index=False))

print("\n" + "=" * 78)
print("7. train 대응도 -- population별 최근접 train 유사도 (자카드, 상위 300유전자 기준, 기존 계산 방식 재사용)")
top300 = (is_mut_tr.mean() + is_mut_te.mean()).sort_values(ascending=False).head(300).index.tolist()
Atr = is_mut_tr[top300].to_numpy(dtype=np.float32)
Ate = is_mut_te[top300].to_numpy(dtype=np.float32)
inter = Ate @ Atr.T
sum_tr = Atr.sum(axis=1); sum_te = Ate.sum(axis=1)
union = sum_te[:, None] + sum_tr[None, :] - inter
jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
best_sim = jacc.max(axis=1)

for name, lo, hi in BANDS:
    mask = (pop_of == name).to_numpy()
    b = best_sim[mask]
    print(f"{name}: median_sim={np.median(b):.3f} p10={np.percentile(b,10):.3f} p90={np.percentile(b,90):.3f}")

print("\ncombo(B 내부)별 train 유사도:")
for code, name in combo_lab.items():
    mask = sub_mask.to_numpy() & (combo_code == code)
    if mask.sum() < 10:
        continue
    b = best_sim[mask]
    print(f"  {name}(n={mask.sum()}): median_sim={np.median(b):.3f} p10={np.percentile(b,10):.3f}")

print("\n" + "=" * 78)
print("9. Agreement 구간별 population 특성")
agree_bins = pd.cut(majority_ratio, [0, 0.5, 0.7, 0.9, 1.01], labels=["<50%", "50-70%", "70-90%", ">90%"])
for lab in ["<50%", "50-70%", "70-90%", ">90%"]:
    mask = (agree_bins == lab).to_numpy()
    if mask.sum() == 0:
        continue
    print(f"{lab} (n={mask.sum()}, {mask.mean():.1%}): burden median={np.median(burden_te[mask]):.0f}, "
          f"TP53+ 비율={is_mut_te.loc[mask,'TP53'].mean():.2f}, train유사도 median={np.median(best_sim[mask]):.3f}, "
          f"주요 population 분포: {pop_of[mask].value_counts(normalize=True).round(2).to_dict()}")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
pd.DataFrame({"ID": test["ID"], "population": pop_of.to_numpy(), "combo4gene": pd.Series(combo_code).map(combo_lab).to_numpy(),
              "majority_ratio": majority_ratio.to_numpy(), "n_unique_pred": n_unique.to_numpy(),
              "entropy": pred_entropy.to_numpy(), "train_nn_sim": best_sim, "burden": burden_te}
             ).to_csv(out_dir / "v12_test_population_rowlevel.csv", index=False)
print(f"\nsaved: {out_dir}")
