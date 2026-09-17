"""approach23 v14: B(31-100) 내부 CACNA1A/CPEB2 축이 실제 암종 정보를 갖는지 검증.
모델 학습/CV/제출/rule 생성 없음 — train SUBCLASS 분포 + test 비교 + 기존 제출 43개 재사용.
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


def four_groups(is_mut, mask=None):
    a = is_mut["CACNA1A"].to_numpy(); b = is_mut["CPEB2"].to_numpy()
    if mask is not None:
        a, b = a[mask], b[mask]
    return a, b


def describe_groups(sub_labels, a, b, label):
    print(f"\n--- {label} (n={len(a)}) ---")
    for name, m in [("CACNA1A+CPEB2+", a & b), ("CACNA1A+CPEB2-", a & ~b),
                    ("CACNA1A-CPEB2+", ~a & b), ("CACNA1A-CPEB2-", ~a & ~b)]:
        n = m.sum()
        if n == 0:
            print(f"{name}: n=0"); continue
        vc = pd.Series(sub_labels[m]).value_counts()
        p = vc / n
        ent = sp_entropy(p.values, base=2)
        top = ", ".join(f"{k}={v:.0%}" for k, v in p.head(4).items())
        print(f"{name}: n={n} ({n/len(a):.1%})  top class: {top}  entropy={ent:.2f}")


print("=" * 78)
print("1. Train 전체 vs Train 31-100에서 CACNA1A/CPEB2 4그룹 x SUBCLASS")
true_lab_tr = train[TARGET].to_numpy()
a_all, b_all = four_groups(is_mut_tr)
describe_groups(true_lab_tr, a_all, b_all, "Train 전체 (n=6201)")
m31tr = (burden_tr >= 31) & (burden_tr <= 100)
a_31, b_31 = four_groups(is_mut_tr, m31tr)
describe_groups(true_lab_tr[m31tr], a_31, b_31, "Train burden 31-100 (n=993)")

print(f"\n참고: train 전체 CACNA1A 빈도={is_mut_tr['CACNA1A'].mean():.4f}, CPEB2 빈도={is_mut_tr['CPEB2'].mean():.4f}")
print(f"      train 31-100 CACNA1A 빈도={is_mut_tr.loc[m31tr,'CACNA1A'].mean():.4f}, CPEB2 빈도={is_mut_tr.loc[m31tr,'CPEB2'].mean():.4f}")
print(f"      test 31-100 CACNA1A 빈도={is_mut_te.loc[(burden_te>=31)&(burden_te<=100),'CACNA1A'].mean():.4f}, "
      f"CPEB2 빈도={is_mut_te.loc[(burden_te>=31)&(burden_te<=100),'CPEB2'].mean():.4f}")

print("\n" + "=" * 78)
print("3. 통제: train에서 CACNA1A+CPEB2+ 그룹의 지배 class(있다면) burden 31-100 비율 및 다른 주요 변이")
m_pp_tr = m31tr & a_31_full if False else None
mask_pp_31 = m31tr.copy()
mask_pp_31_idx = np.flatnonzero(m31tr)[a_31 & b_31]
if len(mask_pp_31_idx) > 0:
    dom_classes = pd.Series(true_lab_tr[mask_pp_31_idx]).value_counts()
    print(f"train 31-100 CACNA1A+CPEB2+ (n={len(mask_pp_31_idx)}) 주요 클래스: {dom_classes.to_dict()}")
    top_c = dom_classes.idxmax()
    m_class = true_lab_tr == top_c
    print(f"  {top_c} 전체 train {m_class.sum()}행 중 burden 31-100 비율: {(m_class & m31tr).sum()/m_class.sum():.1%}")
    # CACNA1A/CPEB2 제외한 landscape로 이 그룹과 나머지(같은 31-100, 같은 top_c 클래스)가 구분되는지
    other_genes = [g for g in genes if g not in ("CACNA1A", "CPEB2")]
    grp = is_mut_tr.loc[mask_pp_31_idx, other_genes].mean()
    rest_idx = np.flatnonzero(m31tr & (true_lab_tr == top_c))
    rest_idx = np.setdiff1d(rest_idx, mask_pp_31_idx)
    if len(rest_idx) >= 5:
        rest = is_mut_tr.loc[rest_idx, other_genes].mean()
        diff = (grp - rest).sort_values(key=np.abs, ascending=False).head(8)
        print(f"  같은 {top_c}, 같은 31-100인데 CACNA1A/CPEB2 상태만 다른 대조군(n={len(rest_idx)})과의 유전자 빈도차 상위8:")
        for g in diff.index:
            print(f"    {g}: 그룹={grp[g]:.2f} 대조군={rest[g]:.2f} diff={diff[g]:+.2f}")
    else:
        print(f"  대조군 표본 부족(n={len(rest_idx)})")
else:
    print("train 31-100에 CACNA1A+CPEB2+ 표본 없음")

print("\n" + "=" * 78)
print("2. Test B(31-100)에서 동일 4그룹")
mte31 = (burden_te >= 31) & (burden_te <= 100)
a_te, b_te = four_groups(is_mut_te, mte31)

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
majority_ratio = pm.eq(pm.mode(axis=1)[0], axis=0).mean(axis=1)
pred_entropy = pm.apply(lambda r: sp_entropy(r.value_counts(normalize=True).values, base=2), axis=1)
current17 = pm["approach14_v3_20260913_2132.csv"] if "approach14_v3_20260913_2132.csv" in pm.columns else pm.mode(axis=1)[0]

idxB = np.flatnonzero(mte31)
print("\n4. Test B 내부 4그룹별 현재 prediction/agreement/entropy")
for name, m in [("CACNA1A+CPEB2+", a_te & b_te), ("CACNA1A+CPEB2-", a_te & ~b_te),
                ("CACNA1A-CPEB2+", ~a_te & b_te), ("CACNA1A-CPEB2-", ~a_te & ~b_te)]:
    ids = idxB[m]
    if len(ids) == 0:
        print(f"{name}: n=0"); continue
    p17 = current17.iloc[ids].value_counts(normalize=True).head(3)
    p17_desc = ", ".join(f"{k}={v:.0%}" for k, v in p17.items())
    print(f"{name}: n={len(ids)} ({len(ids)/mte31.sum():.1%})  17차pred: {p17_desc}  "
          f"agreement={majority_ratio.iloc[ids].mean():.3f}  entropy={pred_entropy.iloc[ids].mean():.3f}")

print(f"\nsaved: (요약 통계만 출력, 별도 파일 저장 없음)")
