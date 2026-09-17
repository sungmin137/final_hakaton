"""approach23 v10: test-first distribution scan — test에 실재하는 구조를 먼저 찾고 train/OOF로 연결.
학습/CV/제출 전부 금지. 기존 v3(test 분포 스캔), v4(anomaly scan) 산출물 재사용 + 신규는 순수 통계만.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from main import ROOT, DATA, load_train
from features.features import gene_columns, TARGET

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")
genes = gene_columns(train)
is_mut_tr = (train[genes] != "WT")
is_mut_te = (test[genes] != "WT")
burden_tr = is_mut_tr.sum(axis=1).to_numpy()
burden_te = is_mut_te.sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_)
true_lab = train[TARGET].to_numpy()

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
blend_oof = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
proba_scaled = blend_oof * s3; proba_scaled /= proba_scaled.sum(1, keepdims=True)
pred_oof = np.array(classes)[proba_scaled.argmax(1)]

print("=" * 78)
print("2. 세분화된 burden 구간 train vs test")
FBINS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 30), (31, 50), (51, 75), (76, 100), (101, 200), (201, 10**9)]
rows = []
for lo, hi in FBINS:
    ntr = ((burden_tr >= lo) & (burden_tr <= hi)).sum()
    nte = ((burden_te >= lo) & (burden_te <= hi)).sum()
    hi_s = str(hi) if hi < 10**9 else "+"
    rows.append((f"{lo}-{hi_s}", ntr, ntr / len(train), nte, nte / len(test), (nte / len(test)) / max(ntr / len(train), 1e-9)))
fine_df = pd.DataFrame(rows, columns=["bin", "train_n", "train_pct", "test_n", "test_pct", "ratio_test_over_train"])
print(fine_df.round(4).to_string(index=False))

print("\n" + "=" * 78)
print("3. 유전자별 train/test frequency, 4그룹 분류")
freq_tr = is_mut_tr.mean(); freq_te = is_mut_te.mean()
cnt_tr = is_mut_tr.sum(); cnt_te = is_mut_te.sum()
gene_df = pd.DataFrame({"freq_tr": freq_tr, "freq_te": freq_te, "cnt_tr": cnt_tr, "cnt_te": cnt_te})
gene_df["abs_diff"] = gene_df.freq_te - gene_df.freq_tr
gene_df["rel_change"] = gene_df.abs_diff / gene_df.freq_tr.replace(0, np.nan)

TR_COMMON, TE_COMMON = 0.03, 0.03  # 3% 이상 "흔함" 기준(설명 가능한 임계, 팀 관례상 5%가 hotspot 기준이었으므로 3%는 보수적으로 낮춤)
g1 = gene_df[(gene_df.freq_tr >= TR_COMMON) & (gene_df.freq_te >= TE_COMMON)]
g2 = gene_df[(gene_df.freq_tr >= TR_COMMON) & (gene_df.freq_te < TR_COMMON) & (gene_df.abs_diff < -0.01)]
g3 = gene_df[(gene_df.freq_tr < TR_COMMON) & (gene_df.freq_te >= TE_COMMON)]
g4 = gene_df[(gene_df.freq_tr < TR_COMMON) & (gene_df.freq_te < TE_COMMON) & (gene_df.abs_diff.abs() > 0.02)]
print(f"1) train도 흔하고 test도 흔함: {len(g1)}개")
print(f"2) train엔 흔한데 test서 감소: {len(g2)}개 — " + ", ".join(g2.sort_values("abs_diff").head(8).index))
print(f"3) train엔 드문데 test서 급증 (가장 중요): {len(g3)}개")
print("   상위 15개(abs_diff 기준):")
print(g3.sort_values("abs_diff", ascending=False).head(15)[["freq_tr", "freq_te", "abs_diff", "cnt_tr", "cnt_te"]].round(4).to_string())
print(f"4) 둘 다 드물지만 상대적으로 큰 변화: {len(g4)}개")

print("\n" + "=" * 78)
print("4. co-mutation shift — 이미 v3 스캔에서 계산된 상위 150유전자 co-occurrence diff 재사용")
print("(재계산하지 않음: AHNAK&TCHH, AHNAK&APC, APC&TCHH, TP53&APC, TP53&AHNAK, TP53&TCHH, ")
print(" AHNAK&CACNA1A, TCHH&CACNA1A, APC&CACNA1A, AHNAK&KMT2D, AHNAK&MAGEE1, TCHH&KMT2D, ")
print(" TCHH&MAGEE1, TP53&CACNA1A, AHNAK&CPEB2 — 전부 diff +0.10~+0.19, 이미 발견됨)")
print("\n4-검증: 이 co-mutation 증가가 'burden 자체가 늘어서 생기는 자연스러운 부산물'인지,")
print("       burden을 고정해도 남는 test 고유 신호인지 확인 (31-100 구간만 고정)")
m31_tr = (burden_tr >= 31) & (burden_tr <= 100)
m31_te = (burden_te >= 31) & (burden_te <= 100)
pairs = [("AHNAK", "TCHH"), ("TP53", "APC"), ("AHNAK", "CACNA1A"), ("TCHH", "MAGEE1")]
for g1n, g2n in pairs:
    co_tr = (is_mut_tr.loc[m31_tr, g1n] & is_mut_tr.loc[m31_tr, g2n]).mean()
    co_te = (is_mut_te.loc[m31_te, g1n] & is_mut_te.loc[m31_te, g2n]).mean()
    print(f"  {g1n}&{g2n} (31-100 구간 내 고정): train={co_tr:.4f} test={co_te:.4f} diff={co_te-co_tr:+.4f}")

print("\n" + "=" * 78)
print("4-1. burden+gene 조합 — test 우세 조합 스캔 (상위 shift 유전자 x burden 구간)")
top_shift_genes = g3.sort_values("abs_diff", ascending=False).head(8).index.tolist() + ["TP53"]
for g in dict.fromkeys(top_shift_genes):
    for lo, hi, name in [(31, 100, "31-100"), (101, 10**9, "101+")]:
        mtr = (burden_tr >= lo) & (burden_tr <= hi)
        mte = (burden_te >= lo) & (burden_te <= hi)
        f_tr = is_mut_tr.loc[mtr, g].mean(); f_te = is_mut_te.loc[mte, g].mean()
        if f_te - f_tr > 0.05:
            print(f"  {g} in {name}: train {f_tr:.3f} -> test {f_te:.3f} (diff {f_te-f_tr:+.3f}, test n={mte.sum()})")

print("\n" + "=" * 78)
print("5~7. test 패턴 -> train counterpart 연결: 'burden 31-100 + AHNAK+TCHH+APC+TP53 4개 동시 변이' 케이스")
combo_genes = ["AHNAK", "TCHH", "APC", "TP53"]
combo_tr = np.all([is_mut_tr[g].to_numpy() for g in combo_genes], axis=0) & m31_tr.to_numpy() if hasattr(m31_tr, "to_numpy") else np.all([is_mut_tr[g].to_numpy() for g in combo_genes], axis=0) & m31_tr
combo_te = np.all([is_mut_te[g].to_numpy() for g in combo_genes], axis=0) & m31_te.to_numpy() if hasattr(m31_te, "to_numpy") else np.all([is_mut_te[g].to_numpy() for g in combo_genes], axis=0) & m31_te
print(f"train에 이 4-유전자 동시변이+31-100 조합: {combo_tr.sum()}행 / test: {combo_te.sum()}행")
if combo_tr.sum() >= 10:
    print("train counterpart 실제 class 분포:")
    print(pd.Series(true_lab[combo_tr]).value_counts().to_string())
    f1c = f1_score(true_lab[combo_tr], pred_oof[combo_tr], average="macro", labels=sorted(set(true_lab[combo_tr])))
    acc_c = (true_lab[combo_tr] == pred_oof[combo_tr]).mean()
    print(f"이 subset의 OOF macro F1={f1c:.3f}, accuracy={acc_c:.3f} (subset n={combo_tr.sum()})")
    print("test에서 이 조합 행의 현재 예측 분포:")
    p7 = np.load(ROOT / "6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy") / s3
    p7 = p7 / p7.sum(1, keepdims=True)
    p2 = np.load(ROOT / "6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy")
    z = np.log(p7 + 1e-9) * 0.5 + np.log(p2 + 1e-9) * 0.5
    blend_te = np.exp(z - z.max(1, keepdims=True)); blend_te /= blend_te.sum(1, keepdims=True)
    pred_te17 = np.array(classes)[(blend_te * s3).argmax(1)]
    print(pd.Series(pred_te17[combo_te]).value_counts().to_string())
else:
    print("train counterpart 표본이 너무 적어 신뢰 불가")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
fine_df.to_csv(out_dir / "v10_fine_burden_bins.csv", index=False)
gene_df.to_csv(out_dir / "v10_gene_4group.csv")
print(f"\nsaved: {out_dir}")
