"""approach23 v1: synthetic GBMLGG/LGG 생성 방식 자체의 sanity check (설명용, 학습/검증에 미사용).

전체 train의 GBMLGG/LGG 실제 행을 도너 풀로 synthetic 50개씩을 만들어
duplicate / mutation burden / gene frequency / class identity / sparsity를 점검한다.
여기서 만든 synthetic 세트는 approach23_v1_run_experiment.py의 fold별 synthetic과는
별개 객체이며(재현성 확인용, seed=42 고정), 어떤 학습·검증에도 사용하지 않는다.

설명 문서: 2. team/approaches/approach23.md
실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach23_synthetic_aug/approach23_v1_sanity_check.py"
"""
import numpy as np
import pandas as pd

from main import ROOT, load_train
from features.features import TARGET, gene_columns
from postprocess.twin_rule import _hash_rows
from approach23_synthetic_aug.synthetic_gen import generate_synthetic

N = 50
SEED = 42
CLASSES = ("GBMLGG", "LGG")

train = load_train()
genes = gene_columns(train)


def burden(df: pd.DataFrame) -> pd.Series:
    return (df[genes] != "WT").sum(axis=1)


def mean_vec(df: pd.DataFrame) -> np.ndarray:
    return (df[genes] != "WT").astype(int).mean().to_numpy()


def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


real = {c: train[train[TARGET] == c] for c in CLASSES}
synth = {c: generate_synthetic(real[c], c, N, SEED) for c in CLASSES}

print("=== A. duplicate 확인 ===")
all_real_hash = set(_hash_rows(train, genes))
for c, s in synth.items():
    syn_hash = _hash_rows(s, genes)
    dup_vs_real = sum(h in all_real_hash for h in syn_hash)
    dup_internal = N - len(set(syn_hash))
    print(f"{c}: 실제 train과 완전동일 {dup_vs_real}/{N}행, synthetic 내부 중복 {dup_internal}/{N}행")

print("\n=== B. mutation burden (mutated gene 개수) ===")
for c in CLASSES:
    rb, sb = burden(real[c]), burden(synth[c])
    for name, b in (("real     ", rb), ("synthetic", sb)):
        d = b.describe(percentiles=[.1, .25, .5, .75, .9])
        print(f"{c} {name}: mean={d['mean']:.2f} median={d['50%']:.1f} std={d['std']:.2f} "
              f"min={d['min']:.0f} max={d['max']:.0f} p10={d['10%']:.1f} p90={d['90%']:.1f}")

print("\n=== C. gene mutation frequency 차이 (real vs synthetic, 상위 15개) ===")
for c in CLASSES:
    rf = (real[c][genes] != "WT").mean()
    sf = (synth[c][genes] != "WT").mean()
    diff = (sf - rf).abs().sort_values(ascending=False)
    print(f"-- {c} (전체 {len(genes)}유전자 중 diff 상위 15) --")
    for g in diff.head(15).index:
        print(f"  {g}: real={rf[g]:.3f} synthetic={sf[g]:.3f} diff={diff[g]:.3f}")

print("\n=== D. class identity check (코사인 유사도, 이진 변이벡터 centroid) ===")
centroid = {c: mean_vec(real[c]) for c in CLASSES}
for c in CLASSES:
    other = "LGG" if c == "GBMLGG" else "GBMLGG"
    sv = mean_vec(synth[c])
    sim_own, sim_other = cos_sim(sv, centroid[c]), cos_sim(sv, centroid[other])
    ok = "정상" if sim_own > sim_other else "⚠ 이상 — 반대 클래스에 더 가까움"
    print(f"synthetic {c}: sim(자기 클래스 centroid)={sim_own:.4f}  sim({other} centroid)={sim_other:.4f}  → {ok}")

print("\n=== E. sparsity (전체 셀 중 변이 있는 비율) ===")
for c in CLASSES:
    r_density = (real[c][genes] != "WT").to_numpy().mean()
    s_density = (synth[c][genes] != "WT").to_numpy().mean()
    print(f"{c}: real density={r_density:.5f}  synthetic density={s_density:.5f}  (배율 {s_density / r_density:.2f}x)")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_synthetic_gbmlgg_lgg"
out_dir.mkdir(parents=True, exist_ok=True)
for c, s in synth.items():
    s.to_csv(out_dir / f"synthetic_{c}_sanitycheck_n{N}.csv", index=False)
print(f"\nsaved synthetic csv (진단용, 학습에 미사용) → {out_dir}")
