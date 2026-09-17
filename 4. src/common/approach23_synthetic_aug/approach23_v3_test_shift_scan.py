"""approach23 v3: test mutation distribution 진단 스캔 (synthetic 후보 발굴 전 단계).

## 이 스크립트가 test.csv를 읽는 것에 대한 근거 (컴플라이언스 경계)
프로젝트 최우선 규칙은 "test.csv는 최종 제출 추론 단계에서만 읽는다"이지만, 팀은 이미 이 경계를
한 번 실무적으로 정리한 적이 있다:
  - `3. docs/05_test_inference.md`: "규칙: test 분포를 보고 모델을 조정하지는 않는다 ... 다만
    '이 구간을 파야 한다'는 우선순위 판단은 test 예측 분포 관찰에서 왔음을 기록해 둔다."
  - 성민님의 21차(초과변이 규칙, `postprocess/hypermut_rule.py`)가 실제로 이 경계 위에서 동작한다:
    test의 burden 분포를 보고 "여기를 파야 한다"는 우선순위를 얻었지만, 규칙의 임계값(396)과
    허용 클래스 집합은 **train 통계로만** 정했다.

이 스크립트도 같은 경계를 따른다: test는 **자기 자신의 원본 피처(변이 여부·개수)를 관찰하는 용도로만**
읽는다. train FeatureMaker fit, 클래스 배율, 어떤 모델 파라미터도 test로 결정하지 않는다. 유일한
예외는 "현재 모델이 test를 어느 클래스로 예측하는가"인데, 이것도 **가설(hypothesis)로만** 쓰고
정답으로 취급하지 않는다(사용자 지시). 이 스크립트의 결과로 만들어지는 것은 다음 단계 실험의
**우선순위 후보 목록**뿐이고, synthetic 생성 자체는 하지 않는다 — 실제로 만들 경우 도너는 반드시
train 행만 사용한다(별도 스크립트, 이 스크립트가 정한 후보 영역만 참고).

## 분석 항목
1. mutation burden 분포 (구간별 train vs test 비율 + 그 구간의 train 라벨 분포 + test 예측 라벨 분포)
2. 유전자별 mutation frequency 차이 (train에 전혀 없는 유전자 별도 표시 — synthetic 불가 영역)
3. 희귀 토큰(특정 유전자의 특정 변이 문자열)이 train에 없이 test에만 등장하는 경우
4. 상위 빈도 유전자 간 co-mutation 패턴 차이 (train vs test)
5. test 행 → train 행 최근접 이웃(자카드 유사도, 상위 빈도 유전자 기준) — 어떤 클래스에 가장 가까운지
6. 위 패턴들이 현재 모델의 예측 클래스에 집중되는지

실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach23_synthetic_aug/approach23_v3_test_shift_scan.py"
설명 문서: 2. team/approaches/approach23.md
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from main import ROOT, DATA, PARAM_SETS, TARGET, load_train, FeatureMaker
from features.features import gene_columns

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")   # 이 스크립트 전용 진단 로더. main.load_test()와
                                                        # 별개로 둬서 "추론 단계 전용" 규칙을 건드리지 않음.
genes = gene_columns(train)
is_mut_tr = (train[genes] != "WT")
is_mut_te = (test[genes] != "WT")
burden_tr = is_mut_tr.sum(axis=1).to_numpy()
burden_te = is_mut_te.sum(axis=1).to_numpy()

# ---- 진단용 단일 모델 (CV 아님, 1회 학습) : test 예측을 "가설"로만 사용 ----
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
classes = list(le.classes_)
fm = FeatureMaker("v4").fit(train)
diag_model = xgb.XGBClassifier(**PARAM_SETS["mild_col"]).fit(fm.transform(train), y)
pred_te = np.array(classes)[diag_model.predict(fm.transform(test))]
print("진단 모델 학습 완료 (v4+mild_col, train 전체, 1회 fit)\n", flush=True)

# ================================================================ 1. burden 구간별 분포
print("=" * 70)
print("1. mutation burden 구간별 train vs test 비율, 그 구간의 라벨(가설)")
BINS = [(0, 10), (11, 30), (31, 100), (101, 300), (301, 500), (501, 900), (901, 10_000)]
for lo, hi in BINS:
    m_tr = (burden_tr >= lo) & (burden_tr <= hi)
    m_te = (burden_te >= lo) & (burden_te <= hi)
    n_tr, n_te = m_tr.sum(), m_te.sum()
    p_tr, p_te = n_tr / len(train), n_te / len(test)
    print(f"\n[{lo}-{hi}] train {n_tr}행({p_tr:.1%})  test {n_te}행({p_te:.1%})  비율(test/train)={p_te / (p_tr + 1e-9):.2f}x")
    if n_tr > 0:
        top_tr = train.loc[m_tr, TARGET].value_counts().head(4)
        print(f"   train 실제 라벨 분포: " + ", ".join(f"{k}={v}" for k, v in top_tr.items()))
    else:
        print("   train 실제 라벨 분포: (train에 이 구간 행 없음)")
    if n_te > 0:
        top_te = pd.Series(pred_te[m_te]).value_counts().head(4)
        print(f"   test 모델예측(가설) 분포: " + ", ".join(f"{k}={v}" for k, v in top_te.items()))

# ================================================================ 2. 유전자별 frequency 차이
print("\n" + "=" * 70)
print("2. 유전자 mutation frequency 차이 (test - train, 상위 20개)")
freq_tr = is_mut_tr.mean()
freq_te = is_mut_te.mean()
diff = (freq_te - freq_tr).sort_values(key=np.abs, ascending=False)
for g in diff.head(20).index:
    flag = " [train 0]" if freq_tr[g] == 0 else ""
    print(f"  {g}: train={freq_tr[g]:.4f} test={freq_te[g]:.4f} diff={diff[g]:+.4f}{flag}")

train_zero_genes = freq_te[(freq_tr == 0) & (freq_te > 0)].sort_values(ascending=False)
print(f"\ntrain에서 단 한 번도 안 변한 유전자인데 test에는 있는 유전자: {len(train_zero_genes)}개")
if len(train_zero_genes):
    for g in train_zero_genes.head(15).index:
        n_te_g = int(is_mut_te[g].sum())
        print(f"  {g}: test {n_te_g}행 ({freq_te[g]:.4f}) — train 근거 0, synthetic(train crossover)로 재현 불가능")

# ================================================================ 3. 희귀 토큰(변이 문자열) 비교
print("\n" + "=" * 70)
print("3. train에 없는 (유전자, 변이토큰) 조합이 test에 반복되는 경우 (상위 15개)")
novel_tokens = {}
for g in genes:
    tr_tokens = set()
    for cell in train.loc[is_mut_tr[g], g]:
        tr_tokens.update(cell.split(" "))
    te_cells = test.loc[is_mut_te[g], g]
    if len(te_cells) == 0:
        continue
    for cell in te_cells:
        for tok in cell.split(" "):
            if tok not in tr_tokens:
                novel_tokens[(g, tok)] = novel_tokens.get((g, tok), 0) + 1
novel_sorted = sorted(novel_tokens.items(), key=lambda kv: -kv[1])
print(f"train에 없는 (유전자,토큰) 조합 총 {len(novel_sorted)}종 (test 전체 {len(test)}행 중 발생)")
for (g, tok), cnt in novel_sorted[:15]:
    print(f"  {g}:{tok} — test {cnt}행 (train에 이 정확한 토큰 없음, 같은 유전자 다른 변이는 train에 있을 수 있음)")

# ================================================================ 4. co-mutation 패턴 차이 (상위빈도 유전자 한정)
print("\n" + "=" * 70)
print("4. co-mutation(동반변이) 비율 차이 — 상위 150개 유전자 쌍 중 상위 15개")
top_genes = (freq_tr + freq_te).sort_values(ascending=False).head(150).index.tolist()
Xtr = is_mut_tr[top_genes].to_numpy(dtype=np.float32)
Xte = is_mut_te[top_genes].to_numpy(dtype=np.float32)
co_tr = (Xtr.T @ Xtr) / len(train)
co_te = (Xte.T @ Xte) / len(test)
co_diff = co_te - co_tr
iu = np.triu_indices(len(top_genes), k=1)
pairs = sorted(zip(iu[0], iu[1], co_diff[iu]), key=lambda t: -abs(t[2]))
for i, j, d in pairs[:15]:
    g1, g2 = top_genes[i], top_genes[j]
    print(f"  {g1} & {g2}: train={co_tr[i, j]:.4f} test={co_te[i, j]:.4f} diff={d:+.4f}")

# ================================================================ 5. 최근접 train 이웃 (자카드, 상위빈도 유전자)
print("\n" + "=" * 70)
print("5. test 행의 최근접 train 이웃 클래스 (자카드 유사도, 상위 300개 유전자 기준)")
top300 = (freq_tr + freq_te).sort_values(ascending=False).head(300).index.tolist()
Atr = is_mut_tr[top300].to_numpy(dtype=np.float32)
Ate = is_mut_te[top300].to_numpy(dtype=np.float32)
inter = Ate @ Atr.T                                    # (n_test, n_train)
sum_tr = Atr.sum(axis=1)
sum_te = Ate.sum(axis=1)
union = sum_te[:, None] + sum_tr[None, :] - inter
jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
best_idx = jacc.argmax(axis=1)
best_sim = jacc[np.arange(len(test)), best_idx]
nn_class = train[TARGET].to_numpy()[best_idx]

print(f"최근접 유사도 분포: mean={best_sim.mean():.3f} median={np.median(best_sim):.3f} "
      f"p10={np.percentile(best_sim,10):.3f} p90={np.percentile(best_sim,90):.3f}")
print(f"매우 낮은 유사도(<0.05, 어느 train 클래스와도 안 닮음) test 행: {(best_sim < 0.05).sum()}개 "
      f"({(best_sim < 0.05).mean():.1%}) — 이런 행은 synthetic 후보에서 제외 대상")

agree = (nn_class == pred_te)
print(f"\n'최근접 train 이웃 클래스'와 '현재 모델 예측 클래스'가 일치하는 비율: {agree.mean():.1%}")
print("불일치가 잦은 클래스 쌍 상위 10개 (nn_class → pred_te):")
mismatch = pd.Series(list(zip(nn_class[~agree], pred_te[~agree]))).value_counts().head(10)
for (a, b), cnt in mismatch.items():
    print(f"  최근접 train={a} / 모델예측={b}: {cnt}행")

# ================================================================ 6. burden 구간별 요약 저장
out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_synthetic_gbmlgg_lgg"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "ID": test["ID"], "burden": burden_te, "pred_hypothesis": pred_te,
    "nn_train_class": nn_class, "nn_jaccard": best_sim,
}).to_csv(out_dir / "v3_test_shift_row_level.csv", index=False)
diff.to_csv(out_dir / "v3_gene_freq_diff.csv")
print(f"\nsaved: {out_dir}")
