"""approach23 v4: 전체 접근 결과에 대한 anomaly-driven diagnostic scan (2단계).

학습/CV/제출 전부 금지. 기존에 저장된 result.json / oof_proba.npy / confusion_matrix.csv /
test_proba*.npy 만 읽어서 분석한다. main.py의 group_twins=True CV는 SEED=42 고정이고
fold 분할은 (train 행 순서, y, twin_groups(train))에만 의존하므로, group_twins=True로
저장된 모든 실험의 OOF는 **행 단위로 서로 정렬돼 있다** — 이 성질을 이용해 서로 다른 feature/모델의
OOF를 직접 비교한다.

출력은 화면 출력 + `6. experiments/2026-09-14_approach23_anomaly_scan/`에 csv로 저장.
기존 실험 결과물은 전혀 덮어쓰지 않는다.
"""
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix as sk_cm
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import ROOT, load_train
from features.features import gene_columns, TARGET

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET])
y = le.transform(train[TARGET])
classes = list(le.classes_)
n = len(train)
OUT = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
OUT.mkdir(parents=True, exist_ok=True)

# ================================================================ 0. 실험 인벤토리
print("=" * 78)
print("0. group_twins=True 이면서 oof_proba.npy가 있는 실험 전부 수집")
runs = []
for rp in sorted(glob.glob(str(ROOT / "6. experiments" / "**" / "result.json"), recursive=True)):
    d = json.load(open(rp))
    if not d.get("group_twins"):
        continue
    npy = Path(rp).parent / "oof_proba.npy"
    if not npy.exists():
        continue
    proba = np.load(npy)
    if proba.shape[0] != n:
        continue     # 행 수가 다르면 다른 train 스냅샷/분할 — 비교 대상에서 제외
    runs.append(dict(name=Path(rp).parent.name, path=str(npy), features=d.get("features"),
                      model=d.get("model"), balanced=d.get("balanced"), oof_macro_f1=d.get("oof_macro_f1"),
                      proba=proba))
print(f"비교 가능(행 수 {n} 일치, group_twins) 실험: {len(runs)}개")
for r in runs:
    print(f"  {r['name']:45s} features={r['features']:<10} model={r['model']:<5} "
          f"balanced={r['balanced']} oof_macro_f1={r['oof_macro_f1']}")

# ================================================================ 1. class x approach F1 매트릭스
print("\n" + "=" * 78)
print("1. class x approach 매트릭스 (각 실험의 저장된 per_class_f1 재사용, 재계산 없음)")
class_mat = {}
for rp in sorted(glob.glob(str(ROOT / "6. experiments" / "**" / "result.json"), recursive=True)):
    d = json.load(open(rp))
    if not d.get("group_twins") or not d.get("per_class_f1"):
        continue
    class_mat[Path(rp).parent.name] = d["per_class_f1"]
cls_df = pd.DataFrame(class_mat).T   # index=approach, columns=class
cls_df = cls_df[[c for c in classes if c in cls_df.columns]]
cls_df.to_csv(OUT / "class_x_approach_f1.csv")
print(f"실험 {cls_df.shape[0]}개 x 클래스 {cls_df.shape[1]}개 매트릭스 저장")

summary = pd.DataFrame({
    "mean_f1": cls_df.mean(), "std_f1": cls_df.std(), "min_f1": cls_df.min(),
    "max_f1": cls_df.max(), "range": cls_df.max() - cls_df.min(), "n_approaches": cls_df.count(),
}).sort_values("mean_f1")
print("\n[A. 반복적으로 낮은 class] mean F1 하위 8개 (n_approaches = 몇 개 실험에서 관측됐는지)")
print(summary.head(8).to_string())
print("\n[B. 접근 간 변동이 큰 class] std_f1 상위 8개")
print(summary.sort_values("std_f1", ascending=False).head(8).to_string())

# ================================================================ 2. burden x approach F1 매트릭스
print("\n" + "=" * 78)
print("2. burden 구간 x approach macro F1 매트릭스 (기존 OOF 재사용, 재학습 없음)")
BINS = [(0, 0), (1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 10**9)]
bin_names = [f"{lo}-{hi if hi < 10**9 else '+'}" for lo, hi in BINS]
burden_rows = {}
for r in runs:
    pred = r["proba"].argmax(1)
    vals = []
    for lo, hi in BINS:
        m = (burden >= lo) & (burden <= hi)
        yt, yp = y[m], pred[m]
        present = sorted(set(yt.tolist()))
        vals.append(f1_score(yt, yp, average="macro", labels=present) if len(present) else np.nan)
    burden_rows[r["name"]] = vals
burden_df = pd.DataFrame(burden_rows, index=bin_names).T
burden_df.to_csv(OUT / "burden_x_approach_f1.csv")
print(burden_df.round(3).to_string())
print("\n구간별 (전체 실험 평균, 표준편차) — '모든 접근이 공통으로 약한 구간' 후보")
print(pd.DataFrame({"mean": burden_df.mean(), "std": burden_df.std(), "min": burden_df.min(), "max": burden_df.max()}).round(4).to_string())

# ================================================================ 3. confusion edge 반복성
print("\n" + "=" * 78)
print("3. confusion edge(A→B) 반복성 — 여러 실험에서 공통으로 나타나는 오답 방향")
edge_counter = {}
edge_detail = {}
for r in runs:
    pred = r["proba"].argmax(1)
    cm = sk_cm(y, pred, labels=range(len(classes)))
    for i in range(len(classes)):
        n_i = cm[i].sum()
        if n_i == 0:
            continue
        for j in range(len(classes)):
            if i == j or cm[i, j] == 0:
                continue
            share = cm[i, j] / n_i   # 실제 클래스 i 중 j로 새는 비율
            if share < 0.05:         # 노이즈성 극소 오답 제외
                continue
            key = (classes[i], classes[j])
            edge_counter[key] = edge_counter.get(key, 0) + 1
            edge_detail.setdefault(key, []).append(round(share, 3))
edge_rows = [(a, b, cnt, np.mean(edge_detail[(a, b)]), np.std(edge_detail[(a, b)]))
             for (a, b), cnt in edge_counter.items()]
edge_df = pd.DataFrame(edge_rows, columns=["from", "to", "n_approaches_present", "mean_share", "std_share"])
edge_df = edge_df.sort_values(["n_approaches_present", "mean_share"], ascending=False)
edge_df.to_csv(OUT / "confusion_edge_repetition.csv", index=False)
print(f"(비교 대상 {len(runs)}개 실험 중) 5% 이상 비중으로 반복되는 edge 상위 20개")
print(edge_df.head(20).to_string(index=False))

# ================================================================ 4. prediction 과대/과소예측 (OOF vs 실제)
print("\n" + "=" * 78)
print("4. OOF prediction 클래스 비중 vs 실제 train 비중 (여러 실험 평균)")
true_share = pd.Series(y).map(lambda i: classes[i]).value_counts(normalize=True)
pred_share_rows = {}
for r in runs:
    pred = r["proba"].argmax(1)
    pred_share_rows[r["name"]] = pd.Series(pred).map(lambda i: classes[i]).value_counts(normalize=True)
pred_share_df = pd.DataFrame(pred_share_rows).T.reindex(columns=classes).fillna(0)
bias = (pred_share_df.mean() - true_share).sort_values(key=np.abs, ascending=False)
print("실제 비중 대비 OOF 예측 비중 평균 괴리 (상위 10개, 여러 실험 평균)")
for c in bias.head(10).index:
    print(f"  {c}: 실제={true_share.get(c,0):.3f} 예측평균={pred_share_df[c].mean():.3f} diff={bias[c]:+.3f} (std={pred_share_df[c].std():.3f})")

# ================================================================ 5. 모델 계열 간 disagreement (같은 features=v3)
print("\n" + "=" * 78)
print("5. 같은 feature(v3), 다른 모델 계열 간 예측 불일치 (xgb vs lgbm vs mlp)")
v3_runs = {r["model"]: r for r in runs if r["features"] == "v3"}
if len(v3_runs) >= 2:
    preds = {m: r["proba"].argmax(1) for m, r in v3_runs.items()}
    models = list(preds)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            agree = (preds[a] == preds[b]).mean()
            print(f"  {a} vs {b}: 일치율 {agree:.1%}")
    if len(models) >= 3:
        all_agree = np.all([preds[m] == y for m in models], axis=0)
        all_wrong = np.all([preds[m] != y for m in models], axis=0)
        print(f"  {len(models)}개 모델 전부 정답: {all_agree.sum()}행 ({all_agree.mean():.1%})")
        print(f"  {len(models)}개 모델 전부 오답(다른 계열인데 동시에 틀림): {all_wrong.sum()}행 ({all_wrong.mean():.1%})")
        if all_wrong.sum() > 0:
            wrong_true = pd.Series(y[all_wrong]).map(lambda i: classes[i]).value_counts()
            print("    전부 오답인 행의 실제 클래스 분포 상위 8개:")
            print("    " + ", ".join(f"{k}={v}" for k, v in wrong_true.head(8).items()))
else:
    print("  v3 feature로 저장된 다른 모델 계열 OOF가 2개 미만이라 생략")

# ================================================================ 6. feature intervention pattern (기존 표 기반, 재계산 아님)
print("\n" + "=" * 78)
print("6. 같은 params/model, feature만 다른 쌍의 전체/burden 변화 (v4 vs v4p, v4p vs v4p2 등)")
by_feat = {r["features"]: r for r in runs if r["model"] == "xgb" and not r["balanced"]}
pairs_to_check = [("v4", "v4p"), ("v4p", "v4p2"), ("v4p", "v4p:pos"), ("v4p", "v4p:prop"),
                   ("v4p", "v4p:pair"), ("v4p", "v4p:band")]
for a, b in pairs_to_check:
    if a not in by_feat or b not in by_feat:
        continue
    pa, pb = by_feat[a]["proba"].argmax(1), by_feat[b]["proba"].argmax(1)
    f_a = f1_score(y, pa, average="macro"); f_b = f1_score(y, pb, average="macro")
    per_a = f1_score(y, pa, average=None, labels=range(len(classes)))
    per_b = f1_score(y, pb, average=None, labels=range(len(classes)))
    diff = pd.Series(per_b - per_a, index=classes).sort_values()
    print(f"\n{a} -> {b}: 전체 macro {f_a:.4f} -> {f_b:.4f} ({f_b-f_a:+.4f})")
    print(f"  가장 나빠진 클래스: " + ", ".join(f"{c}({diff[c]:+.3f})" for c in diff.head(3).index))
    print(f"  가장 좋아진 클래스: " + ", ".join(f"{c}({diff[c]:+.3f})" for c in diff.tail(3).index))

print(f"\nsaved: {OUT}")
