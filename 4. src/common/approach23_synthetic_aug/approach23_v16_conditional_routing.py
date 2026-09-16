"""approach23 v16: 9차 vs v2 OOF 개선/훼손 조건 분석 -> conditional routing/blending 구현 및 OOF 정직 검증.
기존 저장 OOF(9차/v2/16차)만 재사용. 새 모델은 오직 '어느 쪽을 더 신뢰할지'를 정하는 가벼운
로지스틱회귀 meta-router뿐이며, main.py와 동일한 group-twins 5-fold로 중첩 OOF 검증한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from main import ROOT, DATA, SEED, twin_groups, load_train
from features.features import gene_columns, TARGET, ID

train = load_train()
genes = gene_columns(train)
burden = (train[genes] != "WT").sum(axis=1).to_numpy()
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_); y = le.transform(train[TARGET])
n = len(train)

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
oof9 = np.load(ROOT / "6. experiments/2026-09-11_v4_xgb_mild_col_grp/oof_proba.npy")
oofv2 = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/oof_proba.npy")
oof16 = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
assert oof9.shape[0] == oofv2.shape[0] == oof16.shape[0] == n


def scale_norm(p):
    ps = p * s3
    return ps / ps.sum(1, keepdims=True)


p9, pv2, p16 = scale_norm(oof9), scale_norm(oofv2), scale_norm(oof16)
pred9, predv2, pred16 = p9.argmax(1), pv2.argmax(1), p16.argmax(1)
conf9, confv2 = p9.max(1), pv2.max(1)

BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 100), (101, 396), (397, 10**9)]
bin_names = [f"{lo}-{hi if hi<10**9 else '+'}" for lo, hi in BINS]


def macro_by_bin(pred):
    row = {}
    for (lo, hi), name in zip(BINS, bin_names):
        m = (burden >= lo) & (burden <= hi)
        row[name] = round(f1_score(y[m], pred[m], average="macro", labels=sorted(set(y[m].tolist()))), 4)
    row["전체"] = round(f1_score(y, pred, average="macro", labels=sorted(set(y.tolist()))), 4)
    return row


print("=" * 78)
print("2. baseline 성능 (9차 / v2 / 16차)")
base_tbl = pd.DataFrame([macro_by_bin(pred9), macro_by_bin(predv2), macro_by_bin(pred16)],
                        index=["9cha", "v2", "16cha"])
print(base_tbl.to_string())

correct9 = pred9 == y; correctv2 = predv2 == y
improve_mask = (~correct9) & correctv2      # 9차 wrong -> v2 correct
harm_mask = correct9 & (~correctv2)         # 9차 correct -> v2 wrong
disagree = pred9 != predv2
print(f"\n9차wrong->v2correct(개선): {improve_mask.sum()}  9차correct->v2wrong(훼손): {harm_mask.sum()}  "
      f"disagree 총: {disagree.sum()}  둘다correct: {(correct9&correctv2).sum()}  둘다wrong: {(~correct9&~correctv2).sum()}")

print("\n" + "=" * 78)
print("개선/훼손 집단의 burden, confidence 비교")
for name, m in [("개선(9wrong->v2correct)", improve_mask), ("훼손(9correct->v2wrong)", harm_mask)]:
    print(f"{name}: n={m.sum()}, burden mean={burden[m].mean():.1f} median={np.median(burden[m]):.1f}, "
          f"conf9 mean={conf9[m].mean():.3f}, confv2 mean={confv2[m].mean():.3f}, "
          f"conf_diff(v2-9) mean={(confv2[m]-conf9[m]).mean():+.3f}")

print("\nburden 구간별 개선/훼손 비율:")
for (lo, hi), name in zip(BINS, bin_names):
    m = (burden >= lo) & (burden <= hi)
    ni, nh, nd = improve_mask[m].sum(), harm_mask[m].sum(), disagree[m].sum()
    print(f"  {name}: disagree={nd}, 개선={ni}, 훼손={nh}, 개선-훼손={ni-nh}")

# ================================================================ meta-router features (전부 예측/모델 기반, 특정 유전자 미고정)
ent9 = np.apply_along_axis(lambda r: sp_entropy(r, base=2), 1, p9)
entv2 = np.apply_along_axis(lambda r: sp_entropy(r, base=2), 1, pv2)


def margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


marg9, margv2 = margin(p9), margin(pv2)
burden_bin_idx = np.digitize(burden, [4, 8, 16, 31, 101, 397])  # 0..6
onehot = np.eye(7)[burden_bin_idx]
Xmeta = np.column_stack([burden, conf9, confv2, confv2 - conf9, ent9, entv2, marg9, margv2, onehot])
feat_names = ["burden", "conf9", "confv2", "conf_diff", "ent9", "entv2", "marg9", "margv2"] + [f"bbin{i}" for i in range(7)]

print("\n" + "=" * 78)
print("4~5. fold-honest nested routing: disagree 행만 대상, 다른 fold의 (개선/훼손) 라벨로 LR 학습 -> 이 fold 예측")
groups = twin_groups(train)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_of = np.full(n, -1)
for k, (_, vai) in enumerate(skf.split(train, y, groups)):
    fold_of[vai] = k

route_proba = np.full(n, np.nan)   # P(prefer v2), disagree 행에만 채움
clear = disagree & (improve_mask | harm_mask)   # 라우터 학습에 쓸 수 있는 명확한 라벨(둘다 correct/wrong인 disagree는 제외)
y_route_all = improve_mask.astype(int)          # 1=v2가 맞음(선호), 0=9차가 맞음(선호) — clear 마스크에서만 의미 있음

fold_reports = []
for k in range(5):
    tr_idx = np.flatnonzero((fold_of != k) & clear)
    va_idx = np.flatnonzero((fold_of == k) & disagree)
    if len(tr_idx) < 30 or len(va_idx) == 0:
        continue
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xmeta[tr_idx], y_route_all[tr_idx])
    route_proba[va_idx] = lr.predict_proba(Xmeta[va_idx])[:, 1]
    va_clear = va_idx[np.isin(va_idx, np.flatnonzero(clear))]
    if len(va_clear):
        acc_route = ((route_proba[va_clear] > 0.5).astype(int) == y_route_all[va_clear]).mean()
    else:
        acc_route = float("nan")
    fold_reports.append((k, len(tr_idx), len(va_idx), round(acc_route, 3) if not np.isnan(acc_route) else None))
    print(f"fold{k}: 학습 disagree(명확라벨) {len(tr_idx)}행, 검증 disagree {len(va_idx)}행, "
          f"검증 중 명확라벨 라우팅정확도={acc_route:.3f}" if not np.isnan(acc_route) else f"fold{k}: 검증 명확라벨 없음")

print("\nfold별 라우터 정확도 요약(명확 라벨 disagree 행에서 '올바른 쪽을 골랐는가'):")
print(pd.DataFrame(fold_reports, columns=["fold", "n_train", "n_val_disagree", "route_acc_on_clear"]).to_string(index=False))

# ================================================================ conditional blending 최종 확률 구성
final_hard = pred9.copy()
final_hard[disagree] = np.where(np.nan_to_num(route_proba[disagree]) > 0.5, predv2[disagree], pred9[disagree])

alpha_row = np.where(disagree, np.nan_to_num(route_proba, nan=0.5), 0.0)
p_soft = p9.copy()
p_soft[disagree] = (1 - alpha_row[disagree, None]) * p9[disagree] + alpha_row[disagree, None] * pv2[disagree]
pred_soft = p_soft.argmax(1)

print("\n" + "=" * 78)
print("3. OOF 결과 표")
tbl = pd.DataFrame([macro_by_bin(pred9), macro_by_bin(predv2), macro_by_bin(pred16),
                    macro_by_bin(final_hard), macro_by_bin(pred_soft)],
                   index=["9cha", "v2", "16cha(blend w0.5 고정)", "D_hard_routing", "E_soft_routing"])
print(tbl.to_string())

for name, pred_new in [("D_hard_routing", final_hard), ("E_soft_routing", pred_soft)]:
    imp = (~correct9) & (pred_new == y) & disagree
    harm2 = correct9 & (pred_new != y) & disagree
    print(f"\n{name} vs 9차: disagree 중 개선 {imp.sum()}행, 훼손 {harm2.sum()}행, 비율 {imp.sum()}:{harm2.sum()}")
    vs16_diff = (pred_new != pred16).sum()
    print(f"{name} vs 16차(고정블렌드): 다른 행 {vs16_diff}개")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
tbl.to_csv(out_dir / "v16_routing_oof_summary.csv")
np.save(out_dir / "v16_route_proba_oof.npy", route_proba)
print(f"\nsaved: {out_dir}")
