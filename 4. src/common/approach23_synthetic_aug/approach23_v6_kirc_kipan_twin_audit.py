"""approach23 v6: KIRC/KIPAN twin_rule 감사. 학습/CV/제출 없음 — 기존 twin_rule 코드 + 저장된 test
prediction만 재사용해서, twin_rule이 실제 test KIRC/KIPAN 문제에 이미 충분히 개입하고 있는지 확인.

"twin 비율이 높다"와 "twin_rule이 이미 문제를 해결한다"는 다른 질문이다 — 이 스크립트는 후자를 잰다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from main import ROOT, DATA, load_train
from features.features import gene_columns, TARGET
from postprocess.twin_rule import TwinRule, FLIP, _hash_rows

train = load_train()
test = pd.read_csv(DATA / "test.csv").fillna("WT")
genes = gene_columns(train)
le = LabelEncoder().fit(train[TARGET]); classes = list(le.classes_)

s3 = np.array([json.load(open(ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))[c] for c in classes])
p7 = np.load(ROOT / "6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy") / s3
p7 = p7 / p7.sum(1, keepdims=True)
p2 = np.load(ROOT / "6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy")
z = np.log(p7 + 1e-9) * 0.5 + np.log(p2 + 1e-9) * 0.5
blend = np.exp(z - z.max(1, keepdims=True)); blend /= blend.sum(1, keepdims=True)
pred_pre = np.array(classes)[(blend * s3).argmax(1)]   # 17차(규칙 적용 전) 예측

tr = TwinRule().fit(train)
test_hash = _hash_rows(test, genes)
n_mut_te = (test[genes] != "WT").sum(axis=1).to_numpy()

is_twin = np.array([h in tr.lut for h in test_hash])
print("=" * 78)
print(f"0. test 전체 twin 비율: {is_twin.sum()}/{len(test)} = {is_twin.mean():.1%} (참고용, docs 05의 9.2%와 유사한지 확인)")


def twin_majority(h):
    labs = tr.lut.get(h)
    if not labs:
        return None, None
    from collections import Counter
    cnt = Counter(labs); best = cnt.most_common()
    if len(best) > 1 and best[0][1] == best[1][1]:
        return "TIE", None
    return best[0][0], FLIP.get(best[0][0], best[0][0])


twin_raw = []   # train 쪽 다수 라벨 (FLIP 적용 전)
twin_flip = []  # twin_rule이 실제로 내놓을 라벨 (FLIP 적용 후, 동점이면 None=모델 유지)
for h in test_hash:
    r, f = twin_majority(h)
    twin_raw.append(r); twin_flip.append(f)
twin_raw = np.array(twin_raw, dtype=object)
twin_flip = np.array(twin_flip, dtype=object)

pred_post, n_hit_total = tr.apply(test, pred_pre)
changed_any = pred_post != pred_pre
print(f"1. twin_rule 적용 시 전체 변경 행: {changed_any.sum()}행 (n_hit={n_hit_total}, 동점 유지 포함)")

kirc_kipan_change = changed_any & np.isin(pred_pre, ["KIRC", "KIPAN"]) & np.isin(pred_post, ["KIRC", "KIPAN"])
print(f"   그중 KIRC<->KIPAN 방향 변경: {kirc_kipan_change.sum()}행")

print("\n" + "=" * 78)
print("2~3. 예측이 KIRC 또는 KIPAN인 행의 twin/non-twin 분해 (규칙 적용 전 예측 기준)")
for c in ("KIRC", "KIPAN"):
    m = pred_pre == c
    n = m.sum()
    n_twin = (m & is_twin).sum()
    n_nontwin = n - n_twin
    print(f"\n예측={c}: 총 {n}행 | twin {n_twin}행({n_twin/max(n,1):.1%}) | non-twin {n_nontwin}행({n_nontwin/max(n,1):.1%})")
    sub_twin_labels = pd.Series(twin_raw[m & is_twin]).value_counts()
    print(f"  twin인 행의 train 다수라벨 분포: " + ", ".join(f"{k}={v}" for k, v in sub_twin_labels.items()) if n_twin else "  (twin 없음)")
    mismatch = (m & is_twin) & (twin_raw != c) & (twin_raw != "TIE")
    print(f"  twin인데 train 다수라벨이 현재 예측({c})과 다른 행: {mismatch.sum()}행")
    if mismatch.sum():
        print(f"    -> 그 twin 다수라벨 분포: " + ", ".join(f'{k}={v}' for k, v in pd.Series(twin_raw[mismatch]).value_counts().items()))
    flip_applied = mismatch & (twin_flip != None) & (twin_flip != c)  # noqa: E711
    n_actually_flipped = (changed_any & m & flip_applied).sum()
    print(f"  -> 그중 실제로 규칙 적용 시 라벨이 바뀐 행(동점 제외): {n_actually_flipped}행")

print("\n" + "=" * 78)
print("6. KIRC/KIPAN 예측 각각 twin / non-twin / (참고)burden 분해")
for c in ("KIRC", "KIPAN"):
    m = pred_pre == c
    for grp, mm in (("twin", m & is_twin), ("non-twin", m & ~is_twin)):
        if mm.sum() == 0:
            continue
        print(f"예측={c} {grp}: n={mm.sum()}, burden mean={n_mut_te[mm].mean():.1f} median={np.median(n_mut_te[mm]):.1f}")

print("\n" + "=" * 78)
print("7. 121행 KIRC<->KIPAN nearest-neighbor mismatch와 twin 여부의 관계")
row_te = pd.read_csv(ROOT / "6. experiments/2026-09-14_approach23_synthetic_gbmlgg_lgg/v3_test_shift_row_level.csv")
row_te["is_twin"] = is_twin
row_te["twin_raw"] = twin_raw
row_te["pred_pre17"] = pred_pre
mism = row_te[(row_te.nn_train_class.isin(["KIRC", "KIPAN"])) & (row_te.pred_hypothesis.isin(["KIRC", "KIPAN"])) & (row_te.nn_train_class != row_te.pred_hypothesis)]
print(f"(참고: v3 스캔은 다른 진단모델 예측(pred_hypothesis) 기준. 여기선 17차 예측(pred_pre17) 기준으로 재확인)")
mism17 = row_te[(row_te.nn_train_class.isin(["KIRC", "KIPAN"])) & (row_te.pred_pre17.isin(["KIRC", "KIPAN"])) & (row_te.nn_train_class != row_te.pred_pre17)]
print(f"17차 예측 기준 KIRC<->KIPAN 방향 mismatch: {len(mism17)}행")
print(f"  그중 twin: {mism17.is_twin.sum()}행 ({mism17.is_twin.mean():.1%})")
print(f"  그중 non-twin: {(~mism17.is_twin).sum()}행 ({(~mism17.is_twin).mean():.1%})")
if mism17.is_twin.sum():
    matched_twin_correctly = (mism17.is_twin) & (mism17.twin_raw == mism17.nn_train_class)
    print(f"  twin이면서 twin_raw==nn_train_class(즉 twin_rule이 맞는 방향을 가리킴): {matched_twin_correctly.sum()}행")

out_dir = ROOT / "6. experiments" / "2026-09-14_approach23_anomaly_scan"
row_te.to_csv(out_dir / "v6_kirc_kipan_twin_audit_rowlevel.csv", index=False)
print(f"\nsaved: {out_dir}")
