"""BRCA↔OV(141건), BRCA↔PRAD(134건) 재판기 확률 결합 정직 CV.
v8(HNSC/STES, GBMLGG/LGG)과 같은 틀: 16차 블렌드(9차+v2, w=0.5) OOF에 재판기 비율을 기하 혼합.
재판기는 각 fold 학습 부분에서만 fit (test/valid 통계 미사용).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "4. src/common"))

import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, load_train, twin_groups, SEED
from approach14_cw_plus.pair_referee import PairReferee, mix_pair

OUT = Path(__file__).parent
SCALE_PATH = ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"
BLEND_OOF_PATH = ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy"

PAIRS = {"BRCA_OV": ("BRCA", "OV"), "BRCA_PRAD": ("BRCA", "PRAD")}


def macro_f1_scaled(proba, y, s3):
    pred = (proba * s3).argmax(1)
    return f1_score(y, pred, average="macro")


def main():
    train = load_train()
    le = LabelEncoder().fit(train["SUBCLASS"])
    y = le.transform(train["SUBCLASS"])
    classes = list(le.classes_)
    s3 = np.array([json.load(open(SCALE_PATH))[c] for c in classes])

    blend = np.load(BLEND_OOF_PATH)
    baseline = macro_f1_scaled(blend, y, s3)
    print(f"baseline(16차 블렌드 OOF, 배율 적용) macro F1 = {baseline:.4f}")

    groups = twin_groups(train)
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, y, groups))

    for name, pair in PAIRS.items():
        a, b = pair
        ia, ib = classes.index(a), classes.index(b)
        n_pair = int(((train["SUBCLASS"] == a) | (train["SUBCLASS"] == b)).sum())
        ref_oof = np.zeros(len(train))
        for tri, vai in splits:
            ref = PairReferee(pair, mode="top40").fit(train.iloc[tri])
            ref_oof[vai] = ref.proba(train.iloc[vai])
        np.save(OUT / f"referee_oof_{name}.npy", ref_oof)

        print(f"\n== {name} (train {n_pair}건) ==")
        best = (None, -1)
        for v in (0.2, 0.35, 0.5, 0.7, 1.0):
            mixed = mix_pair(blend.copy(), ia, ib, ref_oof, v)
            f1 = macro_f1_scaled(mixed, y, s3)
            print(f"  v={v:>4}  macro F1={f1:.4f}  (기준 대비 {f1 - baseline:+.4f})")
            if f1 > best[1]:
                best = (v, f1)
        print(f"  최고 v={best[0]} → {best[1]:.4f} ({best[1]-baseline:+.4f})")

    print("\n== 둘 다 순차 적용 (BRCA_OV 먼저 최적 v, 그 위에 BRCA_PRAD) ==")
    ref_ov = np.load(OUT / "referee_oof_BRCA_OV.npy")
    ref_prad = np.load(OUT / "referee_oof_BRCA_PRAD.npy")
    ia_ov, ib_ov = classes.index("BRCA"), classes.index("OV")
    ia_pr, ib_pr = classes.index("BRCA"), classes.index("PRAD")
    for v1 in (0.2, 0.35, 0.5):
        m1 = mix_pair(blend.copy(), ia_ov, ib_ov, ref_ov, v1)
        for v2 in (0.2, 0.35, 0.5):
            m2 = mix_pair(m1.copy(), ia_pr, ib_pr, ref_prad, v2)
            f1 = macro_f1_scaled(m2, y, s3)
            print(f"  v_ov={v1} v_prad={v2}  macro F1={f1:.4f}  ({f1-baseline:+.4f})")


if __name__ == "__main__":
    main()
