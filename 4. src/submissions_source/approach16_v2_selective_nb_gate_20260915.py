"""접근16 v2 최종 추론: v16과 NB 보조모델을 환자별 확신도로 선택한다.

임계값 (기존 모델 margin <= .18, NB margin >= .10)은 train OOF의
반복 그룹 교차검증으로 고정한 값이다. 이 파일은 탐색하지 않으며,
test는 이 최종 추론에서만 읽는다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4. src/common"))
from main import ID, TARGET, gene_columns, load_test, load_train  # noqa: E402
from postprocess.twin_rule import TwinRule  # noqa: E402

NAME = "approach16_v2_selective_nb_gate_20260915"
EPS = 1e-12
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
BASE_MARGIN_MAX = 0.18
NB_MARGIN_MIN = 0.10


def spectrum_counts(df: pd.DataFrame, genes: list[str]) -> np.ndarray:
    out = np.zeros((len(df), 383), dtype=np.float32)
    for i, row in enumerate(df[genes].to_numpy()):
        for raw in row[row != "WT"]:
            for token in raw.split():
                if token.endswith("*") or "fs" in token:
                    out[i, 380] += 1
                    continue
                match = MISSENSE.match(token)
                if match is None:
                    out[i, 382] += 1
                    continue
                before, after = match.group(1), match.group(3)
                if before == after:
                    out[i, 381] += 1
                elif before in AA_INDEX and after in AA_INDEX:
                    offset = AA_INDEX[before] * 19 + (AA_INDEX[after] - (after > before))
                    out[i, offset] += 1
                else:
                    out[i, 382] += 1
    return out


def log_blend(*parts: tuple[np.ndarray, float]) -> np.ndarray:
    logits = sum(w * np.log(np.maximum(p, EPS)) for p, w in parts)
    return softmax(logits, axis=1)


def margin(p: np.ndarray) -> np.ndarray:
    top2 = np.partition(p, -2, axis=1)
    return top2[:, -1] - top2[:, -2]


def main() -> None:
    train = load_train()
    test = load_test()  # 최종 답안 생성에서만 test를 읽는다.
    genes = gene_columns(train)
    assert genes == gene_columns(test)
    le = LabelEncoder().fit(train[TARGET])
    y = le.transform(train[TARGET])

    load = lambda rel: np.load(ROOT / rel)
    # v16 생성 시 callable 파트만 저장된다: part0=v4s, part2=v4sp,
    # part3=저차원 spectrum XGB. p2는 기존 고정 저장본을 재사용한다.
    v4s = load("6. experiments/submissions/approach14_v16_20260915_1130/test_proba_part0.npy")
    v2 = load("6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy")
    v4sp = load("6. experiments/submissions/approach14_v16_20260915_1130/test_proba_part2.npy")
    spec = load("6. experiments/submissions/approach14_v16_20260915_1130/test_proba_part3.npy")

    nb = MultinomialNB(alpha=0.5).fit(spectrum_counts(train, genes), y)
    nb_proba = nb.predict_proba(spectrum_counts(test, genes))
    base = log_blend((v4s, .45), (v2, .25), (v4sp, .15), (spec, .15))
    nb_replace = log_blend((v4s, .45), (v2, .25), (v4sp, .15), (nb_proba, .15))

    scale_map = json.loads((ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
    scales = np.array([scale_map[c] for c in le.classes_])
    use_nb = (margin(base * scales) <= BASE_MARGIN_MAX) & (margin(nb_proba) >= NB_MARGIN_MIN)
    final_proba = base.copy()
    final_proba[use_nb] = nb_replace[use_nb]
    pred = le.inverse_transform((final_proba * scales).argmax(1))

    sub = pd.read_csv(ROOT / "1. info/data/sample_submission.csv")
    assert len(sub) == len(test) == 2546 and (sub[ID] == test[ID]).all()
    sub[TARGET] = pred
    out_dir = ROOT / "5. submissions"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / f"{NAME}.csv"
    sub.to_csv(raw_path, index=False, encoding="UTF-8-sig")

    twin_pred, twin_hits = TwinRule().fit(train).apply(test, pred)
    sub[TARGET] = twin_pred
    twin_path = out_dir / f"{NAME}_twin_rule.csv"
    sub.to_csv(twin_path, index=False, encoding="UTF-8-sig")
    assert set(sub[TARGET]) <= set(le.classes_)
    print(f"saved {raw_path}")
    print(f"saved {twin_path}")
    print(f"NB gate changed {use_nb.sum()} / {len(test)} rows | twin matches {twin_hits}")


if __name__ == "__main__":
    main()
