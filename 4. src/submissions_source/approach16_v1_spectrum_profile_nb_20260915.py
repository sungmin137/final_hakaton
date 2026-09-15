"""접근16 최종 추론: v16의 spec XGBoost를 아미노산 필체 NB로 교체한다.

가중치 .45/.25/.15/.15는 train OOF에서 이미 확정된 상수다. 이 파일은
가중치를 탐색하지 않으며 test는 최종 predict 단계에서만 읽는다.
최신 sungmin 접근14 v16의 저장 확률 파일이 같은 저장소에 있어야 한다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4. src/common"))
from main import ID, TARGET, gene_columns, load_test, load_train  # noqa: E402
from postprocess.twin_rule import TwinRule  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
EPS = 1e-12
NAME = "approach16_v1_spectrum_profile_nb_20260915"


def spectrum_counts(df, genes: list[str]) -> np.ndarray:
    """380 치환쌍과 LoF/syn/other 횟수. 행 안의 변이만 센다."""
    values = df[genes].to_numpy()
    out = np.zeros((len(df), 383), dtype=np.float32)
    for i, row in enumerate(values):
        for raw in row[row != "WT"]:
            for token in raw.split():
                if token.endswith("*") or "fs" in token:
                    out[i, 380] += 1; continue
                match = MISSENSE.match(token)
                if match is None:
                    out[i, 382] += 1; continue
                before, after = match.group(1), match.group(3)
                if before == after:
                    out[i, 381] += 1
                elif before in AA_INDEX and after in AA_INDEX:
                    offset = AA_INDEX[before] * 19 + (AA_INDEX[after] - (AA_INDEX[after] > AA_INDEX[before]))
                    out[i, offset] += 1
                else:
                    out[i, 382] += 1
    return out


def main() -> None:
    train = load_train()
    le = LabelEncoder().fit(train[TARGET])
    y = le.transform(train[TARGET])
    genes = gene_columns(train)

    # 최종 답안 생성 단계에서만 test를 읽는다.
    test = load_test()
    assert genes == gene_columns(test)

    v4s = np.load(ROOT / "6. experiments/submissions/approach14_v11_20260914_1627/test_proba_part0.npy")
    v2 = np.load(ROOT / "6. experiments/submissions/approach14_v3_20260913_2132/test_proba_v2.npy")
    v4sp = np.load(ROOT / "6. experiments/submissions/approach14_v12_20260914_1627/test_proba_part3.npy")
    nb = MultinomialNB(alpha=0.5).fit(spectrum_counts(train, genes), y)
    nb_proba = nb.predict_proba(spectrum_counts(test, genes))

    z = (.45 * np.log(np.maximum(v4s, EPS)) + .25 * np.log(np.maximum(v2, EPS))
         + .15 * np.log(np.maximum(v4sp, EPS)) + .15 * np.log(np.maximum(nb_proba, EPS)))
    proba = softmax(z, axis=1)
    scale_map = json.loads((ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
    scales = np.array([scale_map[c] for c in le.classes_])
    pred = le.inverse_transform((proba * scales).argmax(1))

    import pandas as pd
    sub = pd.read_csv(ROOT / "1. info/data/sample_submission.csv")
    assert len(sub) == len(test) == 2546 and (sub[ID] == test[ID]).all()
    sub[TARGET] = pred
    sub.to_csv(ROOT / "5. submissions" / f"{NAME}.csv", index=False, encoding="UTF-8-sig")

    twin_pred, _ = TwinRule().fit(train).apply(test, pred)
    sub[TARGET] = twin_pred
    sub.to_csv(ROOT / "5. submissions" / f"{NAME}_twin_rule.csv", index=False, encoding="UTF-8-sig")


if __name__ == "__main__":
    main()
