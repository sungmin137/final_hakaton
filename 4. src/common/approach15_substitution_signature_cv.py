"""접근 15: 아미노산 치환 스펙트럼 보조 모델의 train 전용 OOF 검증.

유전자 이름이나 암종별 빈도는 사용하지 않는다. 각 변이를 20x20 아미노산
치환표(예: R->C)로 바꾸고, LoF/동의/기타 변이 수도 함께 쓴다. 따라서
기존의 유전자 중심 XGBoost와 다른 힌트를 주는지 검증할 수 있다.

중요: 이 파일은 test.csv를 읽지 않는다. 제출 후보가 되기 전까지 train OOF
점수만 평가한다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4. src/common"))
from main import DATA, SEED, gene_columns, twin_groups  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
EPS = 1e-12


def substitution_signature(values: np.ndarray) -> np.ndarray:
    """행마다 400개 치환 비율 + missense/LoF/syn/기타 개수를 만든다."""
    out = np.zeros((len(values), 404), dtype=np.float32)
    for row_index, row in enumerate(values):
        for raw_value in row[row != "WT"]:
            for token in raw_value.split(" "):
                if token.endswith("*") or "fs" in token:
                    out[row_index, 400] += 1  # 단백질이 중간에 끊기거나 틀이 밀림
                    continue
                match = MISSENSE.match(token)
                if match is None:
                    out[row_index, 403] += 1
                    continue
                before, after = match.group(1), match.group(3)
                if before == after:
                    out[row_index, 401] += 1  # 글자는 바뀌었지만 같은 아미노산
                else:
                    out[row_index, AA_INDEX[before] * 20 + AA_INDEX[after]] += 1
    out[:, 402] = out[:, :400].sum(axis=1)
    denominator = np.maximum(out[:, :400].sum(axis=1, keepdims=True), 1)
    out[:, :400] /= denominator
    return out


def burden_segments(y: np.ndarray, pred: np.ndarray, burden: np.ndarray) -> dict[str, float]:
    masks = {
        "0_10": burden <= 10,
        "11_30": (burden >= 11) & (burden <= 30),
        "31_100": (burden >= 31) & (burden <= 100),
        "101_plus": burden >= 101,
    }
    return {name: float(f1_score(y[mask], pred[mask], average="macro")) for name, mask in masks.items()}


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    genes = gene_columns(train)
    values = train[genes].to_numpy()
    burden = (values != "WT").sum(axis=1)
    X = substitution_signature(values)
    label_encoder = LabelEncoder().fit(train["SUBCLASS"])
    y = label_encoder.transform(train["SUBCLASS"])

    # 16차(v3) OOF는 이미 train의 쌍둥이 그룹 CV로 만든 확률이다.
    raw_base = np.load(ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/blend_w05_oof.npy")
    scale_json = json.loads((ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
    scales = np.array([scale_json[class_name] for class_name in label_encoder.classes_])
    base_pred = (raw_base * scales).argmax(axis=1)

    oof = np.zeros_like(raw_base)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(cv.split(X, y, groups=twin_groups(train)), start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.12, class_weight="balanced", max_iter=2000, solver="lbfgs"),
        )
        model.fit(X[fit_index], y[fit_index])
        oof[valid_index] = model.predict_proba(X[valid_index])
        print(f"fold {fold}: {f1_score(y[valid_index], oof[valid_index].argmax(axis=1), average='macro'):.4f}")

    rows = []
    base_f1 = f1_score(y, base_pred, average="macro")
    for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
        # 확률을 단순 더하지 않고 로그 공간에서 섞어, 한 모델의 과신을 완화한다.
        blended = softmax((1 - weight) * np.log(np.maximum(raw_base, EPS)) + weight * np.log(np.maximum(oof, EPS)), axis=1)
        pred = (blended * scales).argmax(axis=1)
        changed = pred != base_pred
        rows.append({
            "signature_weight": weight,
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "delta_vs_v3": float(f1_score(y, pred, average="macro") - base_f1),
            "accuracy": float(accuracy_score(y, pred)),
            "changed": int(changed.sum()),
            "helped": int((changed & (pred == y) & (base_pred != y)).sum()),
            "hurt": int((changed & (pred != y) & (base_pred == y)).sum()),
            "burden_segments": burden_segments(y, pred, burden),
        })

    output = ROOT / "6. experiments/2026-09-14_approach15_substitution_signature"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "oof_proba.npy", oof)
    result = {
        "base_v3_macro_f1": float(base_f1),
        "signature_only_macro_f1": float(f1_score(y, oof.argmax(axis=1), average="macro")),
        "blends": rows,
        "note": "train 전용. test.csv를 읽지 않음.",
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(pd.DataFrame([{key: value for key, value in row.items() if key != "burden_segments"} for row in rows]).to_string(index=False))


if __name__ == "__main__":
    main()
