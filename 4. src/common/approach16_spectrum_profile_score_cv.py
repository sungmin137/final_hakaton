"""접근16 검증: 아미노산 치환 필체를 암종별 점수표로 합산한다.

중요: train.csv만 읽는다. test.csv는 어떤 경로로도 읽지 않는다.

v11의 XGBoost는 380개 치환 비율을 개별 피처로 본다. 이 파일의
MultinomialNB는 'R->H, V->E, ...' 여러 약한 단서를 암종별 로그 점수로
더해 하나의 26차원 보조 확률을 만든다. 기존 v11 OOF와 로그 평균해
보조 탐정으로서만 평가한다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "4. src/common"))
from main import DATA, TARGET, gene_columns, twin_groups  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
EPS = 1e-12


def spectrum_counts(values: np.ndarray) -> np.ndarray:
    """380개 missense 방향쌍 + LoF/syn/other의 환자별 횟수.

    정상(WT)은 단서가 아니므로 세지 않는다. 비음수 횟수 행렬이므로
    MultinomialNB가 환자 안의 여러 약한 변이 필체를 합산할 수 있다.
    """
    out = np.zeros((len(values), 383), dtype=np.float32)
    for patient, row in enumerate(values):
        for raw in row[row != "WT"]:
            for token in raw.split():
                if token.endswith("*") or "fs" in token:
                    out[patient, 380] += 1
                    continue
                match = MISSENSE.match(token)
                if match is None:
                    out[patient, 382] += 1
                    continue
                before, after = match.group(1), match.group(3)
                if before == after:
                    out[patient, 381] += 1
                elif before in AA_INDEX and after in AA_INDEX:
                    # 대각선(동일 아미노산)을 뺀 20 x 19 = 380칸
                    offset = AA_INDEX[before] * 19 + (AA_INDEX[after] - (AA_INDEX[after] > AA_INDEX[before]))
                    out[patient, offset] += 1
                else:
                    out[patient, 382] += 1
    return out


def log_blend(*parts: tuple[np.ndarray, float]) -> np.ndarray:
    z = sum(np.log(np.maximum(probability, EPS)) * weight for probability, weight in parts)
    return softmax(z, axis=1)


def burden_report(y: np.ndarray, pred: np.ndarray, burden: np.ndarray) -> dict[str, float]:
    bands = {
        "0_10": burden <= 10,
        "11_30": (burden >= 11) & (burden <= 30),
        "31_100": (burden >= 31) & (burden <= 100),
        "101_plus": burden >= 101,
    }
    return {name: round(float(f1_score(y[mask], pred[mask], average="macro")), 5) for name, mask in bands.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4s-oof", type=Path, required=True, help="train OOF 확률 (v4s, 6201x26)")
    parser.add_argument("--p2-oof", type=Path, default=REPO / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp/oof_proba.npy")
    parser.add_argument("--out", type=Path, default=REPO / "6. experiments/2026-09-15_approach16_spectrum_profile_score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(DATA / "train.csv")  # train only
    genes = gene_columns(train)
    values = train[genes].to_numpy()
    X = spectrum_counts(values)
    burden = (values != "WT").sum(axis=1)
    le = LabelEncoder().fit(train[TARGET])
    y = le.transform(train[TARGET])
    classes = len(le.classes_)

    v4s = np.load(args.v4s_oof)
    p2 = np.load(args.p2_oof)
    assert v4s.shape == p2.shape == (len(train), classes)
    scale_path = REPO / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"
    scale_map = json.loads(scale_path.read_text())
    scales = np.array([scale_map[name] for name in le.classes_])
    v11 = log_blend((v4s, 0.5), (p2, 0.5))
    base_pred = (v11 * scales).argmax(axis=1)
    base_f1 = f1_score(y, base_pred, average="macro")

    oof = np.zeros((len(train), classes), dtype=np.float64)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y, groups=twin_groups(train)), start=1):
        scorer = MultinomialNB(alpha=0.5)
        scorer.fit(X[fit_idx], y[fit_idx])
        oof[valid_idx] = scorer.predict_proba(X[valid_idx])
        print(f"fold {fold}: score-table F1={f1_score(y[valid_idx], oof[valid_idx].argmax(1), average='macro'):.4f}")

    report: dict[str, object] = {
        "rule": "train.csv only; test.csv is never read",
        "seed": args.seed,
        "v11_reconstructed_macro_f1": round(float(base_f1), 6),
        "score_table_only_macro_f1": round(float(f1_score(y, oof.argmax(1), average="macro")), 6),
        "blends": [],
    }
    for weight in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        mix = log_blend((v11, 1 - weight), (oof, weight))
        pred = (mix * scales).argmax(axis=1)
        changed = pred != base_pred
        row = {
            "score_table_weight": weight,
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 6),
            "delta_vs_v11": round(float(f1_score(y, pred, average="macro") - base_f1), 6),
            "accuracy": round(float(accuracy_score(y, pred)), 6),
            "changed": int(changed.sum()),
            "helped": int((changed & (pred == y) & (base_pred != y)).sum()),
            "hurt": int((changed & (pred != y) & (base_pred == y)).sum()),
            "burden_macro_f1": burden_report(y, pred, burden),
        }
        report["blends"].append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "oof_spectrum_profile_score.npy", oof)
    (args.out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
