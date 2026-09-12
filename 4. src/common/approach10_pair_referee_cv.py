"""접근 10 — V4가 헷갈린 암종 쌍만 다시 판정하는 train 전용 검증.

첫 모델은 기존 V4 XGBoost다. 첫 모델의 1·2순위가 미리 정한 혼동 쌍이고
두 확률 차이가 작을 때만, 그 두 암종으로 학습한 작은 LogisticRegression이
원본 유전자 '변이 있음/없음'을 보고 답을 다시 고른다.

KIRC↔KIPAN, LGG↔GBMLGG는 동일 프로필에 다른 라벨이 붙은 쌍둥이 충돌이므로
의도적으로 제외한다. 이 파일은 train.csv만 읽고 test.csv를 절대 읽지 않는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import (  # noqa: E402
    DATA,
    ROOT,
    SEED,
    XGB_PARAMS,
    FeatureMaker,
    gene_columns,
    twin_groups,
)
from main import xgb  # noqa: E402


# 과거 V4 OOF에서 반복적으로 확인된 혼동만 고정한다. 이 목록은 실행 중 라벨을
# 보고 새로 만들지 않는다. 쌍둥이 충돌 두 쌍은 넣지 않는다.
PAIR_NAMES = (
    ("BRCA", "OV"),
    ("BRCA", "PRAD"),
    ("HNSC", "STES"),
    ("LIHC", "STES"),
    ("LUAD", "STES"),
    ("LUSC", "STES"),
    ("OV", "PAAD"),
)
GAPS = np.array([0.05, 0.10, 0.15, 0.20, 0.30])


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
    }


def pair_mask(top1: np.ndarray, top2: np.ndarray, a: int, b: int) -> np.ndarray:
    """상위 두 답이 순서와 무관하게 (a, b)인 행만 고른다."""
    return ((top1 == a) & (top2 == b)) | ((top1 == b) & (top2 == a))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    le = LabelEncoder()
    y = le.fit_transform(train["SUBCLASS"])
    class_id = {name: i for i, name in enumerate(le.classes_)}
    pairs = [(class_id[a], class_id[b]) for a, b in PAIR_NAMES]
    genes = gene_columns(train)

    # 작은 재판 모델은 생물학 지식이나 세부 변이 문자열을 추가하지 않는다.
    # 원본 표를 'WT인가 아닌가'라는 가장 보수적인 형태로만 바꾼다.
    X_binary = (train[genes].to_numpy() != "WT").astype(np.int8)
    groups = twin_groups(train)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(train, y, groups))

    base_oof = np.zeros((len(train), len(le.classes_)), dtype=np.float32)
    routed_pred = np.tile(np.zeros(len(train), dtype=np.int16), (len(GAPS), 1))
    fold_of = np.zeros(len(train), dtype=np.int8)
    fold_rows: list[dict] = []
    t0 = time.time()

    for fold, (tri, vai) in enumerate(splits):
        print(f"\n========== Fold {fold + 1} ==========")

        # 1차 판정기: fold의 train 부분으로만 V4를 만든다.
        fm = FeatureMaker("v4").fit(train.iloc[tri])
        first = xgb.XGBClassifier(**XGB_PARAMS)
        first.fit(fm.transform(train.iloc[tri]), y[tri])
        proba = first.predict_proba(fm.transform(train.iloc[vai]))
        base_oof[vai] = proba
        base_pred = proba.argmax(1).astype(np.int16)
        routed_pred[:, vai] = base_pred
        fold_of[vai] = fold

        order = np.argsort(proba, axis=1)
        top1, top2 = order[:, -1], order[:, -2]
        gap = proba[np.arange(len(vai)), top1] - proba[np.arange(len(vai)), top2]
        replaced_by_gap = np.zeros((len(GAPS), len(vai)), dtype=bool)

        # 2차 재판기: 각 쌍의 환자만으로 학습한다. validation 환자의 정답은 사용하지 않는다.
        for (a, b), (a_name, b_name) in zip(pairs, PAIR_NAMES):
            pair_train = tri[(y[tri] == a) | (y[tri] == b)]
            judge = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=SEED + fold,
            )
            judge.fit(X_binary[pair_train], y[pair_train])
            judge_pred = judge.predict(X_binary[vai]).astype(np.int16)

            eligible = pair_mask(top1, top2, a, b)
            if not eligible.any():
                continue
            print(f"  {a_name} vs {b_name}: 재판 후보 {eligible.sum()}명")
            for gi, threshold in enumerate(GAPS):
                use = eligible & (gap <= threshold)
                # 한 행은 상위 2개 조합 하나에만 속하므로 중복 교체되지 않는다.
                routed_pred[gi, vai[use]] = judge_pred[use]
                replaced_by_gap[gi, use] = True

        base_s = score(y[vai], base_pred)
        fold_rows.append({
            "fold": fold + 1,
            "base_macro_f1": base_s["macro_f1"],
            "base_accuracy": base_s["accuracy"],
            "seconds_elapsed": time.time() - t0,
        })
        print(f"  V4 기본: Macro F1 {base_s['macro_f1']:.4f}, Accuracy {base_s['accuracy']:.4f}")

    base_pred = base_oof.argmax(1)
    summary: list[dict] = []
    for gi, threshold in enumerate(GAPS):
        pred = routed_pred[gi]
        changed = pred != base_pred
        helped = changed & (pred == y) & (base_pred != y)
        hurt = changed & (pred != y) & (base_pred == y)
        row = {
            "gap_threshold": float(threshold),
            **score(y, pred),
            "routed_patients": int((pred != base_pred).sum()),
            "helped": int(helped.sum()),
            "hurt": int(hurt.sum()),
        }
        summary.append(row)

    # 각 fold의 문턱값은 나머지 4개 fold의 OOF 결과만 보고 선택한다.
    # 그래서 '가장 좋았던 문턱'을 전체 정답으로 고른 낙관 점수와 구분할 수 있다.
    meta_pred = base_pred.copy()
    chosen: list[dict] = []
    for fold in range(5):
        fit_idx, eval_idx = fold_of != fold, fold_of == fold
        candidates = [f1_score(y[fit_idx], routed_pred[gi, fit_idx], average="macro") for gi in range(len(GAPS))]
        gi = int(np.argmax(candidates))
        meta_pred[eval_idx] = routed_pred[gi, eval_idx]
        chosen.append({"fold": fold + 1, "chosen_gap": float(GAPS[gi]), "fit_macro_f1": float(candidates[gi])})

    result = {
        "description": "V4 상위 2개가 고정 혼동 쌍이고 확률 차이가 작을 때만 pairwise LR로 재판",
        "pairs": [list(p) for p in PAIR_NAMES],
        "excluded_twin_conflicts": [["KIRC", "KIPAN"], ["LGG", "GBMLGG"]],
        "base": score(y, base_pred),
        "fixed_gap_results": summary,
        "heldout_gap_selection": {
            **score(y, meta_pred),
            "chosen_by_fold": chosen,
            "changed_patients": int((meta_pred != base_pred).sum()),
        },
        "folds": fold_rows,
        "elapsed_seconds": time.time() - t0,
    }
    out = ROOT / "6. experiments" / "2026-09-10_approach10_pair_referee"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    np.save(out / "base_oof_proba.npy", base_oof)
    np.save(out / "routed_oof_predictions.npy", routed_pred)

    print("\n==============================")
    print("V4 기본:", score(y, base_pred))
    print(pd.DataFrame(summary).to_string(index=False))
    print("문턱값 교차선택:", result["heldout_gap_selection"])
    print("저장:", out)


if __name__ == "__main__":
    main()
