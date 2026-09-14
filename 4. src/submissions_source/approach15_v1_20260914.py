"""접근15 v1 제출 후보: v3 80% + 아미노산 치환 스펙트럼 LR 20% + 쌍둥이 규칙.

가중치 0.20은 train OOF에서만 확정했다. 이 추론 단계에서 처음 test를 읽으며,
test의 분포/예측 개수로 어떠한 선택이나 보정도 하지 않는다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4. src/common"))
from main import ID, TARGET, load_test, load_train, gene_columns  # noqa: E402
from postprocess.twin_rule import TwinRule  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
EPS = 1e-12
SIGNATURE_WEIGHT = 0.20
NAME = "approach15_v1_substitution_signature_20260914"


def substitution_signature(values: np.ndarray) -> np.ndarray:
    """유전자명을 버린 400개 아미노산 치환 비율 + 4개 변이 유형 요약."""
    out = np.zeros((len(values), 404), dtype=np.float32)
    for row_index, row in enumerate(values):
        for raw_value in row[row != "WT"]:
            for token in raw_value.split(" "):
                if token.endswith("*") or "fs" in token:
                    out[row_index, 400] += 1
                    continue
                match = MISSENSE.match(token)
                if match is None:
                    out[row_index, 403] += 1
                    continue
                before, after = match.group(1), match.group(3)
                # X 등 표준 20개 아미노산 밖 표기는 특정 치환 칸에 억지로 넣지 않는다.
                if before not in AA_INDEX or after not in AA_INDEX:
                    out[row_index, 403] += 1
                    continue
                if before == after:
                    out[row_index, 401] += 1
                else:
                    out[row_index, AA_INDEX[before] * 20 + AA_INDEX[after]] += 1
    out[:, 402] = out[:, :400].sum(axis=1)
    out[:, :400] /= np.maximum(out[:, :400].sum(axis=1, keepdims=True), 1)
    return out


def main() -> None:
    train = load_train()
    test = load_test()  # 최종 추론에서만 사용
    genes = gene_columns(train)
    label_encoder = LabelEncoder().fit(train[TARGET])

    # 최고 LB인 16차(v3)의 저장된, 클래스 배율 전 원시 확률
    base = np.load(ROOT / "6. experiments/submissions/approach14_v3_20260913_2132/test_proba_raw_blend.npy")

    signature_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.12, class_weight="balanced", max_iter=2000, solver="lbfgs"),
    )
    signature_model.fit(substitution_signature(train[genes].to_numpy()), label_encoder.transform(train[TARGET]))
    signature_proba = signature_model.predict_proba(substitution_signature(test[genes].to_numpy()))

    raw = softmax(
        (1 - SIGNATURE_WEIGHT) * np.log(np.maximum(base, EPS))
        + SIGNATURE_WEIGHT * np.log(np.maximum(signature_proba, EPS)),
        axis=1,
    )

    scale_path = ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"
    import json
    scale_map = json.loads(scale_path.read_text())
    scales = np.array([scale_map[class_name] for class_name in label_encoder.classes_])
    prediction = label_encoder.inverse_transform((raw * scales).argmax(axis=1))

    # train만 이용해 만든 완전 동일 프로필 규칙
    prediction_twin, twin_hits = TwinRule().fit(train).apply(test, prediction)

    import pandas as pd
    submission = pd.read_csv(ROOT / "1. info/data/sample_submission.csv")
    assert (submission[ID] == test[ID]).all()
    submission[TARGET] = prediction
    plain_path = ROOT / "5. submissions" / f"{NAME}.csv"
    submission.to_csv(plain_path, index=False, encoding="UTF-8-sig")
    submission[TARGET] = prediction_twin
    twin_path = ROOT / "5. submissions" / f"{NAME}_twin_rule.csv"
    submission.to_csv(twin_path, index=False, encoding="UTF-8-sig")

    artifact_dir = ROOT / "6. experiments/submissions" / NAME
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "test_proba_signature.npy", signature_proba)
    np.save(artifact_dir / "test_proba_raw_blend.npy", raw)
    print(f"saved: {plain_path.name}")
    print(f"saved: {twin_path.name} | twin rule changes: {(prediction_twin != prediction).sum()} / matched profiles: {twin_hits}")


if __name__ == "__main__":
    main()
