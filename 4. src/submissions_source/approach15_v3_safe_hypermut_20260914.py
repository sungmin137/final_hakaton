"""approach15 v3 안전 결합: 21차 초과변이 답은 보존하고 나머지에만 치환 모델을 적용.

v2처럼 치환 모델 뒤에 초과변이 규칙을 적용하면 초과변이 변경이 31→114명으로
불어났다. 따라서 n_mut > 396 영역은 성민님 21차(v3+쌍둥이+초과변이)의 답을
그대로 쓰고, 그 외 환자만 approach15 v1의 답을 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4. src/common"))
from main import ID, TARGET, gene_columns, load_test, load_train  # noqa: E402
from postprocess.hypermut_rule import THRESH, apply_hypermut_rule  # noqa: E402
from postprocess.twin_rule import TwinRule  # noqa: E402

V1_NAME = "approach15_v1_substitution_signature_20260914"
BASE_NAME = "approach14_v3_20260913_2132"
NAME = "approach15_v3_signature_safe_hypermut_20260914"


def main() -> None:
    train = load_train()
    test = load_test()  # 최종 추론 단계에서만 사용
    genes = gene_columns(train)
    labels = LabelEncoder().fit(train[TARGET])
    classes = list(labels.classes_)

    # 21차를 재현: v3 → 쌍둥이 규칙 → 초과변이 규칙
    base_submission = pd.read_csv(ROOT / "5. submissions" / f"{BASE_NAME}.csv")
    assert (base_submission[ID] == test[ID]).all()
    base_twin, _ = TwinRule().fit(train).apply(test, base_submission[TARGET].to_numpy())
    base_raw = np.load(ROOT / "6. experiments/submissions" / BASE_NAME / "test_proba_raw_blend.npy")
    scale_map = json.loads((ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
    scales = np.array([scale_map[class_name] for class_name in classes])
    base_21, hypermut_changes, _ = apply_hypermut_rule(train, test, base_twin, base_raw * scales, classes)

    # approach15의 쌍둥이 규칙 적용 답. 초과변이 영역에서는 21차 답을 절대 덮지 않는다.
    signature_submission = pd.read_csv(ROOT / "5. submissions" / f"{V1_NAME}_twin_rule.csv")
    assert (signature_submission[ID] == test[ID]).all()
    burden = (test[genes] != "WT").sum(axis=1).to_numpy()
    final_prediction = signature_submission[TARGET].to_numpy().copy()
    final_prediction[burden > THRESH] = base_21[burden > THRESH]

    output = signature_submission.copy()
    output[TARGET] = final_prediction
    output_path = ROOT / "5. submissions" / f"{NAME}.csv"
    output.to_csv(output_path, index=False, encoding="UTF-8-sig")
    print(f"saved: {output_path.name}")
    print(f"high-burden patients kept from 21차: {(burden > THRESH).sum()} | 21차 hypermut changes: {len(hypermut_changes)}")
    print(f"approach15 changes retained below threshold: {((final_prediction != base_21) & (burden <= THRESH)).sum()}")


if __name__ == "__main__":
    main()
