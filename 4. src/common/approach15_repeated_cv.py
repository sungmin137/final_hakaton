"""접근15의 독립 반복 정직 CV.

16차(v3)를 매 seed마다 v4 XGB + v4p XGB로 새로 학습하고, 같은 fold에서
아미노산 치환 스펙트럼 LR도 새로 학습한다. test.csv는 읽지 않는다.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = Path("/Users/mac/final_hakaton/final_hakaton")
sys.path.insert(0, str(REPO / "4. src/common"))
from main import FeatureMaker, PARAM_SETS, TARGET, gene_columns, twin_groups  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
EPS = 1e-12
SEEDS = (137, 911)  # 기존 seed=42 OOF와 완전히 다른 두 번의 fold 배치


def signature(values: np.ndarray) -> np.ndarray:
    out = np.zeros((len(values), 404), dtype=np.float32)
    for i, row in enumerate(values):
        for raw in row[row != "WT"]:
            for token in raw.split(" "):
                if token.endswith("*") or "fs" in token:
                    out[i, 400] += 1; continue
                m = MISSENSE.match(token)
                if m is None or m.group(1) not in AA_INDEX or m.group(3) not in AA_INDEX:
                    out[i, 403] += 1; continue
                before, after = m.group(1), m.group(3)
                if before == after: out[i, 401] += 1
                else: out[i, AA_INDEX[before] * 20 + AA_INDEX[after]] += 1
    out[:, 402] = out[:, :400].sum(1)
    out[:, :400] /= np.maximum(out[:, :400].sum(1, keepdims=True), 1)
    return out


def score(y, pred, burden):
    return {
        "all": float(f1_score(y, pred, average="macro")),
        "11_30": float(f1_score(y[(burden >= 11) & (burden <= 30)], pred[(burden >= 11) & (burden <= 30)], average="macro")),
        "31_100": float(f1_score(y[(burden >= 31) & (burden <= 100)], pred[(burden >= 31) & (burden <= 100)], average="macro")),
    }


def main() -> None:
    train = pd.read_csv(REPO / "1. info/data/train.csv")
    le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET])
    values = train[gene_columns(train)].to_numpy(); Xsig = signature(values); burden = (values != "WT").sum(1)
    groups = twin_groups(train)
    scale_map = json.loads((REPO / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json").read_text())
    scales = np.array([scale_map[c] for c in le.classes_])
    results = []

    for seed in SEEDS:
        t0 = time.time(); p4 = np.zeros((len(train), len(le.classes_))); p4p = np.zeros_like(p4); psig = np.zeros_like(p4)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(train, y, groups), 1):
            # 두 v3 구성 모델을 이 seed의 학습 fold에서 처음부터 다시 만든다.
            fm4 = FeatureMaker("v4").fit(train.iloc[tr])
            model4 = xgb.XGBClassifier(**PARAM_SETS["mild_col"]).fit(fm4.transform(train.iloc[tr]), y[tr])
            p4[va] = model4.predict_proba(fm4.transform(train.iloc[va]))
            fm4p = FeatureMaker("v4p").fit(train.iloc[tr])
            model4p = xgb.XGBClassifier(**PARAM_SETS["mild_col"]).fit(fm4p.transform(train.iloc[tr]), y[tr])
            p4p[va] = model4p.predict_proba(fm4p.transform(train.iloc[va]))
            sig_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.12, class_weight="balanced", max_iter=2000, solver="lbfgs"))
            sig_model.fit(Xsig[tr], y[tr]); psig[va] = sig_model.predict_proba(Xsig[va])
            print(f"seed={seed} fold={fold}/5 elapsed={time.time()-t0:.0f}s", flush=True)

        v3_raw = softmax(0.5 * np.log(np.maximum(p4, EPS)) + 0.5 * np.log(np.maximum(p4p, EPS)), axis=1)
        base = (v3_raw * scales).argmax(1)
        mixed_raw = softmax(0.8 * np.log(np.maximum(v3_raw, EPS)) + 0.2 * np.log(np.maximum(psig, EPS)), axis=1)
        mixed = (mixed_raw * scales).argmax(1)
        # 제출 후보와 같은 안전 장치: train 최대 ACC 기준을 넘는 행은 기존 답을 유지.
        safe = mixed.copy(); safe[burden > 396] = base[burden > 396]
        b, m, s = score(y, base, burden), score(y, mixed, burden), score(y, safe, burden)
        row = {"seed": seed, "base": b, "mixed": m, "safe": s, "mixed_delta": {k: m[k]-b[k] for k in b}, "safe_delta": {k: s[k]-b[k] for k in b}, "elapsed_s": round(time.time()-t0, 1)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out = REPO / "6. experiments/2026-09-14_approach15_repeated_cv"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("SUMMARY", json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
