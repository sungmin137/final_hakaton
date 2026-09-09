"""제출 파일 생성 스크립트 — 2026-09-09 1차 제출(submission.csv)을 만든 코드를 재현 가능하게 정리한 것.

흐름: train 전체 학습 → (여기서 처음) test.csv 로드 → 예측 → 형식 검증 → submissions/ 저장
      쌍둥이 규칙 적용본(_twin)도 함께 생성. 기본 submission.csv는 규칙 미적용본.

실행: PYTHONPATH=src python3 src/make_submission.py --features v2 [--params tuned] [--balanced]
"""
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from main import DATA, ID, PARAM_SETS, ROOT, TARGET, FeatureMaker, class_weights, load_test, load_train
from postprocess.twin_rule import TwinRule
from postprocess.class_scale import fit_class_scales
from approach3_class_feature_compare.approach3_class_feature_compare_v2 import EXPERT_COLS, blend


def validate(sub: pd.DataFrame, sample: pd.DataFrame, train_labels: set) -> None:
    assert list(sub.columns) == list(sample.columns), "컬럼 불일치"
    assert len(sub) == len(sample), "행 수 불일치"
    assert (sub[ID] == sample[ID]).all(), "ID 순서 불일치"
    assert sub[TARGET].notna().all(), "라벨 결측"
    assert set(sub[TARGET]) <= train_labels, "train에 없는 라벨"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="v2")
    ap.add_argument("--params", default="official", choices=list(PARAM_SETS))
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--tag", default=None, help="파일명 태그 (기본: <src 모듈>_<버전>[_class_scale]_<YYYYMMDD-HHMM>)")
    ap.add_argument("--approach3-blend", default=None, metavar="WD,WB[,WC]",
                    help="접근 3: driver/burden(/cat) 전문가를 train 전체로 학습해 test 확률을 로그 가중 블렌딩 (v2: 0.5,0.2 / v3: 0.5,0.2,0.4)")
    ap.add_argument("--class-scale", default=None, metavar="OOF_DIR",
                    help="정직 CV OOF 디렉토리(experiments/…_grp). 그 OOF와 train 라벨로 클래스 배율을 맞춰 test 확률에 곱함")
    a = ap.parse_args()
    params = PARAM_SETS[a.params]
    # 파일명 규칙: 어느 src 모듈(접근법)에서 나온 결과인지 + 버전 + 옵션 + 제작 시각
    MODULE = {"official": "main_official", "v1": "features_v1", "v2": "approach1_count_weight_v2",
              "v3": "features_v3", "v4": "approach2_knowledge_v4"}
    tag = MODULE[a.features]
    if a.approach3_blend:
        tag = "approach3_class_feature_compare_" + ("v3" if len(a.approach3_blend.split(",")) > 2 else "v2")
    tag += ("" if a.params == "official" else f"_{a.params}") + ("_balanced" if a.balanced else "")
    tag = a.tag or tag

    # 1. train 전체 학습
    t0 = time.time()
    train = load_train()
    le = LabelEncoder(); y = le.fit_transform(train[TARGET])
    fm = FeatureMaker(a.features).fit(train)
    model = xgb.XGBClassifier(**params).fit(fm.transform(train), y,
                                            sample_weight=class_weights(y) if a.balanced else None)
    print(f"[train] {train.shape} → 피처 {len(fm.columns)}개, 학습 {time.time()-t0:.0f}s")

    # 2. test 로드 (이 스크립트에서 test.csv를 읽는 유일한 지점)
    test = load_test()
    Xte = fm.transform(test)
    proba = model.predict_proba(Xte)
    if a.approach3_blend:                               # 접근 3 v2 — 전문가 블렌딩 (train 전체로 학습)
        ws = [float(v) for v in a.approach3_blend.split(",")]; w_d, w_b = ws[0], ws[1]; w_c = ws[2] if len(ws) > 2 else 0.0
        Xtr_full = fm.transform(train); experts = {}
        for name, sel in EXPERT_COLS.items():
            cols = [c for c in Xtr_full.columns if sel(c)]
            experts[name] = xgb.XGBClassifier(**params).fit(Xtr_full[cols], y).predict_proba(Xte[cols])
            print(f"[approach3] {name} expert: {len(cols)} cols")
        proba = blend(proba, experts["driver"], experts["burden"], w_d, w_b)
        if w_c > 0:                                     # v3: CatBoost 전문가 (요약 피처만)
            from approach3_class_feature_compare.approach3_class_feature_compare_v3 import CAT_COLS, CAT_PARAMS
            from catboost import CatBoostClassifier
            cols = [c for c in Xtr_full.columns if CAT_COLS(c)]
            p_cat = CatBoostClassifier(**CAT_PARAMS).fit(Xtr_full[cols], y).predict_proba(Xte[cols])
            print(f"[approach3] cat expert: {len(cols)} cols")
            z = np.log(proba + 1e-6) + w_c * np.log(p_cat + 1e-6); z = np.exp(z - z.max(1, keepdims=True)); proba = z / z.sum(1, keepdims=True)
        pass  # 이름은 위 MODULE 규칙에서 이미 approach3_…_v2/v3 로 결정
    if a.class_scale:                                   # Macro F1용 클래스 배율 — train OOF로만 결정 (docs/10)
        oof = np.load(ROOT / a.class_scale / "oof_proba.npy")
        scales = fit_class_scales(oof, y)
        print("[post] class scales:", {c: float(v) for c, v in zip(le.classes_, scales) if v != 1.0})
        proba = proba * scales
        tag += "_class_scale"
    pred = le.inverse_transform(proba.argmax(1))
    sample = pd.read_csv(DATA / "sample_submission.csv")
    labels = set(train[TARGET])

    # 3. 저장 — 기본본
    out_dir = ROOT / "submissions"; out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    sub = sample.copy(); sub[TARGET] = pred
    validate(sub, sample, labels)
    sub.to_csv(out_dir / f"{tag}_{stamp}.csv", index=False, encoding="UTF-8-sig")
    sub.to_csv(out_dir / "submission.csv", index=False, encoding="UTF-8-sig")

    # 4. 저장 — 쌍둥이 규칙 적용본 (docs/07 참고, 팀 판단 후 선택)
    pred_tw, n_hit = TwinRule().fit(train).apply(test, pred)
    sub_tw = sample.copy(); sub_tw[TARGET] = pred_tw
    validate(sub_tw, sample, labels)
    sub_tw.to_csv(out_dir / f"{tag}_twin_rule_{stamp}.csv", index=False, encoding="UTF-8-sig")

    exp = ROOT / "experiments" / "submissions" / f"{tag}_{stamp}"; exp.mkdir(parents=True, exist_ok=True)
    np.save(exp / "test_proba.npy", proba)

    # 5. 요약
    print(f"[test] {test.shape}, train과 완전 동일 행 {n_hit} ({n_hit/len(test):.1%}), 규칙으로 바뀐 예측 {(pred != pred_tw).sum()}행")
    print(f"[test] 예측 확신도 중앙값 {np.median(proba.max(1)):.2f}")
    dist = pd.DataFrame({"train": train[TARGET].value_counts(normalize=True),
                         "pred": pd.Series(pred).value_counts(normalize=True)}).fillna(0)
    dist["diff"] = dist.pred - dist.train
    print("[test] train 비율과 가장 다른 예측 클래스:\n", dist.sort_values("diff", key=abs, ascending=False).head(5).round(3).to_string())
    print(f"saved: submissions/submission.csv, {tag}_{stamp}.csv, {tag}_twin_rule_{stamp}.csv")


if __name__ == "__main__":
    main()
