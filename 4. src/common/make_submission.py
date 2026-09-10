"""제출 파일 생성 스크립트 — 2026-09-09 1차 제출(submission.csv)을 만든 코드를 재현 가능하게 정리한 것.

흐름: train 전체 학습 → (여기서 처음) test.csv 로드 → 예측 → 형식 검증 → 5. submissions/ 저장
      쌍둥이 규칙 적용본(_twin)도 함께 생성. 기본 submission.csv는 규칙 미적용본.

실행: PYTHONPATH="4. src/common" python3 "4. src/common/make_submission.py" --features v4 --class-scale … --tag approach2_v2_YYYYMMDD_HHMM
      (보통은 4. src/approachN_vK_일자_시간.py 진입 스크립트가 이 main()을 호출한다)
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
    ap.add_argument("--drop-genes", default=None, metavar="FILE", help="제거할 유전자 열 목록 파일 (결측 분석 결과)")
    ap.add_argument("--fold-bag", action="store_true",
                    help="정직 CV의 fold별 학습 모델(80%% 데이터) 5개 + 전체 모델 1개의 확률 평균 (분산 감소, 피처 변경 없음)")
    ap.add_argument("--bag-seeds", default="42", help="fold 분할 시드 목록 (예: 42,7,123 → 15개 fold 모델 + 전체 모델)")
    ap.add_argument("--scale-avg", action="store_true", help="클래스 배율을 OOF 절반 분할 여러 번에서 맞춰 기하평균 (배율 안정화)")
    ap.add_argument("--class-scale-file", default=None, metavar="JSON",
                    help="클래스 배율을 파일에서 그대로 읽어 적용 (예: 3차 복원 배율 6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json)")
    ap.add_argument("--drop-cols-prefix", default=None, help="이 접두사로 시작하는 피처 열 제거 (쉼표 구분, 예: cw_variant)")
    ap.add_argument("--class-scale", default=None, metavar="OOF_DIR",
                    help="정직 CV OOF 디렉토리(6. experiments/…_grp). 그 OOF와 train 라벨로 클래스 배율을 맞춰 test 확률에 곱함")
    a = ap.parse_args()
    params = PARAM_SETS[a.params]
    # 파일명 규칙: 어느 src 모듈(접근법)에서 나온 결과인지 + 버전 + 옵션 + 제작 시각
    MODULE = {"official": "main_official", "v1": "features_v1", "v2": "approach1_count_weight_v2",
              "v3": "features_v3", "v4": "approach2_knowledge_v4"}
    tag = MODULE.get(a.features, f"features_{a.features}")
    if a.approach3_blend:
        tag = "approach3_class_feature_compare_" + ("v3" if len(a.approach3_blend.split(",")) > 2 else "v2")
    tag += ("" if a.params == "official" else f"_{a.params}") + ("_balanced" if a.balanced else "")
    fixed_name = bool(a.tag)          # 진입 스크립트가 이름을 지정한 경우: 2. team/file_rules.md 규칙 그대로
    tag = a.tag or tag

    # 1. train 전체 학습
    t0 = time.time()
    train = load_train()
    le = LabelEncoder(); y = le.fit_transform(train[TARGET])
    drop = [l.strip() for l in open(ROOT / a.drop_genes) if l.strip()] if a.drop_genes else None
    fm = FeatureMaker(a.features, drop).fit(train)
    model = xgb.XGBClassifier(**params).fit(fm.transform(train), y,
                                            sample_weight=class_weights(y) if a.balanced else None)
    print(f"[train] {train.shape} → 피처 {len(fm.columns)}개, 학습 {time.time()-t0:.0f}s")

    # 2. test 로드 (이 스크립트에서 test.csv를 읽는 유일한 지점)
    test = load_test()
    Xte = fm.transform(test)
    proba = model.predict_proba(Xte)
    if a.fold_bag:                                      # fold 모델 평균 (train만 사용, 정직 CV와 같은 fold 분할)
        from sklearn.model_selection import StratifiedGroupKFold
        from main import twin_groups, SEED
        probs = [proba]
        for sd in [int(v) for v in a.bag_seeds.split(",")]:
            for k, (tri, _) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(train, y, twin_groups(train))):
                fm_k = FeatureMaker(a.features).fit(train.iloc[tri])
                m_k = xgb.XGBClassifier(**params).fit(fm_k.transform(train.iloc[tri]), y[tri],
                                                      sample_weight=class_weights(y[tri]) if a.balanced else None)
                probs.append(m_k.predict_proba(fm_k.transform(test))); print(f"[bag] seed{sd} fold{k} 모델 완료 ({time.time()-t0:.0f}s)")
        proba = np.mean(probs, 0)
        if not fixed_name: tag += "_foldbag"
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
    if a.class_scale:                                   # Macro F1용 클래스 배율 — train OOF로만 결정 (3. docs/10)
        oof = np.load(ROOT / a.class_scale / "oof_proba.npy")
        if a.scale_avg:                                 # 절반 분할 6회에서 맞춘 배율의 기하평균 → 과적합 완화
            rng = np.random.RandomState(0); logs = []
            for _ in range(6):
                idx = rng.permutation(len(y)); half = idx[: len(y) // 2]
                logs.append(np.log(fit_class_scales(oof[half], y[half])))
            scales = np.exp(np.mean(logs, 0))
        else:
            scales = fit_class_scales(oof, y)
        print("[post] class scales:", {c: float(v) for c, v in zip(le.classes_, scales) if v != 1.0})
        proba = proba * scales
        if not fixed_name:
            tag += "_class_scale"
    pred = le.inverse_transform(proba.argmax(1))
    sample = pd.read_csv(DATA / "sample_submission.csv")
    labels = set(train[TARGET])

    # 3. 저장 — 기본본
    out_dir = ROOT / "5. submissions"; out_dir.mkdir(exist_ok=True)
    stamp = "" if fixed_name else "_" + datetime.now().strftime("%Y%m%d_%H%M")
    sub = sample.copy(); sub[TARGET] = pred
    validate(sub, sample, labels)
    sub.to_csv(out_dir / f"{tag}{stamp}.csv", index=False, encoding="UTF-8-sig")
    sub.to_csv(out_dir / "submission.csv", index=False, encoding="UTF-8-sig")

    # 4. 저장 — 쌍둥이 규칙 적용본 (3. docs/04_duplicate_twins.md 참고, 팀 판단 후 선택)
    pred_tw, n_hit = TwinRule().fit(train).apply(test, pred)
    sub_tw = sample.copy(); sub_tw[TARGET] = pred_tw
    validate(sub_tw, sample, labels)
    sub_tw.to_csv(out_dir / f"{tag}{stamp}_twin_rule.csv", index=False, encoding="UTF-8-sig")

    exp = ROOT / "6. experiments" / "submissions" / f"{tag}{stamp}"; exp.mkdir(parents=True, exist_ok=True)
    np.save(exp / "test_proba.npy", proba)

    # 5. 요약
    print(f"[test] {test.shape}, train과 완전 동일 행 {n_hit} ({n_hit/len(test):.1%}), 규칙으로 바뀐 예측 {(pred != pred_tw).sum()}행")
    print(f"[test] 예측 확신도 중앙값 {np.median(proba.max(1)):.2f}")
    dist = pd.DataFrame({"train": train[TARGET].value_counts(normalize=True),
                         "pred": pd.Series(pred).value_counts(normalize=True)}).fillna(0)
    dist["diff"] = dist.pred - dist.train
    print("[test] train 비율과 가장 다른 예측 클래스:\n", dist.sort_values("diff", key=abs, ascending=False).head(5).round(3).to_string())
    print(f"saved: 5. submissions/submission.csv, {tag}{stamp}.csv, {tag}{stamp}_twin_rule.csv")


if __name__ == "__main__":
    main()
