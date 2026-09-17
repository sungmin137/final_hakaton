"""approach11 v6: v5 실패 원인 보정판 — 제출 전 정직 CV로 먼저 검증만 함(train만 사용, test 안 읽음).

v5 대비 바뀐 것 (2026-09-11 회의 자료 보정안 a/b/c/d 반영):
  (a) 서브모델 학습에서 쌍둥이 행 제외 — GBMLGG/LGG 비쌍둥이만(286+52=338행)
  (b) 서브모델을 로지스틱회귀(강한 L2)로 단순화 — 52개 표본에 XGBoost는 과적합 위험 큼
  (c) 신뢰도 임계값 라우팅 — 서브모델 확률이 임계값 이상일 때만 메인 예측을 덮어씀
  (d) 검증 행이 train 쌍둥이 그룹과 프로필이 같으면(=쌍둥이 성격) 라우팅 자체를 안 함, 메인 예측 유지
  (e) 시드 5개 반복 CV로 분산 확인 — 이 스크립트의 목적

메인 모델은 v7과 동일 설정(features v4, mild_col: colsample_bytree 0.7).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from main import PARAM_SETS, TARGET, twin_groups, load_train, class_weights, FeatureMaker

PAIR = ["GBMLGG", "LGG"]
THRESHOLDS = [0.6, 0.7, 0.8]
SEEDS = [42, 7, 123, 2024, 999]

train = load_train()
le = LabelEncoder()
y_all = le.fit_transform(train[TARGET])
pair_idx = {c: le.transform([c])[0] for c in PAIR}

# 쌍둥이 여부(라벨이 섞인 그룹) — 전체 train 기준, 고정
genes = [c for c in train.columns if c not in ("ID", TARGET)]
gid = train.groupby(genes, sort=False).ngroup()
label_sets = train.groupby(gid)[TARGET].apply(lambda s: frozenset(s.unique()))
is_twin_all = gid.map(label_sets).apply(lambda s: len(s) > 1)

params = PARAM_SETS["mild_col"]


def run_seed(seed: int):
    groups = twin_groups(train)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_main = np.zeros((len(train), len(le.classes_)))
    oof_combined = {th: np.full(len(train), -1) for th in THRESHOLDS}
    n_routed = {th: 0 for th in THRESHOLDS}

    for tri, vai in skf.split(train, y_all, groups):
        tr_part, va_part = train.iloc[tri], train.iloc[vai]

        # 메인 모델 (v7과 동일)
        fm = FeatureMaker("v4").fit(tr_part)
        Xtr, Xva = fm.transform(tr_part), fm.transform(va_part)
        main_model = xgb.XGBClassifier(**params).fit(Xtr, y_all[tri])
        proba_va = main_model.predict_proba(Xva)
        oof_main[vai] = proba_va
        main_pred_va = proba_va.argmax(1)

        # 서브모델: 이 fold의 학습 데이터 중 GBMLGG/LGG "비쌍둥이"만
        tr_pair_mask = tr_part[TARGET].isin(PAIR) & ~is_twin_all.loc[tr_part.index]
        tr_pair = tr_part[tr_pair_mask]
        if len(tr_pair) < 10:      # 극단적으로 적으면 스킵(안전장치)
            for th in THRESHOLDS:
                oof_combined[th][vai] = main_pred_va
            continue
        y_pair = (tr_pair[TARGET] == "LGG").astype(int).to_numpy()

        fm2 = FeatureMaker("v4").fit(tr_pair)
        Xtr2 = fm2.transform(tr_pair)
        scaler = StandardScaler(with_mean=False)   # 희소 이진 컬럼 다수라 평균은 안 뺌
        Xtr2_s = scaler.fit_transform(Xtr2)
        w = class_weights(y_pair)
        spec = LogisticRegression(C=0.05, max_iter=2000, class_weight=None)
        spec.fit(Xtr2_s, y_pair, sample_weight=w)

        # 검증 fold 중 GBMLGG/LGG로 예측된 행만 서브모델 확률 계산
        route_candidates = np.isin(main_pred_va, [pair_idx["GBMLGG"], pair_idx["LGG"]])
        va_pair_idx_local = np.flatnonzero(route_candidates)
        if len(va_pair_idx_local) > 0:
            Xva2 = scaler.transform(fm2.transform(va_part.iloc[va_pair_idx_local]))
            spec_proba = spec.predict_proba(Xva2)  # col0=GBMLGG, col1=LGG (LogisticRegression 클래스 순서 확인 필요)
            spec_classes = spec.classes_  # [0,1] = [GBMLGG,LGG]
            p_lgg = spec_proba[:, list(spec_classes).index(1)]
        else:
            p_lgg = np.array([])

        va_is_twin = is_twin_all.loc[va_part.index].to_numpy()

        for th in THRESHOLDS:
            combined = main_pred_va.copy()
            for k, local_i in enumerate(va_pair_idx_local):
                if va_is_twin[local_i]:           # (d) 쌍둥이 성격이면 라우팅 안 함
                    continue
                conf = max(p_lgg[k], 1 - p_lgg[k])
                if conf >= th:                     # (c) 확신할 때만 덮어씀
                    combined[local_i] = pair_idx["LGG"] if p_lgg[k] >= 0.5 else pair_idx["GBMLGG"]
                    n_routed[th] += 1
            oof_combined[th][vai] = combined

    macro_main = f1_score(y_all, oof_main.argmax(1), average="macro")
    res = {"seed": seed, "main": round(macro_main, 4)}
    for th in THRESHOLDS:
        res[f"th{th}"] = round(f1_score(y_all, oof_combined[th], average="macro"), 4)
        res[f"routed{th}"] = n_routed[th]
    return res


if __name__ == "__main__":
    rows = []
    for sd in SEEDS:
        r = run_seed(sd)
        print(r)
        rows.append(r)
    df = pd.DataFrame(rows)
    print("\n=== 시드별 결과 ===")
    print(df.to_string(index=False))
    print("\n=== 평균 / 표준편차 ===")
    for col in ["main"] + [f"th{th}" for th in THRESHOLDS]:
        print(f"{col}: mean={df[col].mean():.4f} std={df[col].std():.4f}")
