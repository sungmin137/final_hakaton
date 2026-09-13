"""메인 파이프라인 — 공식 베이스라인([Baseline]_XGB) 구조를 유지한다.

  1. Load Data        : train.csv만 로드
  2. Preprocessing    : 라벨 인코딩 + 피처 생성 (official | v1 | ...)
  3. Model Train      : XGBoost, Stratified 5-Fold로 OOF Macro F1 / Accuracy 측정
  4. Inference        : 전체 train 재학습 → 이 단계에서만 test.csv 로드 → 예측
  5. Submission       : 5. submissions/{날짜}_{피처}_xgb.csv

실행 예)
  PYTHONPATH="4. src/common" python3 "4. src/common/main.py" --features v4 --cv --group-twins
  (제출은 4. src/approachN_vK_일자_시간.py 진입 스크립트로)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import re

from features.features import ID, TARGET, InsightFeatures, build_features, gene_columns
from approach1_count_weight.count_weights import CountWeightFeatures
from postprocess.twin_rule import TwinRule
from approach2_knowledge.knowledge_features import KnowledgeFeatures
from approach4_literature.literature_features import LiteratureFeatures
from approach14_cw_plus.cw_plus import CWPlusFeatures
from approach7_preprocess.preprocess_features import PreprocessFeatures

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "1. info" / "data"          # 원본 데이터 (git 제외)
SEED = 42

# 공식 베이스라인과 동일한 하이퍼파라미터 (tree_method만 hist로 고정: 속도)
XGB_PARAMS = dict(
    n_estimators=100, learning_rate=0.1, max_depth=6, random_state=SEED,
    eval_metric="mlogloss", tree_method="hist", n_jobs=8,
)
# 조정안: 트리 수↑ 학습률↓, 피처 서브샘플링(희소 4,000+ 컬럼), 정규화
XGB_TUNED = dict(
    n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.3,
    min_child_weight=2, reg_lambda=2.0, random_state=SEED, eval_metric="mlogloss",
    tree_method="hist", n_jobs=8,
)
# mild 변형 (2026-09-10): 3차 구성에서 한 요소만 살짝 바꿈
XGB_MILD_COL = {**XGB_PARAMS, "colsample_bytree": 0.7}
XGB_MILD_REG = {**XGB_PARAMS, "min_child_weight": 3, "reg_lambda": 3.0}
XGB_MILD_COL5 = {**XGB_PARAMS, "colsample_bytree": 0.5}
XGB_MILD_COL_REG = {**XGB_PARAMS, "colsample_bytree": 0.7, "min_child_weight": 3, "reg_lambda": 3.0}
XGB_MILD_COL5_REG = {**XGB_PARAMS, "colsample_bytree": 0.5, "min_child_weight": 3, "reg_lambda": 3.0}
PARAM_SETS = {"official": XGB_PARAMS, "tuned": XGB_TUNED, "mild_col": XGB_MILD_COL, "mild_reg": XGB_MILD_REG,
              "mild_col5": XGB_MILD_COL5, "mild_col_reg": XGB_MILD_COL_REG, "mild_col5_reg": XGB_MILD_COL5_REG}

# 앙상블용 다른 부스팅 모델
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, feature_fraction=0.3,
                   bagging_fraction=0.8, bagging_freq=1, min_child_samples=10, reg_lambda=1.0,
                   random_state=SEED, n_jobs=8, verbose=-1)
CAT_PARAMS = dict(iterations=300, learning_rate=0.1, depth=6, rsm=0.3, random_seed=SEED,
                  loss_function="MultiClass", thread_count=8, verbose=0)


def make_model(name: str, params: dict):
    if name == "xgb":
        return xgb.XGBClassifier(**params)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**LGBM_PARAMS)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**CAT_PARAMS)
    if name == "mlp":                                  # GPU(MPS/CUDA) 신경망
        from models.nn_model import MLPClassifier
        return MLPClassifier()
    raise ValueError(name)


def class_weights(y: np.ndarray) -> np.ndarray:
    """클래스 빈도의 역수(제곱근 완화)로 샘플 가중치. Macro F1 대응."""
    cnt = np.bincount(y)
    w = (cnt.max() / cnt) ** 0.5
    return w[y]


# ---------------------------------------------------------------- 1. Load
def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train.csv")


def load_test() -> pd.DataFrame:
    """⚠️ Inference 단계 외에서는 호출 금지 (프로젝트 규칙).
    test.csv에는 train에 없던 결측 셀이 있다 → WT(변이 없음)로 간주."""
    return pd.read_csv(DATA / "test.csv").fillna("WT")


def _safe_names(cols) -> list[str]:
    """LightGBM 등이 거부하는 특수문자(: * > 등)를 _로 치환. 중복 시 번호 부여."""
    out, seen = [], {}
    for c in cols:
        n = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
        if n in seen:
            seen[n] += 1; n = f"{n}__{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return out


# ---------------------------------------------------------------- 2. Preprocessing
class FeatureMaker:
    """fit(train 부분) → transform(valid/test). fit 통계는 train 부분에서만 나온다."""

    def __init__(self, kind: str, drop_genes: list[str] | None = None):
        self.kind = kind
        self.drop_genes = set(drop_genes or [])      # 결측 분석으로 제거하기로 한 유전자 열 (docs/06)
        self.genes: list[str] = []
        self.columns: list[str] = []
        self._enc: OrdinalEncoder | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureMaker":
        if self.drop_genes:
            df = df.drop(columns=[g for g in self.drop_genes if g in df.columns])
        self.genes = gene_columns(df)
        if self.kind == "official":
            self._enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self._enc.fit(df[self.genes])
            self.columns = list(self.genes)
        elif self.kind in ("v1", "v2", "v3", "v4", "v5", "v6", "a2", "a4", "a7", "v4p"):
            X = build_features(df, self.genes)
            # 전부 WT인 유전자 컬럼 제거 (train 부분 기준)
            self.columns = [c for c in X.columns if not (c.startswith("g_") and X[c].sum() == 0)]
            if self.kind in ("v2", "v3", "v4", "v5", "v6", "v4p"):  # 접근 1: 클래스별 개수 가중치 점수 피처
                self._cw = CountWeightFeatures().fit(df)
                self.columns += list(self._cw.transform(df).columns)
            if self.kind in ("v3", "v4", "v5", "v6", "a2", "v4p"):  # 인사이트 피처: hotspot 위치, LoF 유전자, 조합, 특수 그룹
                self._ins = InsightFeatures().fit(df)
                self.columns += list(self._ins.transform(df).columns)
            if self.kind in ("v4", "v5", "v6", "a2", "v4p"):  # 지식 피처: BLOSUM62·아미노산 특성 변화 (3. docs/10 해석)
                self._kf = KnowledgeFeatures().fit(df)
                self.columns += list(self._kf.transform(df).columns)
            if self.kind == "v4p":                     # 접근 14: cw_ 점수 고도화 (유형 분리·위치 구간·군집 상대)
                self._cwp = CWPlusFeatures().fit(df)
                self.columns += list(self._cwp.transform(df).columns)
            if self.kind in ("v5", "a4"):              # 접근 4: 문헌 driver·경로·역할 피처 (a4 = 기본 피처 + 문헌 피처만, 단독 평가)
                self._lit = LiteratureFeatures().fit(df)
                self.columns += list(self._lit.transform(df).columns)
            if self.kind in ("v6", "a7"):              # 접근 7: 전처리 확장 (유형 분리 이진화, 도메인 구간, 부담 정규화·희귀 변이)
                self._pp = PreprocessFeatures().fit(df)
                self.columns += list(self._pp.transform(df).columns)
        elif self.kind == "a8":                        # 접근 8: v5 피처를 중요도 상위 500개로 압축 (2. team/approaches/approach8.md)
            K = 500
            fm5 = FeatureMaker("v5").fit(df)
            X5 = fm5.transform(df)
            y5 = LabelEncoder().fit_transform(df[TARGET])
            ranker = xgb.XGBClassifier(**XGB_PARAMS).fit(X5, y5)
            ranked = pd.Series(ranker.feature_importances_, index=X5.columns).sort_values(ascending=False)
            self._fm5 = fm5
            self.columns = list(ranked.head(K).index)
        else:
            raise ValueError(f"unknown features: {self.kind}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.drop_genes:
            df = df.drop(columns=[g for g in self.drop_genes if g in df.columns])
        if self.kind == "official":
            return pd.DataFrame(self._enc.transform(df[self.genes]), columns=self.genes, index=df.index)
        if self.kind == "a8":
            X = self._fm5.transform(df)
            return X.reindex(columns=self.columns, fill_value=0)
        X = build_features(df, self.genes)
        if self.kind in ("v2", "v3", "v4", "v5", "v6", "v4p"):
            X = pd.concat([X, self._cw.transform(df)], axis=1)
        if self.kind in ("v3", "v4", "v5", "v6", "a2", "v4p"):
            X = pd.concat([X, self._ins.transform(df)], axis=1)
        if self.kind in ("v4", "v5", "v6", "a2", "v4p"):
            X = pd.concat([X, self._kf.transform(df)], axis=1)
        if self.kind == "v4p":
            X = pd.concat([X, self._cwp.transform(df)], axis=1)
        if self.kind in ("v5", "a4"):
            X = pd.concat([X, self._lit.transform(df)], axis=1)
        if self.kind in ("v6", "a7"):
            X = pd.concat([X, self._pp.transform(df)], axis=1)
        X = X.reindex(columns=self.columns, fill_value=0)
        X.columns = _safe_names(X.columns)
        return X


# ---------------------------------------------------------------- 3. Model Train (CV)
def twin_groups(train: pd.DataFrame) -> np.ndarray:
    """완전 동일 프로필(쌍둥이)을 같은 그룹으로. 3. docs/07 참고."""
    from postprocess.twin_rule import _hash_rows
    genes = gene_columns(train)
    return pd.factorize(_hash_rows(train, genes))[0]


def cross_validate(train: pd.DataFrame, kind: str, params: dict, n_splits: int = 5,
                   out_dir: Path | None = None, group_twins: bool = False, balanced: bool = False,
                   model_name: str = "xgb", drop_genes: list[str] | None = None) -> dict:
    le = LabelEncoder()
    y = le.fit_transform(train[TARGET])
    if group_twins:   # 쌍둥이를 같은 fold에 묶어 중복 학습 효과를 제거한 '정직한' CV
        splits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(train, y, twin_groups(train))
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(train, y)
    oof = np.zeros((len(train), len(le.classes_)))
    folds = []
    t0 = time.time()
    for k, (tri, vai) in enumerate(splits):
        fm = FeatureMaker(kind, drop_genes).fit(train.iloc[tri])
        Xtr, Xva = fm.transform(train.iloc[tri]), fm.transform(train.iloc[vai])
        model = make_model(model_name, params)
        model.fit(Xtr, y[tri], sample_weight=class_weights(y[tri]) if balanced else None)
        oof[vai] = model.predict_proba(Xva)
        pred = oof[vai].argmax(1)
        f1, acc = f1_score(y[vai], pred, average="macro"), accuracy_score(y[vai], pred)
        folds.append(dict(fold=k, macro_f1=round(f1, 4), acc=round(acc, 4)))
        print(f"fold{k} macroF1={f1:.4f} acc={acc:.4f} ({time.time()-t0:.0f}s)")

    pred = oof.argmax(1)
    res = dict(
        features=kind, model=model_name, params=params if model_name == "xgb" else None, folds=folds, group_twins=group_twins, balanced=balanced,
        oof_macro_f1=round(f1_score(y, pred, average="macro"), 4),
        oof_acc=round(accuracy_score(y, pred), 4),
        per_class_f1=dict(zip(le.classes_, f1_score(y, pred, average=None).round(4).tolist())),
        elapsed_s=round(time.time() - t0, 1),
    )
    print(f"OOF macroF1={res['oof_macro_f1']} acc={res['oof_acc']}")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        np.save(out_dir / "oof_proba.npy", oof)
        pd.DataFrame(confusion_matrix(y, pred), index=le.classes_, columns=le.classes_
                     ).to_csv(out_dir / "confusion_matrix.csv")
    return res


# ---------------------------------------------------------------- 4~5. Inference & Submission
def fit_full_and_submit(train: pd.DataFrame, kind: str, params: dict, tag: str,
                        twin_rule: bool = False, balanced: bool = False) -> Path:
    le = LabelEncoder()
    y = le.fit_transform(train[TARGET])
    fm = FeatureMaker(kind).fit(train)
    model = xgb.XGBClassifier(**params).fit(fm.transform(train), y,
                                            sample_weight=class_weights(y) if balanced else None)

    test = load_test()                      # ← test.csv는 여기서 처음 읽힌다
    pred = le.inverse_transform(model.predict(fm.transform(test)))
    if twin_rule:                           # 3. docs/04_duplicate_twins.md — 선택 적용
        pred, n_hit = TwinRule().fit(train).apply(test, pred)
        print(f"twin rule 적용: test {len(test)}행 중 {n_hit}행이 train 행과 완전 동일")
        tag += "_twin"
    sub = pd.read_csv(DATA / "sample_submission.csv")
    assert (sub[ID] == test[ID]).all(), "ID 순서 불일치"
    sub[TARGET] = pred

    out = ROOT / "5. submissions" / f"{tag}.csv"
    sub.to_csv(out, index=False, encoding="UTF-8-sig")
    print("saved", out, "\n", sub[TARGET].value_counts().head(8).to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="v1", choices=["official", "v1", "v2", "v3", "v4", "v5", "v6", "a2", "a4", "a7", "a8", "v4p"])
    ap.add_argument("--cv", action="store_true", help="Stratified 5-Fold 평가")
    ap.add_argument("--submit", action="store_true", help="전체 학습 후 test 추론 및 제출 파일 생성")
    ap.add_argument("--twin-rule", action="store_true", help="추론 시 쌍둥이 규칙 적용 (3. docs/07 참고, 기본 꺼짐)")
    ap.add_argument("--group-twins", action="store_true", help="CV에서 쌍둥이를 같은 fold에 묶음 (정직한 CV)")
    ap.add_argument("--params", default="official", choices=list(PARAM_SETS), help="XGB 파라미터 세트")
    ap.add_argument("--balanced", action="store_true", help="클래스 빈도 역수 샘플 가중치")
    ap.add_argument("--model", default="xgb", choices=["xgb", "lgbm", "cat", "mlp"], help="부스팅 모델 (CV 전용)")
    ap.add_argument("--drop-genes", default=None, metavar="FILE", help="제거할 유전자 열 목록 파일 (한 줄에 하나). 결측 분석 결과 적용")
    a = ap.parse_args()

    params = PARAM_SETS[a.params]
    tag = (f"{date.today().isoformat()}_{a.features}_{a.model}" + ("" if a.params == "official" else f"_{a.params}")
           + ("_bal" if a.balanced else "") + ("_grp" if a.group_twins else ""))
    train = load_train()
    print("train", train.shape, "| features:", a.features)
    if a.cv:
        drop = [l.strip() for l in open(ROOT / a.drop_genes) if l.strip()] if a.drop_genes else None
        tag += "_drop" if drop else ""
        cross_validate(train, a.features, params, out_dir=ROOT / "6. experiments" / tag, drop_genes=drop,
                       group_twins=a.group_twins, balanced=a.balanced, model_name=a.model)
    if a.submit:
        fit_full_and_submit(train, a.features, params, tag, twin_rule=a.twin_rule, balanced=a.balanced)


if __name__ == "__main__":
    main()
