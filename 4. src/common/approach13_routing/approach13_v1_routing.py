"""접근 13 v1 — 개선판 라우팅 (김혜성 제안 (a)~(e) + Rim457 재판기 틀). train만 사용, 메인 OOF는 9차 설정 재사용.

메인: v4 피처 + XGB colsample 0.7 (9차, LB 0.4377) 의 정직 CV OOF × 3차 복원 배율 → top1/top2/확률차.
재판(라우팅): top1·top2가 미리 정한 혼동 쌍이고 확률차 ≤ gap 일 때, 그 쌍만으로 학습한 작은 로지스틱 회귀(강한 L2)가 답을 다시 고른다.
  (a) 서브모델 학습에서 완전 동일 프로필 쌍둥이 행 제외 (쌍둥이 쌍 KIRC/KIPAN·GBMLGG/LGG에 특히 중요)
  (b) 표본 수십~수백 개에 맞는 단순 모델: 유전자 변이 유무 이진 + 변이 수, L2 C=0.1
  (c) 서브모델 확신 ≥ conf 일 때만 덮어씀
  (d) (추론 시) train 쌍둥이가 있는 test 행은 라우팅 제외 — 쌍둥이 규칙이 담당
  (e) 임계값(gap, conf)은 fold 밖 4개 fold로 고르는 교차선택 + 최종 안정성 표기
실행: PYTHONPATH="4. src/common" python3 "4. src/common/approach13_routing/approach13_v1_routing.py" [--submit TAG]
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from main import ROOT, SEED, DATA, TARGET, ID, load_train, load_test, twin_groups, gene_columns

MAIN_OOF = ROOT / "6. experiments/2026-09-11_v4_xgb_mild_col_grp/oof_proba.npy"      # 9차 설정 OOF
MAIN_TEST = ROOT / "6. experiments/submissions/approach12_v4_20260910_1651/test_proba.npy"  # 9차 test 확률 (배율 곱해진 상태)
SCALES = ROOT / "6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"
OUT = ROOT / "6. experiments/approach13_routing_v1"
PAIRS_TWIN = [("GBMLGG", "LGG"), ("KIRC", "KIPAN")]
PAIRS_REF = [("BRCA", "OV"), ("BRCA", "PRAD"), ("HNSC", "STES"), ("LIHC", "STES"), ("LUAD", "STES"), ("LUSC", "STES"), ("OV", "PAAD"), ("PRAD", "THCA"), ("BRCA", "SARC")]
GAPS = [0.10, 0.20, 0.30, 1.0]; CONFS = [0.5, 0.6, 0.7, 0.8]


def binary_X(df, genes):
    B = (df[genes].to_numpy() != "WT").astype(np.float32)
    n = B.sum(1, keepdims=True)
    return np.hstack([B, np.log1p(n)])


def hashes(df, genes):
    return np.array([hashlib.md5("|".join(r).encode()).hexdigest() for r in df[genes].to_numpy()])


def fit_pair(Xtr, ytr_lab, a, b, twin_mask):
    m = np.isin(ytr_lab, [a, b]) & ~twin_mask
    if m.sum() < 20 or len(set(ytr_lab[m])) < 2: return None
    yb = (ytr_lab[m] == b).astype(int)
    return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(Xtr[m], yb)


def route(P_scaled, classes, models, Xva, gap, conf, pairs):
    ci = {c: i for i, c in enumerate(classes)}; order = np.argsort(-P_scaled, 1); top1, top2 = order[:, 0], order[:, 1]
    Pn = P_scaled / P_scaled.sum(1, keepdims=True); pred = top1.copy(); n_changed = 0
    for a, b in pairs:
        ia, ib = ci[a], ci[b]; mdl = models.get((a, b))
        if mdl is None: continue
        m = (((top1 == ia) & (top2 == ib)) | ((top1 == ib) & (top2 == ia))) & ((Pn[np.arange(len(Pn)), top1] - Pn[np.arange(len(Pn)), top2]) <= gap)
        if not m.any(): continue
        pb = mdl.predict_proba(Xva[m])[:, 1]; new = np.where(pb >= conf, ib, np.where(pb <= 1 - conf, ia, pred[m]))
        n_changed += int((new != pred[m]).sum()); pred[m] = new
    return pred, n_changed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", default=None, help="test 적용 후 이 태그로 제출 파일 생성"); ap.add_argument("--gap", type=float, default=None); ap.add_argument("--conf", type=float, default=None); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train(); genes = gene_columns(train); le = LabelEncoder(); y = le.fit_transform(train[TARGET]); classes = list(le.classes_); ylab = train[TARGET].to_numpy()
    s3 = np.array([json.load(open(SCALES))[c] for c in classes]); oof = np.load(MAIN_OOF); P = oof * s3
    X = binary_X(train, genes); h = hashes(train, genes); nm = (train[genes] != "WT").sum(1).to_numpy(); hc = Counter(h[nm > 0]); twin = np.array([nm[i] > 0 and hc[h[i]] >= 2 for i in range(len(train))])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, y, twin_groups(train)))
    base = P.argmax(1); f_base = f1_score(y, base, average="macro"); print(f"기준(9차 설정 OOF × 배율): {f_base:.4f}")
    pairs_all = PAIRS_TWIN + PAIRS_REF
    # 모든 (gap, conf) 조합의 OOF 예측을 fold별로 생성
    results = {}; preds = {}
    for gap, conf in itertools.product(GAPS, CONFS):
        pred = base.copy(); changed = 0
        for k, (tri, vai) in enumerate(folds):
            models = {(a_, b_): fit_pair(X[tri], ylab[tri], a_, b_, twin[tri]) for a_, b_ in pairs_all}
            p, c = route(P[vai], classes, models, X[vai], gap, conf, pairs_all); pred[vai] = p; changed += c
        f = f1_score(y, pred, average="macro"); results[(gap, conf)] = (f, changed); preds[(gap, conf)] = pred
        print(f"gap={gap:<4} conf={conf}: F1 {f:.4f} ({f-f_base:+.4f}), 바뀐 행 {changed}")
    # (e) 교차선택: 각 fold의 최적 (gap,conf)를 나머지 4개 fold에서 고름
    fold_id = np.zeros(len(y), int)
    for k, (_, vai) in enumerate(folds): fold_id[vai] = k
    sel_pred = base.copy(); chosen = []
    for k in range(5):
        others = fold_id != k
        best = max(results, key=lambda gc: f1_score(y[others], preds[gc][others], average="macro")); chosen.append(best)
        sel_pred[fold_id == k] = preds[best][fold_id == k]
    f_sel = f1_score(y, sel_pred, average="macro")
    print(f"\n교차선택 F1: {f_sel:.4f} ({f_sel-f_base:+.4f}) | fold별 선택: {chosen}")
    # 쌍별 기여 (최적 조합에서)
    best_gc = max(results, key=lambda gc: results[gc][0]); print(f"OOF 최적 (gap,conf)={best_gc} F1 {results[best_gc][0]:.4f}")
    per = {}
    for pr in pairs_all:
        pred = base.copy()
        for k, (tri, vai) in enumerate(folds):
            models = {pr: fit_pair(X[tri], ylab[tri], pr[0], pr[1], twin[tri])}
            p, _ = route(P[vai], classes, models, X[vai], best_gc[0], best_gc[1], [pr]); pred[vai] = p
        per["+".join(pr)] = round(f1_score(y, pred, average="macro") - f_base, 4)
    print("쌍별 단독 기여(최적 gap/conf):", per)
    json.dump(dict(base=f_base, grid={f"{g},{c}": v for (g, c), v in results.items()}, cross_selected=f_sel, chosen=chosen, best=best_gc, per_pair=per), open(OUT / "result.json", "w"), indent=1)
    np.save(OUT / "oof_pred_best.npy", preds[best_gc])
    if a.submit:
        gap = a.gap or best_gc[0]; conf = a.conf or best_gc[1]
        test = load_test(); Xte = binary_X(test, genes); Pte = np.load(MAIN_TEST)          # 이미 배율 곱해진 확률
        models = {(a_, b_): fit_pair(X, ylab, a_, b_, twin) for a_, b_ in pairs_all}
        # (d) train 쌍둥이가 있는 test 행은 라우팅 제외
        hte = hashes(test, genes); nte = (test[genes] != "WT").sum(1).to_numpy(); lut = {}
        for hh, lab, n in zip(h, ylab, nm):
            if n > 0: lut.setdefault(hh, []).append(lab)
        has_twin = np.array([nte[i] > 0 and hte[i] in lut for i in range(len(test))])
        pred, changed = route(Pte, classes, models, Xte, gap, conf, pairs_all)
        base_te = Pte.argmax(1); pred[has_twin] = base_te[has_twin]
        from postprocess.twin_rule import TwinRule
        lab = le.inverse_transform(pred); lab_tw, n_hit = TwinRule().fit(train).apply(test, lab)
        sub = pd.read_csv(DATA / "sample_submission.csv"); assert (sub[ID] == test[ID]).all()
        sub[TARGET] = lab; sub.to_csv(ROOT / "5. submissions" / f"{a.submit}.csv", index=False, encoding="UTF-8-sig")
        sub[TARGET] = lab_tw; (ROOT / "6. experiments/submissions/twin_rule").mkdir(parents=True, exist_ok=True); sub.to_csv(ROOT / "6. experiments/submissions/twin_rule" / f"{a.submit}_twin_rule.csv", index=False, encoding="UTF-8-sig")
        print(f"[submit] gap={gap} conf={conf} | 9차 대비 바뀐 행 {int((pred != base_te).sum())} (라우팅 제외 쌍둥이 test 행 {has_twin.sum()}) | saved {a.submit}.csv (+twin_rule)")


if __name__ == "__main__":
    main()
