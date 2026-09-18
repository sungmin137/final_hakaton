"""mutation-event 단위(gene:specific_mutation) sparse LR.
v9(gene presence/absence)과의 차이: TP53=1 이 아니라 TP53:R175H=1, TP53:R248Q=1 을 별도 피처로 취급.
같은 정직 CV split(twin_groups, SEED=42, 5-fold)으로 fold별 fit → OOF.
과적합 방지: fold의 train 부분에서 2명 미만 등장한 토큰은 어휘에서 제외(cw_plus.py의 기존 규칙과 동일 원칙).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "4. src/common"))

import json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from main import ROOT, load_train, twin_groups, SEED
from features.features import gene_columns

OUT = Path(__file__).parent
D_PREV = ROOT / "6. experiments/2026-09-13_v4p_xgb_mild_col_grp"
MIN_COUNT = 2


def row_tokens(df, genes):
    """행별 {'GENE:MUT': 1} 딕셔너리 리스트 (WT 제외)."""
    arr = df[genes].to_numpy()
    out = []
    for row in arr:
        out.append({f"{g}:{v}": 1 for g, v in zip(genes, row) if v != "WT"})
    return out


def main():
    train = load_train()
    genes = gene_columns(train)
    le = LabelEncoder().fit(train["SUBCLASS"])
    y = le.transform(train["SUBCLASS"])
    classes = list(le.classes_)

    groups = twin_groups(train)
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, y, groups))

    tokens_all = row_tokens(train, genes)
    oof = np.zeros((len(train), len(classes)))
    vocab_sizes = []

    for k, (tri, vai) in enumerate(splits):
        # fold train 부분에서 토큰 빈도 계산 → 2명 미만 제외
        from collections import Counter
        cnt = Counter()
        for i in tri:
            cnt.update(tokens_all[i].keys())
        keep = {t for t, c in cnt.items() if c >= MIN_COUNT}

        dv = DictVectorizer(sparse=True)
        Xtr = dv.fit_transform([{t: 1 for t in tokens_all[i] if t in keep} for i in tri])
        # 어휘를 keep에 맞춰 고정했으므로 valid는 동일 dv로 transform (없는 토큰은 자동 무시)
        Xva = dv.transform([{t: 1 for t in tokens_all[i] if t in keep} for i in vai])
        vocab_sizes.append(Xtr.shape[1])

        for C in (0.01, 0.05, 0.1, 0.3, 1.0):
            clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
            clf.fit(Xtr, y[tri])
            p = clf.predict_proba(Xva)
            f1 = f1_score(y[vai], p.argmax(1), average="macro")
            top = np.bincount(p.argmax(1)).max() / len(vai)
            print(f"fold{k} C={C}: macroF1={f1:.4f}, 최다예측클래스비율={top:.1%}")
        clf = LogisticRegression(C=0.05, max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, y[tri])
        oof[vai] = clf.predict_proba(Xva)

    solo_f1 = f1_score(y, oof.argmax(1), average="macro")
    print(f"\n평균 어휘 크기: {np.mean(vocab_sizes):.0f}개")
    print(f"mutation-event LR 단독 OOF macro F1 = {solo_f1:.4f}")
    np.save(OUT / "oof_mutation_event_lr.npy", oof)

    # === 블렌드(9차+v2)와 비교 ===
    blend = np.load(D_PREV / "blend_w05_oof.npy")
    blend_pred = blend.argmax(1)
    wrong = blend_pred != y
    n_wrong = wrong.sum()
    pred = oof.argmax(1)
    fixed = (wrong & (pred == y)).sum()
    disagree = (pred != blend_pred).sum()
    print(f"\n블렌드가 틀린 {n_wrong}개 중 mutation-event LR이 맞힌 것: {fixed} ({fixed/n_wrong:.1%})")
    print(f"블렌드와 예측이 다른 전체 행 수: {disagree} ({disagree/len(y):.1%})")

    # 클래스별 복구 현황
    print(f"\n{'클래스':<8}{'블렌드 오답 수':>10}{'그중 LR이 맞힘':>12}{'복구율':>8}")
    per_class = {}
    for ci, c in enumerate(classes):
        mask = wrong & (y == ci)
        n = mask.sum()
        if n == 0:
            continue
        fixed_c = (mask & (pred == y)).sum()
        per_class[c] = dict(n_wrong=int(n), n_fixed=int(fixed_c), rate=round(float(fixed_c / n), 4))
        print(f"{c:<8}{n:>10}{fixed_c:>12}{fixed_c/n:>8.1%}")

    json.dump(dict(solo_macro_f1=round(float(solo_f1), 4), n_wrong=int(n_wrong), n_fixed=int(fixed),
                   fix_rate=round(float(fixed/n_wrong), 4), disagree=int(disagree),
                   avg_vocab=int(np.mean(vocab_sizes)), per_class=per_class),
              open(OUT / "result.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
