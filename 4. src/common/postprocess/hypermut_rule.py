"""초과변이 규칙 (2026-09-14). train에서 변이 수가 THRESH(=396, train ACC 최대)를 넘는 행 61개의 라벨은 SKCM 17·UCEC 15·STES 12·COAD 9·BRCA 3·CESC 2 (+DLBC 1, PAAD 1).
test는 초과변이 행이 119개(4.7%, train 1.0%)이고 그중 ACC 18·DLBC 6·SARC 3·LUSC 3행이 예측돼 있다. ACC는 동의변이 배치 표지로 맞히는 클래스인데
이 행들은 ACC 정확 표지를 0~2개만 가져(train ACC 평균 2.7) 초과변이가 유전자 단위 점수를 오작동시킨 것 → 확실히 틀림.
규칙: n_mut > THRESH 이고 예측 클래스가 train 초과변이 행에 MIN_ROWS(=2) 미만으로 나온 클래스면, 허용 클래스 집합 안에서 배율 적용 확률 argmax로 교체.
허용 집합·임계는 train으로만 정한다."""
import numpy as np
THRESH, MIN_ROWS = 396, 2

def fit_allowed(train, genes):
    n = (train[genes] != "WT").sum(axis=1).to_numpy(); lab = train["SUBCLASS"].to_numpy()[n > THRESH]
    vals, cnt = np.unique(lab, return_counts=True); return sorted(vals[cnt >= MIN_ROWS])

def apply_hypermut_rule(train, df, pred, proba_scaled, classes):
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]; allowed = fit_allowed(train, genes); ai = [classes.index(c) for c in allowed]
    n = (df[genes] != "WT").sum(axis=1).to_numpy(); out = np.asarray(pred, dtype=object).copy(); changed = []
    for i in np.flatnonzero(n > THRESH):
        if out[i] in allowed: continue
        out[i] = classes[ai[int(np.argmax(proba_scaled[i][ai]))]]; changed.append(i)
    return out, changed, allowed
