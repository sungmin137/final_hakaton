"""test 안 동일 행 쌍 규칙 (2026-09-14). train에서 변이 3개 이상인 완전 동일 행 그룹은 100% 신장(KIPAN/KIRC) 또는 뇌(GBMLGG/LGG) 쌍둥이 쌍이다
(3. docs/04, approach14.md '진단'). 따라서 test 안에서만 서로 동일한 행 쌍(train에 없음, 변이 ≥3)이 신장·뇌 이외로 예측돼 있으면 확실히 틀린 것 →
블렌드 확률의 신장 질량 vs 뇌 질량으로 쌍을 고르고 한 행씩 (KIPAN, KIRC) 또는 (GBMLGG, LGG)로 배정한다.
이미 신장·뇌로 예측된 쌍은 건드리지 않는다(동일 행이라 확률이 같아 나누는 것은 동전 던지기, 기대값 동일). test 통계로 어떤 값도 맞추지 않는다."""
from collections import defaultdict
import numpy as np, pandas as pd
from postprocess.twin_rule import _hash_rows
KID, BRN = ("KIPAN", "KIRC"), ("GBMLGG", "LGG")

def apply_test_pair_rule(train, test, pred, proba, classes, min_mut=3):
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]; ci = {c: i for i, c in enumerate(classes)}
    htr = set(_hash_rows(train, genes)); hte = _hash_rows(test, genes); nm = (test[genes] != "WT").sum(axis=1).to_numpy()
    groups = defaultdict(list)
    for i, h in enumerate(hte):
        if nm[i] >= min_mut and h not in htr: groups[h].append(i)
    out = np.asarray(pred, dtype=object).copy(); changed = []
    for ii in groups.values():
        if len(ii) < 2 or all(out[i] in KID + BRN for i in ii): continue
        mk = proba[ii][:, [ci[c] for c in KID]].sum(); mb = proba[ii][:, [ci[c] for c in BRN]].sum()
        pair = KID if mk >= mb else BRN
        for r, i in enumerate(ii): out[i] = pair[r % 2]; changed.append(i)
    return out, changed
