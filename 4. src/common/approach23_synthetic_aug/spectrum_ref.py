"""성민님 origin/sungmin의 4. src/common/features/spectrum.py를 그대로 복사(read-only 참고용).
approach23는 branch_khs라 sungmin 브랜치를 머지하지 않고 독립적으로 검증하기 위해 복사본을 둔다.
원본 커밋: ec18b88 (2026-09-14)
"""
import re
import numpy as np, pandas as pd
_MIS = re.compile(r"^([A-Z])(\d+)([A-Z])$"); AA = "ACDEFGHIKLMNPQRSTVWY"
_CLS = {**{a: "H" for a in "AVILMFWY"}, **{a: "P" for a in "STNQC"}, **{a: "+" for a in "KRH"}, **{a: "-" for a in "DE"}, "G": "G", "P": "G"}
PAIRS = [f"{a}>{b}" for a in AA for b in AA if a != b]; CLSP = [f"{a}>{b}" for a in "HP+-G" for b in "HP+-G"]

def spectrum_features(df, genes):
    G = df[genes].to_numpy(); n = len(df); pi = {p: i for i, p in enumerate(PAIRS)}; ci = {p: i for i, p in enumerate(CLSP)}
    S = np.zeros((n, len(PAIRS))); Cm = np.zeros((n, len(CLSP))); T = np.zeros((n, 5))   # mis, syn, nonsense, fs, other
    for i in range(n):
        for j in np.nonzero(G[i] != "WT")[0]:
            for t in G[i, j].split(" "):
                m = _MIS.match(t)
                if m:
                    a, b = m.group(1), m.group(3)
                    if a == b: T[i, 1] += 1
                    else:
                        T[i, 0] += 1
                        if f"{a}>{b}" in pi: S[i, pi[f"{a}>{b}"]] += 1
                        k = f"{_CLS.get(a,'?')}>{_CLS.get(b,'?')}"
                        if k in ci: Cm[i, ci[k]] += 1
                elif t.endswith("*"): T[i, 2] += 1
                elif "fs" in t: T[i, 3] += 1
                else: T[i, 4] += 1
    tot = T.sum(1, keepdims=True); mis = np.maximum(S.sum(1, keepdims=True), 1)
    out = pd.DataFrame(np.round(S / mis, 4), columns=[f"sp_{p}" for p in PAIRS], index=df.index)
    for k, p in enumerate(CLSP): out[f"spc_{p}"] = np.round(Cm[:, k] / mis[:, 0], 4)
    for k, name in enumerate(["mis", "syn", "nons", "fs", "oth"]): out[f"spt_{name}"] = np.round(T[:, k] / np.maximum(tot[:, 0], 1), 4)
    out["spt_trunc"] = np.round((T[:, 2] + T[:, 3]) / np.maximum(tot[:, 0], 1), 4)
    return out
