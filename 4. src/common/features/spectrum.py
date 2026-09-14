"""아미노산 치환 스펙트럼 피처 (2026-09-14, train 전용). 변이 서명(UV·흡연·POLE·MSI·APOBEC)은 특정 치환 쌍 분포로 드러난다는 도메인 지식.
행 단위: missense 20×20 치환 쌍 비율(sp_X>Y, 합=1), 변이 유형 비율(missense/nonsense/frameshift/synonymous/other), 종결·프레임시프트 비율,
치환 쌍을 성질 클래스(H 소수성/P 극성/+ 양전하/- 음전하/G 특수)로 접은 5×5 비율. fit 통계 없음(행 내부 정규화만) → test 누수 없음."""
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
