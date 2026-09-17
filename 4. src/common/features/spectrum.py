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


# --- v2 (2026-09-15): 종결·프레임시프트에도 서명 정보가 있다. 종결은 "어느 아미노산이 *로"(X→*, 20열), 프레임시프트는 직전 잔기(20열).
_STOP = re.compile(r"^([A-Z])(\d+)\*$"); _FS = re.compile(r"^([A-Z])(\d+)[A-Za-z]*fs")
def spectrum_features_v2(df, genes):
    base = spectrum_features(df, genes); G = df[genes].to_numpy(); n = len(df); ai = {a: i for i, a in enumerate(AA)}
    S = np.zeros((n, 20)); F = np.zeros((n, 20))
    for i in range(n):
        for j in np.nonzero(G[i] != "WT")[0]:
            for t in G[i, j].split(" "):
                m = _STOP.match(t)
                if m and m.group(1) in ai: S[i, ai[m.group(1)]] += 1; continue
                m = _FS.match(t)
                if m and m.group(1) in ai: F[i, ai[m.group(1)]] += 1
    for k, a in enumerate(AA): base[f"spx_{a}_stop"] = np.round(S[:, k] / np.maximum(S.sum(1), 1), 4)
    for k, a in enumerate(AA): base[f"spf_{a}_fs"] = np.round(F[:, k] / np.maximum(F.sum(1), 1), 4)
    base["spx_n_stop"] = S.sum(1); base["spf_n_fs"] = F.sum(1)
    return base


# --- v3 (2026-09-15): 표본 수 기반 수축. missense가 적은 행의 치환 비율은 잡음이므로 균등분포(1/380) 쪽으로 당긴다: (count + K/380) / (n_mis + K). fit 없음.
def spectrum_features_v3(df, genes, K=10.0):
    base = spectrum_features(df, genes); cols = [f"sp_{p}" for p in PAIRS]
    n_mis = (base[cols].to_numpy() > 0).sum(1)  # 근사 대신 원래 count 복원: ratio*n_mis
    # base의 sp_는 ratio(반올림)이므로 count를 다시 세는 편이 정확
    G = df[genes].to_numpy(); n = len(df); pi = {p: i for i, p in enumerate(PAIRS)}; S = np.zeros((n, len(PAIRS)))
    for i in range(n):
        for j in np.nonzero(G[i] != "WT")[0]:
            for t in G[i, j].split(" "):
                m = _MIS.match(t)
                if m and m.group(1) != m.group(3) and f"{m.group(1)}>{m.group(3)}" in pi: S[i, pi[f"{m.group(1)}>{m.group(3)}"]] += 1
    nm = S.sum(1, keepdims=True); shr = (S + K / len(PAIRS)) / (nm + K)
    base[cols] = np.round(shr, 4); base["sp_conf"] = np.round(nm[:, 0] / (nm[:, 0] + K), 4)   # 비율 신뢰도
    return base


# --- v4 (2026-09-16): TP53 한 유전자의 치환 성질 스펙트럼(5×5 = 25열, 행 내부 비율) + TP53 변이 유형(missense/절단/동의) 비율. fit 없음.
def spectrum_features_tp53(df, genes, gene="TP53"):
    base = spectrum_features(df, genes); n = len(df); ci = {p: i for i, p in enumerate(CLSP)}; Cm = np.zeros((n, len(CLSP))); T = np.zeros((n, 3))
    if gene in df.columns:
        col = df[gene].to_numpy()
        for i in range(n):
            if col[i] == "WT": continue
            for t in str(col[i]).split(" "):
                m = _MIS.match(t)
                if m and m.group(1) != m.group(3):
                    T[i, 0] += 1; k = f"{_CLS.get(m.group(1), '?')}>{_CLS.get(m.group(3), '?')}"
                    if k in ci: Cm[i, ci[k]] += 1
                elif m: T[i, 2] += 1
                elif t.endswith("*") or "fs" in t: T[i, 1] += 1
    for k, p in enumerate(CLSP): base[f"tp53c_{p}"] = np.round(Cm[:, k] / np.maximum(Cm.sum(1), 1), 4)
    tot = np.maximum(T.sum(1), 1)
    for k, name in enumerate(["mis", "trunc", "syn"]): base[f"tp53t_{name}"] = np.round(T[:, k] / tot, 4)
    base["tp53_has"] = (T.sum(1) > 0).astype(np.int8)
    return base
