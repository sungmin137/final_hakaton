"""Stage1 exploratory: PMI-based gene co-occurrence embedding, TRAIN-ONLY, fold-fit.

Never reads test.csv. Uses cached OOF (v2/v4sp/spec-NB, seed=42) from
/private/tmp/claude-501/.../5ebeae57.../scratchpad to reconstruct 접근16 without
recomputing those legs; only v4s (+embedding) is refit per fold here.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb

CACHE = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad")
SRC = CACHE / "src_sungmin" / "4. src" / "common"
sys.path.insert(0, str(SRC))
import main  # noqa: E402

REPO = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
main.ROOT = REPO
main.DATA = REPO / "1. info" / "data"

OUT = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")

SEED = 42
N_SPLITS = 5
FREQ_CUTOFF = 20   # fold-train carrier count cutoff (see justification printout)
K = 12             # SVD embedding dim


def build_pmi_embedding(bin_mat: np.ndarray, k: int) -> np.ndarray:
    """bin_mat: (n_samples, n_genes) 0/1. Returns (n_genes, k) embedding via SVD of PMI matrix."""
    n = bin_mat.shape[0]
    freq = bin_mat.sum(axis=0).astype(np.float64)          # carrier counts
    p_g = freq / n
    co = bin_mat.T.astype(np.float64) @ bin_mat.astype(np.float64)   # (n_genes, n_genes) co-occurrence counts
    p_gg = co / n
    denom = np.outer(p_g, p_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_gg / denom
    # smoothing: add a small pseudo-count of 0.5/n to co-occurrence prob to avoid log(0) on
    # zero-co-occurrence pairs; floor the ratio so log doesn't blow up to -inf.
    smooth_p_gg = (co + 0.5) / (n + 0.5)
    ratio_smoothed = smooth_p_gg / denom
    pmi = np.log(ratio_smoothed)
    pmi = np.clip(pmi, -10, 10)          # floor/ceiling smoothing
    np.fill_diagonal(pmi, 0.0)            # self-PMI carries no co-occurrence info; zero it out
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    emb = svd.fit_transform(pmi)
    return emb, svd.explained_variance_ratio_


def sample_embedding(values: np.ndarray, gene_index: dict[str, int], freq_genes: list[str], emb: np.ndarray, k: int) -> np.ndarray:
    """Mean embedding vector across each sample's mutated genes restricted to freq_genes.
    Vectorized: build (n_samples, n_freq_genes) binary matrix, then row-normalized matmul with emb."""
    cols = [gene_index[g] for g in freq_genes]
    bin_all = (values[:, cols] != "WT").astype(np.float64)          # (n_samples, n_freq_genes)
    counts = bin_all.sum(axis=1, keepdims=True)                     # mutated-frequent-gene count per sample
    summed = bin_all @ emb                                           # (n_samples, k)
    safe_counts = np.where(counts == 0, 1.0, counts)
    out = summed / safe_counts
    out[counts.ravel() == 0] = 0.0
    return out


def main_run():
    train = main.load_train()
    genes = main.gene_columns(train)
    gene_index = {g: j for j, g in enumerate(genes)}
    values = train[genes].to_numpy()
    le = LabelEncoder().fit(train[main.TARGET])
    y = le.transform(train[main.TARGET])
    classes = list(le.classes_)
    n_classes = len(classes)
    burden = (values != "WT").sum(axis=1)
    groups = main.twin_groups(train)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(train, y, groups))

    # ---- Step0/1 justification printout: fold-train frequency distribution ----
    print("=== frequency cutoff justification (per-fold train portion) ===")
    for fold, (tri, vai) in enumerate(splits):
        cc = (values[tri] != "WT").sum(axis=0)
        n_keep = int((cc >= FREQ_CUTOFF).sum())
        print(f"fold{fold}: n_fit={len(tri)}  genes total={len(genes)}  genes>=|{FREQ_CUTOFF}|carriers={n_keep}  "
              f"median_carrier_count={np.median(cc):.1f}")

    oof_embed_only = np.zeros((len(train), K), dtype=np.float64)   # embedding feature block, OOF-safe
    oof_embed_standalone_proba = np.zeros((len(train), n_classes), dtype=np.float64)
    oof_v4s_plus_embed = np.zeros((len(train), n_classes), dtype=np.float64)
    oof_v4s_only_refit = np.zeros((len(train), n_classes), dtype=np.float64)  # sanity: refit v4s alone here too

    params = main.PARAM_SETS["mild_col"]
    t0 = time.time()
    for fold, (tri, vai) in enumerate(splits):
        # ---- build PMI embedding fit on fold TRAIN ONLY ----
        cc = (values[tri] != "WT").sum(axis=0)
        freq_mask = cc >= FREQ_CUTOFF
        freq_genes = [g for g, m in zip(genes, freq_mask) if m]
        bin_tr = (values[tri][:, freq_mask] != "WT").astype(np.float32)
        emb, expl_var = build_pmi_embedding(bin_tr, K)
        if fold == 0:
            print(f"fold0 embedding: n_freq_genes={len(freq_genes)}  explained_var_ratio(top{K})={expl_var.round(4).tolist()}")

        # sample-level embedding for ALL rows in this fold (both fit and valid), fold-safe
        emb_all = sample_embedding(values, gene_index, freq_genes, emb, K)
        oof_embed_only[vai] = emb_all[vai]

        # ---- Step2: standalone classifier on JUST the embedding block ----
        model_e = xgb.XGBClassifier(**params)
        model_e.fit(emb_all[tri], y[tri])
        oof_embed_standalone_proba[vai] = model_e.predict_proba(emb_all[vai])

        # ---- Step3: v4s feature matrix (fold-fit) + embedding columns ----
        fm = main.FeatureMaker("v4s").fit(train.iloc[tri])
        Xtr_v4s = fm.transform(train.iloc[tri])
        Xva_v4s = fm.transform(train.iloc[vai])

        emb_cols = [f"pmi_emb_{i}" for i in range(K)]
        Xtr_full = pd.concat([Xtr_v4s.reset_index(drop=True),
                               pd.DataFrame(emb_all[tri], columns=emb_cols)], axis=1)
        Xva_full = pd.concat([Xva_v4s.reset_index(drop=True),
                               pd.DataFrame(emb_all[vai], columns=emb_cols)], axis=1)

        model_f = xgb.XGBClassifier(**params)
        model_f.fit(Xtr_full, y[tri])
        oof_v4s_plus_embed[vai] = model_f.predict_proba(Xva_full)

        # sanity refit of plain v4s in THIS run (same fold split / params) to compare apples-to-apples
        model_v4s = xgb.XGBClassifier(**params)
        model_v4s.fit(Xtr_v4s, y[tri])
        oof_v4s_only_refit[vai] = model_v4s.predict_proba(Xva_v4s)

        f1_e = f1_score(y[vai], oof_embed_standalone_proba[vai].argmax(1), average="macro")
        f1_v4s_e = f1_score(y[vai], oof_v4s_plus_embed[vai].argmax(1), average="macro")
        f1_v4s = f1_score(y[vai], oof_v4s_only_refit[vai].argmax(1), average="macro")
        print(f"fold{fold} done ({time.time()-t0:.0f}s)  embed-only F1={f1_e:.4f}  v4s(refit)F1={f1_v4s:.4f}  v4s+embedF1={f1_v4s_e:.4f}")

    np.save(OUT / "oof_embed_standalone.npy", oof_embed_standalone_proba)
    np.save(OUT / "oof_v4s_plus_embed.npy", oof_v4s_plus_embed)
    np.save(OUT / "oof_v4s_only_refit.npy", oof_v4s_only_refit)
    np.save(OUT / "oof_embed_features.npy", oof_embed_only)

    print("\n=== Step2 standalone summary ===")
    print("embed-only OOF macro F1:", f1_score(y, oof_embed_standalone_proba.argmax(1), average="macro"))

    print("\n=== sanity: v4s refit-in-this-script vs cached v4s ===")
    v4s_cached = np.load(CACHE / "oof" / "v4s_oof_proba.npy")
    print("cached v4s OOF macro F1:", f1_score(y, v4s_cached.argmax(1), average="macro"))
    print("this-run v4s-refit OOF macro F1:", f1_score(y, oof_v4s_only_refit.argmax(1), average="macro"))

    # ---- Step3: reconstruct 접근16 (BASE) vs (v4s+embed) ----
    v2 = np.load(CACHE / "oof" / "v4p_oof_proba.npy")
    v4sp = np.load(CACHE / "oof" / "v4sp_oof_proba.npy")
    specnb = np.load(CACHE / "oof_specNB_seed42.npy")
    scale_map = json.loads((CACHE / "oof" / "class_scales_recovered.json").read_text())
    scales = np.array([scale_map[c] for c in classes])

    EPS = 1e-9
    def log_blend(parts):
        z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
        return softmax(z, axis=1)

    weights = dict(v4s=.45, v2=.25, v4sp=.15, specnb=.15)

    base_blend = log_blend([(v4s_cached, weights["v4s"]), (v2, weights["v2"]),
                             (v4sp, weights["v4sp"]), (specnb, weights["specnb"])])
    base_pred = (base_blend * scales).argmax(1)
    base_f1 = f1_score(y, base_pred, average="macro")

    # also a same-script BASE using v4s refit here (controls for any refit-vs-cache drift)
    base_blend_refit = log_blend([(oof_v4s_only_refit, weights["v4s"]), (v2, weights["v2"]),
                                   (v4sp, weights["v4sp"]), (specnb, weights["specnb"])])
    base_pred_refit = (base_blend_refit * scales).argmax(1)
    base_f1_refit = f1_score(y, base_pred_refit, average="macro")

    emb_blend = log_blend([(oof_v4s_plus_embed, weights["v4s"]), (v2, weights["v2"]),
                            (v4sp, weights["v4sp"]), (specnb, weights["specnb"])])
    emb_pred = (emb_blend * scales).argmax(1)
    emb_f1 = f1_score(y, emb_pred, average="macro")

    def burden_report(pred):
        bands = {"0_10": burden <= 10, "11_30": (burden >= 11) & (burden <= 30),
                 "31_100": (burden >= 31) & (burden <= 100), "101_plus": burden >= 101}
        return {k: round(float(f1_score(y[m], pred[m], average="macro")), 5) for k, m in bands.items()}

    print("\n=== Step3 ensemble comparison ===")
    print(f"BASE (cached v4s)      macro F1 = {base_f1:.6f}  burden={burden_report(base_pred)}")
    print(f"BASE (v4s refit here)  macro F1 = {base_f1_refit:.6f}  burden={burden_report(base_pred_refit)}")
    print(f"v4s+embed              macro F1 = {emb_f1:.6f}  burden={burden_report(emb_pred)}  delta_vs_refit_base={emb_f1-base_f1_refit:.6f}")

    changed = emb_pred != base_pred_refit
    helped = changed & (emb_pred == y) & (base_pred_refit != y)
    hurt = changed & (emb_pred != y) & (base_pred_refit == y)
    print(f"changed={changed.sum()}  helped={helped.sum()}  hurt={hurt.sum()}  net={helped.sum()-hurt.sum()}")

    per_class_delta = (pd.Series(f1_score(y, emb_pred, average=None), index=classes)
                        - pd.Series(f1_score(y, base_pred_refit, average=None), index=classes)).sort_values()
    print("\nworst 5 class deltas:\n", per_class_delta.head(5))
    print("\nbest 5 class deltas:\n", per_class_delta.tail(5))

    report = {
        "freq_cutoff": FREQ_CUTOFF, "k": K,
        "embed_standalone_macro_f1": float(f1_score(y, oof_embed_standalone_proba.argmax(1), average="macro")),
        "base_cached_v4s_macro_f1": float(base_f1),
        "base_refit_v4s_macro_f1": float(base_f1_refit),
        "v4s_plus_embed_macro_f1": float(emb_f1),
        "delta_vs_refit_base": float(emb_f1 - base_f1_refit),
        "changed": int(changed.sum()), "helped": int(helped.sum()), "hurt": int(hurt.sum()),
        "burden_base_refit": burden_report(base_pred_refit),
        "burden_emb": burden_report(emb_pred),
        "per_class_delta": per_class_delta.round(5).to_dict(),
    }
    (OUT / "pmi_embedding_result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nsaved report to", OUT / "pmi_embedding_result.json")


if __name__ == "__main__":
    main_run()
