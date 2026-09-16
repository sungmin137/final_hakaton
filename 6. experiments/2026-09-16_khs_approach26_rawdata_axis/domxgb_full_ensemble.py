"""접근16 전체 앙상블에서 v4s -> v4s+domain-hit로 교체했을 때 델타.
v2(v4p)/v4sp/spec-NB는 캐시된 OOF 그대로 재사용 (unchanged).
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
sys.path.insert(0, str(ROOT / "4. src/common"))
from main import load_train, TARGET

SC = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof")
OUT = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad")

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)

v4s_cached = np.load(SC / "v4s_oof_proba.npy")   # cached plain v4s (5ebeae57 session)
v2 = np.load(SC / "v4p_oof_proba.npy")
v4sp = np.load(SC / "v4sp_oof_proba.npy")
spec = np.load(SC / "oof_spectrum_profile_score.npy")
scale_map = json.load(open(SC / "class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

base_this = np.load(OUT / "domxgb_oof_base.npy")   # this session's fresh v4s refit (same recipe)
dom_this = np.load(OUT / "domxgb_oof_dom.npy")     # v4s + domain-hit

EPS = 1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p, EPS)) * w for p, w in parts)
    return softmax(z, axis=1)

# sanity: this-session v4s refit vs cached v4s (both same recipe/seed, should be close)
f1_cached_v4s = f1_score(y, v4s_cached.argmax(1), average="macro")
f1_this_v4s = f1_score(y, base_this.argmax(1), average="macro")
print(f"[sanity] cached v4s standalone   = {f1_cached_v4s:.4f}")
print(f"[sanity] this-session v4s refit  = {f1_this_v4s:.4f}  (should be close to cached)")

# 접근16 BASE using cached v4s (matches ensemble.py base_f1=0.5306)
base_ens = log_blend((v4s_cached, .45), (v2, .25), (v4sp, .15), (spec, .15))
base_pred = (base_ens * scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")
print(f"\n접근16 BASE (cached v4s) macro F1 = {base_f1:.4f}")

# 접근16 with v4s -> this-session v4s refit (own recipe, isolate refit-noise from domain-hit effect)
refit_ens = log_blend((base_this, .45), (v2, .25), (v4sp, .15), (spec, .15))
refit_pred = (refit_ens * scales).argmax(1)
refit_f1 = f1_score(y, refit_pred, average="macro")
print(f"접근16 with this-session v4s refit (no domain) macro F1 = {refit_f1:.4f}  delta vs BASE = {refit_f1-base_f1:+.4f}")

# 접근16 with v4s+domain-hit replacing plain v4s
dom_ens = log_blend((dom_this, .45), (v2, .25), (v4sp, .15), (spec, .15))
dom_pred = (dom_ens * scales).argmax(1)
dom_f1 = f1_score(y, dom_pred, average="macro")
print(f"접근16 with v4s+domain-hit macro F1        = {dom_f1:.4f}  delta vs BASE = {dom_f1-base_f1:+.4f}  delta vs refit-only = {dom_f1-refit_f1:+.4f}")

json.dump({
    "f1_cached_v4s_standalone": float(f1_cached_v4s),
    "f1_this_session_v4s_refit_standalone": float(f1_this_v4s),
    "approach16_base_cached_v4s": float(base_f1),
    "approach16_this_session_v4s_refit_no_domain": float(refit_f1),
    "approach16_delta_refit_vs_cached": float(refit_f1 - base_f1),
    "approach16_v4s_plus_domain": float(dom_f1),
    "approach16_delta_domain_vs_base": float(dom_f1 - base_f1),
    "approach16_delta_domain_vs_refit_only": float(dom_f1 - refit_f1),
}, open(OUT / "domxgb_ensemble_result.json", "w"), indent=2, ensure_ascii=False)
print("\nsaved to", OUT / "domxgb_ensemble_result.json")
