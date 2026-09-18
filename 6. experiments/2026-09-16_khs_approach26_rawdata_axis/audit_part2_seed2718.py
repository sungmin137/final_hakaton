import sys, json
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from main import load_train, TARGET

S2DIR = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/"
SB3 = "/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/"

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)

v4s = np.load(S2DIR+"stage2_v4s_oof.npy")
v2  = np.load(S2DIR+"stage2_v2_oof.npy")
v4sp = np.load(S2DIR+"stage2_v4sp_oof.npy")
spec = np.load(S2DIR+"stage2_nb_oof.npy")
genenb = np.load(SB3+"genenb_multi_oof_seed2718.npy")

scale_map = json.load(open("/Users/admin/Desktop/해커톤_암종분류/branch_khs/6. experiments/2026-09-09_v4_xgb_cs/class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

EPS=1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p,EPS))*w for p,w in parts)
    return softmax(z, axis=1)

base = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15))
base_pred = (base*scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")
print(f"[seed=2718] BASE macro F1 = {base_f1:.4f}")

for name, sw, gw in [("spec.12/gene.03", .12, .03), ("spec.13/gene.02", .13, .02), ("spec.075/gene.075",.075,.075)]:
    blend = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,sw),(genenb,gw))
    pred = (blend*scales).argmax(1)
    f1 = f1_score(y, pred, average="macro")
    print(f"  [seed=2718] {name:<20} f1={f1:.4f} delta={f1-base_f1:+.4f}")
