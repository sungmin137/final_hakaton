import sys, json
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from main import load_train, TARGET

SC = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof/"

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)

v4s = np.load(SC+"v4s_oof_proba.npy")
v2  = np.load(SC+"v4p_oof_proba.npy")
v4sp = np.load(SC+"v4sp_oof_proba.npy")
spec = np.load(SC+"oof_spectrum_profile_score.npy")
scale_map = json.load(open(SC+"class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

EPS=1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p,EPS))*w for p,w in parts)
    return softmax(z, axis=1)

base = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15))
pred = (base*scales).argmax(1)
f1 = f1_score(y, pred, average="macro")
print("BASE macro F1 =", f1)

np.save("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/y.npy", y)
np.save("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/scales.npy", scales)
with open("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/round2/classes.json","w") as f:
    json.dump(classes, f)
