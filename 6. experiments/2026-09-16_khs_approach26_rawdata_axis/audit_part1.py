import sys, json
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from main import load_train, TARGET

SB1 = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/layer_oof/"
SB2 = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof/"
SB3 = "/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/"

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
y2 = np.load(SB3+"y.npy")
assert np.array_equal(y, y2), "label order mismatch!"

bern = np.load(SB3+"genenb_bern_oof.npy")
multi = np.load(SB3+"genenb_multi_oof.npy")
gi = np.load(SB1+"gene_identity/oof_proba.npy")   # v1 gene-identity-XGB
dl = np.load(SB1+"driver_lit/oof_proba.npy")       # a4 driver-literature-XGB
v4s = np.load(SB2+"v4s_oof_proba.npy")
v2  = np.load(SB2+"v4p_oof_proba.npy")
v4sp = np.load(SB2+"v4sp_oof_proba.npy")
spec = np.load(SB2+"oof_spectrum_profile_score.npy")
scale_map = json.load(open(SB2+"class_scales_recovered.json"))
scales = np.array([scale_map[c] for c in classes])

EPS=1e-12
def log_blend(*parts):
    z = sum(np.log(np.maximum(p,EPS))*w for p,w in parts)
    return softmax(z, axis=1)

base = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15))
base_pred = (base*scales).argmax(1)  # 접근16 ensemble prediction

pred_bern = bern.argmax(1)
pred_multi = multi.argmax(1)
pred_gi = gi.argmax(1)
pred_dl = dl.argmax(1)

def phi(e1, e2):
    # e1, e2 boolean error indicators
    a = (e1 & e2).sum(); b = (e1 & ~e2).sum(); c = (~e1 & e2).sum(); d = (~e1 & ~e2).sum()
    n = a+b+c+d
    num = a*d - b*c
    den = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return num/den if den>0 else float('nan')

def agreement(p1, p2):
    return (p1==p2).mean()

def full_report(name, pred):
    print(f"\n===== {name} =====")
    f1 = f1_score(y, pred, average="macro")
    print(f"macro F1 = {f1:.4f}")
    print("per-class F1:")
    per_class = {}
    for ci, c in enumerate(classes):
        f1c = f1_score((y==ci).astype(int), (pred==ci).astype(int))
        per_class[c] = f1c
    for c in classes:
        print(f"  {c:<8} {per_class[c]:.4f}")
    # prediction distribution vs true distribution
    true_counts = np.bincount(y, minlength=26)
    pred_counts = np.bincount(pred, minlength=26)
    print("\nclass | true_n | pred_n | ratio(pred/true)")
    for ci, c in enumerate(classes):
        ratio = pred_counts[ci]/true_counts[ci] if true_counts[ci]>0 else float('nan')
        print(f"  {c:<8} {true_counts[ci]:>5} {pred_counts[ci]:>5}   {ratio:.2f}")
    n_classes_predicted = (pred_counts>0).sum()
    print(f"\n# distinct classes ever predicted: {n_classes_predicted} / 26")
    # entropy of prediction distribution vs true (normalized)
    p_pred = pred_counts/pred_counts.sum()
    p_true = true_counts/true_counts.sum()
    ent_pred = -np.sum(p_pred[p_pred>0]*np.log(p_pred[p_pred>0]))
    ent_true = -np.sum(p_true[p_true>0]*np.log(p_true[p_true>0]))
    print(f"entropy(pred dist)={ent_pred:.3f} vs entropy(true dist)={ent_true:.3f} (max=ln26={np.log(26):.3f})")
    # confusion highlights: top 8 off-diagonal confusion pairs
    cm = confusion_matrix(y, pred, labels=list(range(26)))
    pairs = []
    for i in range(26):
        for j in range(26):
            if i!=j and cm[i,j]>0:
                pairs.append((cm[i,j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    print("\ntop 10 confusions (true -> pred : count):")
    for cnt, ti, tj in pairs[:10]:
        print(f"  {ti:<8} -> {tj:<8} : {cnt}")
    return per_class, pred_counts, true_counts, f1

per_bern, pc_bern, tc, f1_bern = full_report("BernoulliNB (gene-presence)", pred_bern)
per_multi, pc_multi, tc, f1_multi = full_report("MultinomialNB (gene-NB, chosen model)", pred_multi)

print("\n\n===== PHI CORRELATION TABLE =====")
err_multi = pred_multi != y
err_gi = pred_gi != y
err_dl = pred_dl != y
err_base = base_pred != y

print(f"gene-NB standalone macro F1 = {f1_score(y, pred_multi, average='macro'):.4f}")
print(f"gene-identity-XGB[v1] standalone macro F1 = {f1_score(y, pred_gi, average='macro'):.4f}")
print(f"driver-lit-XGB[a4] standalone macro F1 = {f1_score(y, pred_dl, average='macro'):.4f}")
print(f"접근16 base ensemble macro F1 = {f1_score(y, base_pred, average='macro'):.4f}")

print(f"\nphi(gene-NB error, gene-XGB[v1] error)        = {phi(err_multi, err_gi):.4f}")
print(f"phi(gene-NB error, driver-lit[a4] error)       = {phi(err_multi, err_dl):.4f}")
print(f"phi(gene-NB error, 접근16 ensemble error)       = {phi(err_multi, err_base):.4f}")
print(f"phi(gene-identity[v1] error, driver-lit[a4] error) [reference] = {phi(err_gi, err_dl):.4f}")

print(f"\nraw prediction agreement rate (argmax==argmax):")
print(f"  agree(gene-NB, gene-XGB[v1])   = {agreement(pred_multi, pred_gi):.4f}")
print(f"  agree(gene-NB, driver-lit[a4]) = {agreement(pred_multi, pred_dl):.4f}")
print(f"  agree(gene-NB, 접근16 base)     = {agreement(pred_multi, base_pred):.4f}")
print(f"  agree(gene-XGB[v1], driver-lit[a4]) [reference] = {agreement(pred_gi, pred_dl):.4f}")

json.dump({
  "f1_bern": f1_bern, "f1_multi": f1_multi,
  "f1_gi": float(f1_score(y, pred_gi, average='macro')),
  "f1_dl": float(f1_score(y, pred_dl, average='macro')),
  "f1_base": float(f1_score(y, base_pred, average='macro')),
  "phi_geneNB_geneXGB": float(phi(err_multi, err_gi)),
  "phi_geneNB_driverlit": float(phi(err_multi, err_dl)),
  "phi_geneNB_base16": float(phi(err_multi, err_base)),
  "phi_geneXGB_driverlit_ref": float(phi(err_gi, err_dl)),
  "agree_geneNB_geneXGB": float(agreement(pred_multi, pred_gi)),
  "agree_geneNB_driverlit": float(agreement(pred_multi, pred_dl)),
  "agree_geneNB_base16": float(agreement(pred_multi, base_pred)),
  "agree_geneXGB_driverlit_ref": float(agreement(pred_gi, pred_dl)),
  "pred_class_counts_bern": {c:int(v) for c,v in zip(classes, pc_bern)},
  "pred_class_counts_multi": {c:int(v) for c,v in zip(classes, pc_multi)},
  "true_class_counts": {c:int(v) for c,v in zip(classes, tc)},
}, open(SB3+"audit_part1_result.json","w"), indent=2)
print("\nsaved audit_part1_result.json")
