import sys, json
sys.path.insert(0, "/Users/admin/Desktop/해커톤_암종분류/branch_khs/4. src/common")
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from main import load_train, TARGET

SB1 = "/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/layer_oof/"
SB2 = "/private/tmp/claude-501/-Users-admin-Desktop---------/5ebeae57-3cef-4fa1-9493-159d76f5e298/scratchpad/oof/"
SB3 = "/private/tmp/claude-501/-Users-admin-Desktop---------/9403559b-a83c-4ed4-b2a6-937dec6f03d0/scratchpad/"

train = load_train()
le = LabelEncoder().fit(train[TARGET]); y = le.transform(train[TARGET]); classes = list(le.classes_)
y2 = np.load(SB3+"y.npy"); assert np.array_equal(y,y2)

genenb = np.load(SB3+"genenb_multi_oof.npy")
dl = np.load(SB1+"driver_lit/oof_proba.npy")
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
base_pred = (base*scales).argmax(1)
base_f1 = f1_score(y, base_pred, average="macro")
print(f"BASE (접근16) macro F1 = {base_f1:.4f}")

# ---------- (b) finer ADD weight grid, arithmetic only ----------
print("\n----- (b) finer ADD weight grid -----")
grid = [
    ("spec.12/gene.03", {"spec":.12, "gene":.03}),
    ("spec.08/gene.07", {"spec":.08, "gene":.07}),
    ("spec.13/gene.02", {"spec":.13, "gene":.02}),
    ("spec.05/gene.10", {"spec":.05, "gene":.10}),
    ("spec.00/gene.15", {"spec":.00, "gene":.15}),
]
for name, w in grid:
    parts = [(v4s,.45),(v2,.25),(v4sp,.15)]
    if w["spec"]>0: parts.append((spec, w["spec"]))
    if w["gene"]>0: parts.append((genenb, w["gene"]))
    blend = log_blend(*parts)
    pred = (blend*scales).argmax(1)
    f1 = f1_score(y, pred, average="macro")
    print(f"  {name:<20} f1={f1:.4f}  delta={f1-base_f1:+.4f}")

# gene-NB as genuine 5th weight, renormalized to sum 1 (not stealing from existing .15 split)
print("\n  as 5th weight, renormalize existing weights proportionally:")
for gene_w in [0.02, 0.03, 0.05, 0.07, 0.10]:
    remaining = 1 - gene_w
    # keep relative proportions of .45/.25/.15/.15
    orig = np.array([.45,.25,.15,.15])
    orig_scaled = orig / orig.sum() * remaining
    blend = log_blend((v4s,orig_scaled[0]),(v2,orig_scaled[1]),(v4sp,orig_scaled[2]),(spec,orig_scaled[3]),(genenb,gene_w))
    pred = (blend*scales).argmax(1)
    f1 = f1_score(y, pred, average="macro")
    print(f"  gene_w={gene_w:.2f} (others renorm to sum {remaining:.2f}) f1={f1:.4f} delta={f1-base_f1:+.4f}")

# ---------- (a) class-gated blending ----------
print("\n----- (a) class-gated blending -----")
# per-class F1 for base ensemble and gene-NB
per_class_base = {}
per_class_gene = {}
gene_pred = genenb.argmax(1)
for ci,c in enumerate(classes):
    per_class_base[c] = f1_score((y==ci).astype(int), (base_pred==ci).astype(int))
    per_class_gene[c] = f1_score((y==ci).astype(int), (gene_pred==ci).astype(int))

print("class: base_F1 | geneNB_F1 | geneNB-base")
rows = sorted(classes, key=lambda c: per_class_base[c])
for c in rows:
    print(f"  {c:<8} base={per_class_base[c]:.3f}  geneNB={per_class_gene[c]:.3f}  diff={per_class_gene[c]-per_class_base[c]:+.3f}")

# classes where base is weak (bottom third) AND gene-NB is comparatively strong
weak_base_classes = [c for c in classes if per_class_base[c] < np.median(list(per_class_base.values()))]
gate_gain_classes = [c for c in weak_base_classes if per_class_gene[c] > per_class_base[c]]
print(f"\nclasses where base is below-median AND geneNB beats base: {gate_gain_classes}")

# try gating: blend gene-NB in ONLY for rows where base_pred is in gate_gain_classes (using base's own top pred as gate)
class_to_idx = {c:i for i,c in enumerate(classes)}
gate_idx = [class_to_idx[c] for c in gate_gain_classes]
gate_mask = np.isin(base_pred, gate_idx)
print(f"rows gated (base predicted one of {len(gate_gain_classes)} classes): {gate_mask.sum()} / {len(y)}")

for gene_w in [0.10, 0.15, 0.20, 0.30]:
    blend_gated = base.copy()
    # recompute blend only on gated rows with extra gene-NB weight added
    parts_gated = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15),(genenb,gene_w))
    pred_gated = base_pred.copy()
    pred_gated[gate_mask] = (parts_gated[gate_mask]*scales).argmax(1)
    f1 = f1_score(y, pred_gated, average="macro")
    print(f"  gate-blend gene_w={gene_w:.2f} on gated rows only: f1={f1:.4f} delta={f1-base_f1:+.4f} (n_changed={ (pred_gated!=base_pred).sum() })")

# alternative gate: gene-NB's OWN top prediction is one of gate_gain_classes
gate_mask2 = np.isin(gene_pred, gate_idx)
print(f"\nalt gate (geneNB's own argmax in gate_gain_classes): {gate_mask2.sum()} rows")
for gene_w in [0.10, 0.15, 0.20, 0.30]:
    parts_gated = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.15),(genenb,gene_w))
    pred_gated = base_pred.copy()
    pred_gated[gate_mask2] = (parts_gated[gate_mask2]*scales).argmax(1)
    f1 = f1_score(y, pred_gated, average="macro")
    print(f"  gate-blend gene_w={gene_w:.2f}: f1={f1:.4f} delta={f1-base_f1:+.4f} (n_changed={(pred_gated!=base_pred).sum()})")

# ---------- (c) rescue concentration where BOTH v4s and driver-lit wrong ----------
print("\n----- (c) rescue concentration check -----")
v4s_pred = v4s.argmax(1)
dl_pred = dl.argmax(1)
both_wrong = (v4s_pred != y) & (dl_pred != y)
gene_correct = gene_pred == y
base_wrong = base_pred != y

print(f"rows where BOTH v4s and driver-lit(a4) standalone wrong: {both_wrong.sum()} / {len(y)}")
print(f"  of those, gene-NB standalone correct: {(both_wrong & gene_correct).sum()} ({(both_wrong & gene_correct).sum()/both_wrong.sum()*100:.1f}%)")
print(f"  compare: gene-NB overall standalone accuracy on ALL rows: {gene_correct.mean()*100:.1f}%")

print(f"\nrows where 접근16 base_pred wrong AND both v4s,driver-lit standalone wrong: {(base_wrong & both_wrong).sum()}")
resc = base_wrong & both_wrong & gene_correct
print(f"  of those, gene-NB standalone correct (true unique rescue candidates): {resc.sum()}")

# how many of these rescue candidates actually flip to correct in the ADD-weighted ensemble (spec.10/gene.05)?
add1 = log_blend((v4s,.45),(v2,.25),(v4sp,.15),(spec,.10),(genenb,.05))
add1_pred = (add1*scales).argmax(1)
flipped = resc & (add1_pred == y)
print(f"  of these {resc.sum()} rescue candidates, ADD(spec.10/gene.05) ensemble actually flips to correct: {flipped.sum()}")
newly_wrong_elsewhere = (base_pred==y) & (add1_pred!=y)
print(f"  meanwhile ADD ensemble newly breaks (base correct -> add1 wrong) on: {newly_wrong_elsewhere.sum()} rows total")
