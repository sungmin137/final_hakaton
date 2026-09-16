import sys, time
from pathlib import Path
REPO = Path("/Users/admin/Desktop/해커톤_암종분류/branch_khs")
sys.path.insert(0, str(REPO / "4. src/common"))
from main import load_train, cross_validate, PARAM_SETS  # noqa

OUT = Path("/private/tmp/claude-501/-Users-admin-Desktop---------/1f6224ce-a33c-488d-b884-d2f27a03da05/scratchpad/layer_oof")
OUT.mkdir(parents=True, exist_ok=True)

train = load_train()
print("train shape", train.shape)

for kind, tag in [("a4", "driver_lit"), ("v1", "gene_identity")]:
    t0 = time.time()
    print(f"=== {kind} ({tag}) start ===", flush=True)
    res = cross_validate(train, kind, PARAM_SETS["mild_col"], n_splits=5,
                          out_dir=OUT / tag, group_twins=True, balanced=False, model_name="xgb")
    print(f"=== {kind} done in {time.time()-t0:.0f}s -> macroF1={res['oof_macro_f1']} ===", flush=True)
