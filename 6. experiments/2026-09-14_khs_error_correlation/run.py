"""실험 ④: 오류 상관 분석. 새로 학습하지 않고 기존 OOF만으로
'현재 블렌드(9차+v2)가 틀리는 샘플을 Linear(LR)/NB(BNB)가 맞히는지'를 확인한다.
third_oof_*.npy는 v9 실험(2026-09-13) 때 같은 정직 CV split으로 이미 저장된 것.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "4. src/common"))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from main import load_train

D = Path(__file__).resolve().parents[1] / "2026-09-13_v4p_xgb_mild_col_grp"

train = load_train()
le = LabelEncoder().fit(train["SUBCLASS"])
y = le.transform(train["SUBCLASS"])

blend = np.load(D / "blend_w05_oof.npy")           # 16차 블렌드(9차+v2) OOF
blend_pred = blend.argmax(1)
wrong = blend_pred != y
n_wrong = wrong.sum()
print(f"블렌드(9차+v2) OOF macro F1 = {f1_score(y, blend_pred, average='macro'):.4f}")
print(f"블렌드가 틀린 샘플: {n_wrong} / {len(y)} ({n_wrong/len(y):.1%})\n")

candidates = {
    "LR C=0.05": "third_oof_LR_C0.05.npy", "LR C=0.2": "third_oof_LR_C0.2.npy",
    "LR C=0.5": "third_oof_LR_C0.5.npy", "LR C=1.0": "third_oof_LR_C1.0.npy",
    "LR C=2.0": "third_oof_LR_C2.0.npy", "LR balanced C=0.1": "third_oof_LR_bal_C0.1.npy",
    "NB(BNB) alpha=0.5": "third_oof_BNB_a0.5.npy", "NB(BNB) alpha=2": "third_oof_BNB_a2.npy",
}

print(f"{'모델':<20}{'단독 macroF1':>14}{'블렌드가 틀렸는데 얘가 맞힘':>22}{'블렌드도 얘도 틀림(둘다 오답)':>22}{'둘 다 다른 오답(불일치)':>16}")
for name, fname in candidates.items():
    p = np.load(D / fname)
    pred = p.argmax(1)
    solo_f1 = f1_score(y, pred, average="macro")
    fixed = (wrong & (pred == y)).sum()                       # 블렌드 오답을 이 모델이 맞힘
    both_wrong = (wrong & (pred != y)).sum()                  # 둘 다 틀림
    both_wrong_diff = (wrong & (pred != y) & (pred != blend_pred)).sum()  # 틀린 답끼리도 다름(다양성)
    print(f"{name:<20}{solo_f1:>14.4f}{fixed:>18} ({fixed/n_wrong:.1%}){both_wrong:>20}{both_wrong_diff:>18}")
