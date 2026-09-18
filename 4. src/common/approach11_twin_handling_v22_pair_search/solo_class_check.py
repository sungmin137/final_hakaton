"""쌍(pair)이 아니라 단일 클래스를 one-vs-rest로 다루는 서브모델 검증.
PAAD/LIHC처럼 오답이 특정 한 클래스가 아니라 여러 클래스로 흩어지는 경우를 위한 구조.
메인 예측이 무엇이었든 상관없이, 서브모델이 "이 클래스다"라고 높은 확신도로 판단하면 덮어씀.
기존 v11 캐시(oof_main) 재사용, 메인모델 재학습 없음. test.csv는 읽지 않는다.

사용: python3 solo_class_check.py PAAD 0.85
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import TARGET, load_train, twin_groups

V11_CACHE_PATH = Path("6. experiments/2026-09-12_approach11_v11_integration/oof_main_cache.pkl")
TOP_K = 40


def check_solo(target, threshold):
    with open(V11_CACHE_PATH, "rb") as f:
        raw_cache = pickle.load(f)
    train = load_train()
    le = LabelEncoder(); y = le.fit_transform(train[TARGET])
    genes = [c for c in train.columns if c not in ("ID", TARGET)]
    Xbin = (train[genes].to_numpy() != "WT").astype(np.int8)
    groups = twin_groups(train)
    ti = list(le.classes_).index(target)

    rows = []
    for sd, d in raw_cache.items():
        raw = d["oof_main"]
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
        specialist_conf = np.zeros(len(y))
        for fold, (tri, vai) in enumerate(skf.split(train, y, groups)):
            yb_tr = (y[tri] == ti).astype(int)
            pa = Xbin[tri][yb_tr == 1].mean(0); pb = Xbin[tri][yb_tr == 0].mean(0)
            top = np.argsort(np.abs(pa - pb))[-TOP_K:]
            clf = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                                     max_iter=2000, random_state=42 + fold).fit(Xbin[tri][:, top], yb_tr)
            p = clf.predict_proba(Xbin[vai][:, top])
            pos_col = list(clf.classes_).index(1)
            specialist_conf[vai] = p[:, pos_col]
        base = raw.argmax(1)
        use = specialist_conf >= threshold
        routed = base.copy(); routed[use] = ti
        rows.append({"seed": int(sd), "base_f1": float(f1_score(y, base, average="macro")),
                     "router_f1": float(f1_score(y, routed, average="macro")),
                     "delta": float(f1_score(y, routed, average="macro") - f1_score(y, base, average="macro")),
                     "changed": int(use.sum()),
                     "changed_correct": int((use & (y == ti)).sum()),
                     "changed_wrong": int((use & (y != ti)).sum())})
    d_arr = np.array([r["delta"] for r in rows])
    return rows, d_arr


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "PAAD"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    rows, d_arr = check_solo(target, threshold)
    print(f"=== {target} one-vs-rest (임계값 {threshold}) ===")
    for r in rows:
        print(f"  seed={r['seed']}: base={r['base_f1']:.5f} router={r['router_f1']:.5f} "
              f"delta={r['delta']:+.5f} 개입={r['changed']}건(정답전환 {r['changed_correct']}, 오답전환 {r['changed_wrong']})")
    print(f"mean_delta={d_arr.mean():+.5f} std_delta={d_arr.std():.5f} positive_seeds={(d_arr>0).sum()}/{len(d_arr)}")
    ok = d_arr.mean() > d_arr.std() and (d_arr > 0).sum() == len(d_arr)
    print("판정:", "통과(제출 후보)" if ok else "탈락(노이즈 수준 또는 음수 seed 존재)")
