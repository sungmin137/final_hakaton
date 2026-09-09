"""Macro F1용 후처리: 클래스별 확률 배율(class scale)을 train OOF에서 최적화.

Macro F1은 소수 클래스 recall에 민감하므로, argmax 전에 클래스별 배율 s_c 를 곱해
소수 클래스가 조금 더 자주 선택되게 하면 점수가 오르는 경우가 많다.
배율은 **train의 OOF 확률과 train 라벨**만으로 좌표 상승(coordinate ascent)으로 찾는다.
test 확률·분포는 전혀 보지 않는다 (규칙 준수).

과적합 방지: OOF를 절반으로 나눠 한쪽에서 배율을 찾고 다른 쪽에서 평가하는 교차 검증도 제공.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

GRID = np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])


def fit_class_scales(oof: np.ndarray, y: np.ndarray, n_rounds: int = 3, grid=GRID) -> np.ndarray:
    K = oof.shape[1]
    s = np.ones(K)
    best = f1_score(y, (oof * s).argmax(1), average="macro")
    for _ in range(n_rounds):
        improved = False
        for c in range(K):
            cur = s[c]
            for g in grid:
                s[c] = g
                f = f1_score(y, (oof * s).argmax(1), average="macro")
                if f > best + 1e-6:
                    best, cur, improved = f, g, True
            s[c] = cur
        if not improved:
            break
    return s


def cross_check(oof: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """OOF를 절반씩 나눠 배율의 일반화 효과를 확인."""
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    base = f1_score(y, oof.argmax(1), average="macro")
    res = {"base": base}
    gains = []
    for fit_i, ev_i in [(a, b), (b, a)]:
        s = fit_class_scales(oof[fit_i], y[fit_i])
        f0 = f1_score(y[ev_i], oof[ev_i].argmax(1), average="macro")
        f1 = f1_score(y[ev_i], (oof[ev_i] * s).argmax(1), average="macro")
        gains.append(f1 - f0)
    res["heldout_gain_mean"] = float(np.mean(gains)); res["heldout_gains"] = gains
    return res
