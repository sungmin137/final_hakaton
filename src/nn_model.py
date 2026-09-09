"""GPU(Apple Metal / CUDA) 로 학습하는 MLP 분류기. sklearn 스타일 fit / predict_proba.

- 장치 자동 선택: cuda > mps(Apple Silicon) > cpu
- 입력: FeatureMaker가 만든 피처 행렬 (희소 0/1 + 수치). 수치 컬럼은 train 통계로 표준화(fit에서만 계산).
- 구조: Linear → BatchNorm → GELU → Dropout ×2, 클래스 수만큼 출력. label smoothing, AdamW, cosine LR.
- 조기 종료용 검증은 학습 데이터 내부 10%로만 (외부 검증 fold는 보지 않음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _MLP(nn.Module):
    def __init__(self, d_in: int, n_cls: int, hidden=(512, 256), p=0.4):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier:
    def __init__(self, hidden=(512, 256), dropout=0.4, lr=1e-3, weight_decay=1e-4, epochs=60,
                 batch_size=256, label_smoothing=0.1, patience=8, seed=42, verbose=False):
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, weight_decay
        self.epochs, self.bs, self.ls, self.patience, self.seed, self.verbose = epochs, batch_size, label_smoothing, patience, seed, verbose
        self.device = pick_device()
        self.mu = self.sd = None
        self.model: _MLP | None = None
        self.classes_: np.ndarray | None = None

    # 표준화 통계는 fit 데이터에서만
    def _prep(self, X, fit=False) -> torch.Tensor:
        A = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
        if fit:
            self.mu = A.mean(0); self.sd = A.std(0) + 1e-6
        return torch.from_numpy((A - self.mu) / self.sd)

    def fit(self, X, y, sample_weight=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.classes_ = np.unique(y)
        Xt = self._prep(X, fit=True); yt = torch.from_numpy(np.asarray(y, np.int64))
        n = len(Xt); idx = np.random.permutation(n); n_val = max(1, int(n * 0.1))
        va, tr = idx[:n_val], idx[n_val:]
        self.model = _MLP(Xt.shape[1], len(self.classes_), self.hidden, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        w = None if sample_weight is None else torch.from_numpy(np.asarray(sample_weight, np.float32))
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.ls, reduction="none")
        Xd, yd = Xt.to(self.device), yt.to(self.device)
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.epochs):
            self.model.train(); perm = torch.from_numpy(np.random.permutation(tr))
            for b in range(0, len(perm), self.bs):
                bi = perm[b:b + self.bs].to(self.device)
                out = self.model(Xd[bi]); loss = loss_fn(out, yd[bi])
                if w is not None:
                    loss = loss * w.to(self.device)[bi]
                loss = loss.mean(); opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            # 내부 검증 (macro F1 대신 가벼운 accuracy 로 조기 종료)
            self.model.eval()
            with torch.no_grad():
                acc = (self.model(Xd[va]).argmax(1) == yd[va]).float().mean().item()
            if acc > best:
                best, bad = acc, 0; best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose:
                print(f"ep{ep} val_acc={acc:.4f}")
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        Xt = self._prep(X).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(Xt), 1).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]
