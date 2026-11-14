"""Model zoo. Hyperparameters are fixed a priori (no test-set tuning)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HP = {
    "LogReg": dict(C=1.0, max_iter=2000, class_weight="balanced"),
    "RandomForest": dict(n_estimators=300, max_depth=15, min_samples_leaf=20, min_samples_split=40,
                         max_features="sqrt", class_weight="balanced", n_jobs=-1),
    "XGBoost": dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=30, subsample=0.7,
                    colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0, gamma=0.5, tree_method="hist",
                    n_jobs=-1, verbosity=0),
    "LightGBM": dict(n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05, min_child_samples=50,
                     subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
                     class_weight="balanced", n_jobs=-1, verbose=-1),
    "IsolationForest": dict(n_estimators=200, max_samples=256, max_features=1.0, contamination="auto", n_jobs=-1),
    "Autoencoder": dict(hidden=(64, 32), latent=8, dropout=0.1, lr=1e-3, weight_decay=1e-5, batch_size=256,
                        max_epochs=200, patience=15, clip=10.0),
}


def make_supervised(name: str, seed: int):
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **HP[name]))
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, **HP[name])
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, **HP[name])
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, **HP[name])
    raise ValueError(name)


def fit_supervised(name, seed, Xtr, ytr):
    m = make_supervised(name, seed)
    if name == "XGBoost":   # XGBoost has no class_weight: balanced sample weights instead
        from sklearn.utils.class_weight import compute_sample_weight
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        m.fit(Xtr, ytr)
    return m


class IFDetector:
    """Isolation Forest fitted on benign rows only; score = -decision_function (higher = more anomalous)."""
    def __init__(self, seed):
        self.sc = StandardScaler(); self.m = IsolationForest(random_state=seed, **HP["IsolationForest"])
    def fit(self, Xn):
        self.m.fit(self.sc.fit_transform(Xn)); return self
    def score(self, X):
        return -self.m.decision_function(self.sc.transform(X))


class AEDetector:
    """Symmetric dense autoencoder fitted on benign rows only; score = per-row reconstruction MSE."""
    def __init__(self, seed):
        self.seed = seed; self.hp = HP["Autoencoder"]; self.sc = StandardScaler(); self.epochs_ = 0
    def _net(self, d):
        import torch.nn as nn
        h, z, p = self.hp["hidden"], self.hp["latent"], self.hp["dropout"]
        enc, i = [], d
        for k in h: enc += [nn.Linear(i, k), nn.ReLU(), nn.Dropout(p)]; i = k
        enc += [nn.Linear(i, z)]
        dec, i = [], z
        for k in reversed(h): dec += [nn.Linear(i, k), nn.ReLU()]; i = k
        dec += [nn.Linear(i, d)]
        return nn.Sequential(*enc, *dec)
    def _prep(self, X):
        return np.clip(self.sc.transform(X), -self.hp["clip"], self.hp["clip"]).astype(np.float32)
    def fit(self, Xn_train, Xn_val):
        import torch
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.sc.fit(Xn_train)
        tr, va = torch.from_numpy(self._prep(Xn_train)), torch.from_numpy(self._prep(Xn_val))
        self.net = self._net(tr.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        best, bad, state = np.inf, 0, None
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.hp["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(tr), generator=g)
            for i in range(0, len(tr), self.hp["batch_size"]):
                b = tr[perm[i:i + self.hp["batch_size"]]]
                opt.zero_grad(); loss = ((self.net(b) - b) ** 2).mean(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): v = float(((self.net(va) - va) ** 2).mean())
            if v < best - 1e-5: best, bad, state = v, 0, {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.hp["patience"]: break
        self.net.load_state_dict(state); self.epochs_ = ep + 1; self.net.eval(); return self
    def score(self, X):
        import torch
        with torch.no_grad():
            t = torch.from_numpy(self._prep(X)); return ((self.net(t) - t) ** 2).mean(1).numpy()
    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())
