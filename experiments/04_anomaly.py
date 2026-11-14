"""Benign-only anomaly detectors (Isolation Forest, Autoencoder) vs. a supervised binary detector.

Thresholds are label-free: the 99th percentile of the detector's scores on held-out benign
validation rows (the chronological tail of every benign training session).
Also runs leave-one-attack-out: a supervised detector that never saw attack k, tested on k.
"""
import sys, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from haids.data import load_raw, add_features, splits, CLASSES, PURGE
from haids.models import IFDetector, AEDetector, fit_supervised
df, groups = add_features(load_raw("data/raw")); y = df["label"].values; yb = (y > 0).astype(int)
SETS = {"all": [f for g in groups.values() for f in g],
        "node": groups["p2p"] + groups["chain"] + groups["rpc"]}
Q = 0.99
def met(score, thr, te):
    p = (score > thr).astype(int)
    d = dict(auc=roc_auc_score(yb[te], score), f1=f1_score(yb[te], p), prec=precision_score(yb[te], p, zero_division=0),
             rec=recall_score(yb[te], p), fpr=float(p[yb[te] == 0].mean()))
    for k in range(1, 5): d[f"rec_{CLASSES[k]}"] = float(p[y[te] == k].mean())
    return d
rows, loo, epochs = [], [], []
for fs_name, fs in SETS.items():
    X = df[fs].values
    for proto in ["within", "cross_day"]:
        for seed in range(5):
            for fold, tr, te in splits(df, proto, seed):
                ntr = tr[y[tr] == 0]
                cut = (df["sess_len"].values[ntr] * 0.8 * 0.85).astype(int) if proto == "within" else (df["sess_len"].values[ntr] * 0.85).astype(int)
                pos = df["pos"].values[ntr]
                fit_idx, val_idx = ntr[pos < cut - PURGE], ntr[pos >= cut]
                base = dict(features=fs_name, protocol=proto, fold=fold, seed=seed)
                iforest = IFDetector(seed).fit(X[fit_idx]); ae = AEDetector(seed).fit(X[fit_idx], X[val_idx])
                epochs.append(ae.epochs_)
                s_if, s_ae = iforest.score(X[te]), ae.score(X[te])
                t_if, t_ae = np.quantile(iforest.score(X[val_idx]), Q), np.quantile(ae.score(X[val_idx]), Q)
                rows.append({**base, "detector": "IsolationForest", **met(s_if, t_if, te)})
                rows.append({**base, "detector": "Autoencoder", **met(s_ae, t_ae, te)})
                n_if, n_ae = s_if / t_if if t_if > 0 else s_if - t_if + 1, s_ae / t_ae
                # ensembles on threshold-normalised scores (alarm when > 1)
                z_if = (s_if - t_if) / (np.std(iforest.score(X[val_idx])) + 1e-9); z_ae = (s_ae - t_ae) / (np.std(ae.score(X[val_idx])) + 1e-9)
                rows.append({**base, "detector": "IF OR AE", **met(np.maximum(z_if, z_ae), 0, te)})
                rows.append({**base, "detector": "IF AND AE", **met(np.minimum(z_if, z_ae), 0, te)})
                # supervised binary reference, same rows; threshold 0.5
                m = fit_supervised("LightGBM", seed, X[tr], yb[tr]); ps = m.predict_proba(X[te])[:, 1]
                rows.append({**base, "detector": "LightGBM (supervised, all attacks seen)", **met(ps, 0.5, te)})
                if proto == "cross_day":
                    for k in range(1, 5):   # zero-day: attack k absent from training
                        trk = tr[y[tr] != k]; mk = fit_supervised("LightGBM", seed, X[trk], yb[trk])
                        tek = te[y[te] == k]
                        loo.append({**base, "held_out": CLASSES[k], "LightGBM_unseen": float((mk.predict_proba(X[tek])[:, 1] > 0.5).mean()),
                                    "IsolationForest": float((s_if[y[te] == k] > t_if).mean()), "Autoencoder": float((s_ae[y[te] == k] > t_ae).mean())})
        print(fs_name, proto, flush=True)
r = pd.DataFrame(rows); r.to_json("results/anomaly_runs.json", orient="records")
a = r.groupby(["features", "protocol", "detector"]).mean(numeric_only=True).drop(columns="seed").round(3)
sd = r.groupby(["features", "protocol", "detector"])["f1"].std().round(3).rename("f1_sd")
a = a.join(sd); a.to_csv("results/anomaly_summary.csv"); pd.set_option("display.width", 250); print(a.to_string())
l = pd.DataFrame(loo); l.to_json("results/zero_day_runs.json", orient="records")
la = l.groupby(["features", "held_out"])[["LightGBM_unseen", "IsolationForest", "Autoencoder"]].agg(["mean", "std"]).round(3)
la.to_csv("results/zero_day_summary.csv"); print(la.to_string()); print("AE epochs: median", np.median(epochs), "range", min(epochs), max(epochs))
json.dump({"ae_epochs_median": float(np.median(epochs)), "ae_epochs_min": int(min(epochs)), "ae_epochs_max": int(max(epochs))}, open("results/ae_epochs.json", "w"))
