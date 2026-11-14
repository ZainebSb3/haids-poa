"""Feature-layer ablation (leave-one-group-out, single-group, rolling vs point-in-time)
under the within-session and cross-day protocols. LightGBM, 5 seeds."""
import sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from sklearn.metrics import f1_score, recall_score
from haids.data import load_raw, add_features, splits, GROUPS
from haids.models import fit_supervised
MODEL = sys.argv[1] if len(sys.argv) > 1 else "LightGBM"
raw = load_raw("data/raw"); df, groups = add_features(raw); y = df["label"].values
allf = [f for g in groups.values() for f in g]
point = [f for f in allf if not (f.endswith("_delta") or "_mean" in f or "_std" in f)]
cfg = {"all": allf, "point-in-time only": point, "rolling/delta only": [f for f in allf if f not in point]}
for g in groups:
    cfg[f"- {g}"] = [f for f in allf if f not in groups[g]]
    cfg[f"only {g}"] = groups[g]
cfg["- host - os_net"] = [f for f in allf if f not in groups["host"] + groups["os_net"]]
cfg["only p2p+chain+rpc (node telemetry)"] = groups["p2p"] + groups["chain"] + groups["rpc"]
cfg["only host+os_net (OS telemetry)"] = groups["host"] + groups["os_net"]
cfg["only p2p+rpc"] = groups["p2p"] + groups["rpc"]
cfg["only p2p+chain"] = groups["p2p"] + groups["chain"]
cfg["only p2p+host"] = groups["p2p"] + groups["host"]
cfg["node telemetry, point-in-time"] = [f for f in groups["p2p"] + groups["chain"] + groups["rpc"] if f in point]
rows = []
for name, fs in cfg.items():
    X = df[fs].values
    for proto in ["within", "cross_day"]:
        for seed in range(5):
            for fold, tr, te in splits(df, proto, seed):
                m = fit_supervised(MODEL, seed, X[tr], y[tr]); p = np.asarray(m.predict(X[te])).ravel().astype(int)
                rows.append(dict(config=name, n_feat=len(fs), protocol=proto, fold=fold, seed=seed,
                                 macro_f1=f1_score(y[te], p, average="macro"),
                                 recall=recall_score(y[te], p, average=None, labels=range(5)).tolist()))
    print(name, flush=True)
r = pd.DataFrame(rows); r.to_json(f"results/ablation_features_{MODEL}_runs.json", orient="records")
a = r.groupby(["config", "n_feat", "protocol"]).agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std")).round(3).unstack("protocol")
a.columns = [f"{b}_{c}" for b, c in a.columns]
rec = r[r.protocol == "cross_day"].groupby("config")["recall"].apply(lambda s: np.mean(np.array(s.tolist()), 0).round(2).tolist())
a = a.reset_index().merge(rec.rename("cross_day_recall[N,E,F,S,R]").reset_index()).set_index("config").loc[list(cfg)]
a.to_csv(f"results/ablation_features_{MODEL}.csv"); pd.set_option("display.width", 250); print(a.to_string())
