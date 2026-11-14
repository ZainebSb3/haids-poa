"""Computational cost: training time, single-sample and batch inference latency, model size.
Timer: time.perf_counter around fit / predict only (no I/O). Feature vectors are precomputed;
the cost of building one feature vector online is measured separately."""
import sys, os, io, time, json, platform, warnings, pickle
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from haids.data import load_raw, add_features, splits, PURGE, WINDOW, RAW
from haids.models import fit_supervised, IFDetector, AEDetector, HP
raw = load_raw("data/raw"); df, groups = add_features(raw); F = [f for g in groups.values() for f in g]
X, y = df[F].values, df["label"].values
_, tr, te = next(splits(df, "cross_day")); Xtr, ytr = X[tr], y[tr]
x1, xb = X[te][:1], X[te][:1024]
REPS_FIT, WARM, REPS = 5, 50, 1000
def lat(fn):
    for _ in range(WARM): fn(x1)
    a = []
    for i in range(REPS):
        x = X[te][i % len(te)][None, :]; s = time.perf_counter(); fn(x); a.append(time.perf_counter() - s)
    b = []
    for _ in range(20):
        s = time.perf_counter(); fn(xb); b.append(time.perf_counter() - s)
    return dict(single_ms_median=round(np.median(a) * 1e3, 3), single_ms_p95=round(np.percentile(a, 95) * 1e3, 3),
                batch1024_ms=round(np.median(b) * 1e3, 2), rows_per_s=int(1024 / np.median(b)))
def size(o): return round(len(pickle.dumps(o)) / 1e6, 2)
out = {"conditions": dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
        logical_cpus=os.cpu_count(), ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        os=platform.platform(), python=platform.python_version(), n_train=len(tr), n_features=len(F),
        fit_repeats=REPS_FIT, warmup=WARM, single_repeats=REPS, batch_repeats=20, gpu="none")}
import sklearn, xgboost, lightgbm, torch
out["versions"] = dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, lightgbm=lightgbm.__version__,
                       torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__)
for name in ["LogReg", "RandomForest", "XGBoost", "LightGBM"]:
    ts = []
    for s_ in range(REPS_FIT):
        t0 = time.perf_counter(); m = fit_supervised(name, s_, Xtr, ytr); ts.append(time.perf_counter() - t0)
    if name == "RandomForest": m.set_params(n_jobs=1)   # avoid thread start-up cost at batch size 1
    out[name] = dict(fit_s_mean=round(np.mean(ts), 2), fit_s_sd=round(np.std(ts, ddof=1), 2), model_MB=size(m), **lat(m.predict_proba))
ntr = tr[y[tr] == 0]; cut = (df["sess_len"].values[ntr] * 0.85).astype(int); pos = df["pos"].values[ntr]
fi, vi = ntr[pos < cut - PURGE], ntr[pos >= cut]
ts = []
for s_ in range(REPS_FIT):
    t0 = time.perf_counter(); d = IFDetector(s_).fit(X[fi]); ts.append(time.perf_counter() - t0)
d.m.set_params(n_jobs=1)
out["IsolationForest"] = dict(fit_s_mean=round(np.mean(ts), 2), fit_s_sd=round(np.std(ts, ddof=1), 2), model_MB=size(d.m), n_fit=len(fi), **lat(d.score))
ts, eps = [], []
for s_ in range(REPS_FIT):
    t0 = time.perf_counter(); a = AEDetector(s_).fit(X[fi], X[vi]); ts.append(time.perf_counter() - t0); eps.append(a.epochs_)
out["Autoencoder"] = dict(fit_s_mean=round(np.mean(ts), 2), fit_s_sd=round(np.std(ts, ddof=1), 2), epochs_mean=float(np.mean(eps)),
        s_per_epoch=round(np.mean(ts) / np.mean(eps), 3), params=a.n_params(), model_MB=round(a.n_params() * 4 / 1e6, 3), n_fit=len(fi), **lat(a.score))
# online feature construction for one new sample given the last WINDOW raw samples of its session
buf = raw[raw.session == raw.session.iloc[0]].iloc[:WINDOW].copy(); a_ = []
for _ in range(300):
    s = time.perf_counter(); add_features(buf); a_.append(time.perf_counter() - s)
out["feature_vector_online_ms_median"] = round(np.median(a_) * 1e3, 2)
json.dump(out, open("results/cost.json", "w"), indent=1); print(json.dumps(out, indent=1))
