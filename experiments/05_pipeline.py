"""Detection-pipeline ablation under the cross-day protocol (all features, 5 seeds).

Compares, on the same held-out days:
  single-stage : one 5-class classifier
  two-stage    : benign-only autoencoder raises the alarm, a classifier trained on attacks only names the type
and measures what the Decision stage's k-of-n temporal vote changes:
  false alarms per benign hour, alarm episodes, and time-to-first-alarm per attack session.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from sklearn.metrics import f1_score
from haids.data import load_raw, add_features, splits, CLASSES, PURGE
from haids.models import AEDetector, fit_supervised
df, groups = add_features(load_raw("data/raw")); F = [f for g in groups.values() for f in g]
X, y = df[F].values, df["label"].values; sess = df["session"].values; t = df["t"].values
VOTES = [(1, 1), (2, 3), (3, 5), (5, 5)]
def vote(a, s, k, n):
    """causal k-of-n vote inside each session"""
    out = np.zeros_like(a)
    for u in pd.unique(s):
        m = s == u; c = pd.Series(a[m]).rolling(n, min_periods=1).sum().values; out[m] = (c >= k).astype(int)
    return out
rows, stage = [], []
for seed in range(5):
    for fold, tr, te in splits(df, "cross_day", seed):
        ntr = tr[y[tr] == 0]; cut = (df["sess_len"].values[ntr] * 0.85).astype(int); pos = df["pos"].values[ntr]
        fit_idx, val_idx = ntr[pos < cut - PURGE], ntr[pos >= cut]
        ae = AEDetector(seed).fit(X[fit_idx], X[val_idx]); thr = np.quantile(ae.score(X[val_idx]), 0.99)
        alarm = (ae.score(X[te]) > thr).astype(int)
        single = np.asarray(fit_supervised("LightGBM", seed, X[tr], y[tr]).predict(X[te])).ravel().astype(int)
        atk = tr[y[tr] > 0]; typer = fit_supervised("LightGBM", seed, X[atk], y[atk] - 1)
        typed = np.asarray(typer.predict(X[te])).ravel().astype(int) + 1
        two = np.where(alarm == 1, typed, 0)
        isatk = y[te] > 0
        stage.append(dict(seed=seed, fold=fold,
            single_macro_f1=f1_score(y[te], single, average="macro"), two_macro_f1=f1_score(y[te], two, average="macro"),
            single_bin_f1=f1_score(isatk, single > 0), two_bin_f1=f1_score(isatk, alarm),
            typing_acc_given_alarm=float((typed[isatk & (alarm == 1)] == y[te][isatk & (alarm == 1)]).mean())))
        ste, tte = sess[te], t[te]
        for k, n in VOTES:
            v = vote(alarm, ste, k, n); ben = ~isatk
            hours = sum((tte[ste == u][-1] - tte[ste == u][0]) / np.timedelta64(1, "h") for u in pd.unique(ste[ben]))
            episodes = sum(int((np.diff(np.r_[0, v[ste == u]]) == 1).sum()) for u in pd.unique(ste[ben]))
            delays, missed = [], 0
            for u in pd.unique(ste[isatk]):
                m = ste == u
                if m.sum() < 30: continue
                hit = np.flatnonzero(v[m])
                if len(hit): delays.append((tte[m][hit[0]] - tte[m][0]) / np.timedelta64(1, "s"))
                else: missed += 1
            rows.append(dict(seed=seed, fold=fold, vote=f"{k}-of-{n}", fpr=float(v[ben].mean()), recall=float(v[isatk].mean()),
                f1=f1_score(isatk, v), false_alarm_episodes_per_h=episodes / hours, median_delay_s=float(np.median(delays)),
                max_delay_s=float(np.max(delays)), sessions_missed=missed, sessions=len(delays) + missed))
    print(seed, flush=True)
s = pd.DataFrame(stage); s.to_json("results/pipeline_stage_runs.json", orient="records")
r = pd.DataFrame(rows); r.to_json("results/pipeline_vote_runs.json", orient="records")
pd.set_option("display.width", 250)
sa = s.groupby("fold").mean(numeric_only=True).drop(columns="seed").round(3); sa.loc["mean"] = s.drop(columns=["seed", "fold"]).mean().round(3)
sa.to_csv("results/pipeline_stage_summary.csv"); print(sa.to_string())
ra = r.groupby("vote", sort=False).agg(fpr=("fpr", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), f1_sd=("f1", "std"),
      false_alarm_episodes_per_h=("false_alarm_episodes_per_h", "mean"), median_delay_s=("median_delay_s", "mean"),
      max_delay_s=("max_delay_s", "max"), sessions_missed=("sessions_missed", "sum"), sessions=("sessions", "sum")).round(3)
ra.to_csv("results/pipeline_vote_summary.csv"); print(ra.to_string())
