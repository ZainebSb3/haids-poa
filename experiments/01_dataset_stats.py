"""Dataset inventory: sessions, days, cadence, per-class raw statistics, confound checks."""
import sys, json
import pandas as pd, numpy as np
sys.path.insert(0, "src")
from haids.data import load_raw, CLASSES, RAW, cross_day_side
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
df = load_raw("data/raw")
s = df.groupby(["label", "session"]).agg(n=("pos", "size"), start=("t", "min"), end=("t", "max"), day=("day", "first"),
        block_time=("block_time", "median"), dt=("t", lambda x: x.diff().dt.total_seconds().median())).reset_index()
s["class"] = s.label.map(lambda i: CLASSES[i]); s["minutes"] = ((s.end - s.start).dt.total_seconds() / 60).round(1)
s.drop(columns="label").to_csv("results/sessions.csv", index=False)
summ = df.groupby("label").agg(rows=("pos", "size"), sessions=("session", "nunique"), days=("day", "nunique"),
        block_period=("block_time", "median"), poll_s=("t", lambda x: x.diff().dt.total_seconds().median())).reset_index()
summ["class"] = summ.label.map(lambda i: CLASSES[i])
hrs = s.groupby("class").minutes.sum() / 60
summ["hours"] = summ["class"].map(hrs).round(2)
print(summ.to_string()); print("total rows", len(df), "total hours", round(hrs.sum(), 2))
summ.to_csv("results/class_summary.csv", index=False)
print(df.groupby("label")[RAW + ["block_time", "blocks_behind", "tx_invalid_total"]].median().T.round(2).to_string())
print("constant columns:", [c for c in ["blocks_behind", "tx_invalid_total", "tx_invalid_delta"] if df[c].nunique() == 1])
side = cross_day_side(df)
print(pd.crosstab(df.label.map(lambda i: CLASSES[i]), side))
pd.crosstab(df.label.map(lambda i: CLASSES[i]), side).to_csv("results/cross_day_fold_sizes.csv")
df.groupby("label")[RAW].agg(["median", lambda x: x.quantile(.05), lambda x: x.quantile(.95)]).round(2).to_csv("results/raw_stats.csv")
