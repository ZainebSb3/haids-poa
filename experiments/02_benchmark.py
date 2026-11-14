"""Supervised 5-class benchmark under three evaluation protocols, 5 seeds each."""
import sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from haids.data import load_raw, add_features, splits, CLASSES
from haids.models import fit_supervised
SEEDS = range(5); MODELS = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]
df, groups = add_features(load_raw("data/raw")); F = [f for g in groups.values() for f in g]
X, y = df[F].values, df["label"].values
rows = []
for proto in ["random", "within", "cross_day"]:
    for seed in SEEDS:
        for fold, tr, te in splits(df, proto, seed):
            for name in MODELS:
                t0 = time.perf_counter(); m = fit_supervised(name, seed, X[tr], y[tr]); fit_s = time.perf_counter() - t0
                p = np.asarray(m.predict(X[te])).ravel().astype(int)
                rows.append(dict(protocol=proto, fold=fold, seed=seed, model=name, n_train=len(tr), n_test=len(te),
                    acc=accuracy_score(y[te], p), macro_f1=f1_score(y[te], p, average="macro"),
                    macro_prec=precision_score(y[te], p, average="macro", zero_division=0),
                    macro_rec=recall_score(y[te], p, average="macro"),
                    recall=recall_score(y[te], p, average=None, labels=range(5)).round(4).tolist(),
                    cm=confusion_matrix(y[te], p, labels=range(5)).tolist(), fit_s=round(fit_s, 2)))
        print(proto, seed, flush=True)
r = pd.DataFrame(rows); r.to_json("results/benchmark_runs.json", orient="records", indent=0)
agg = r.groupby(["protocol", "model"]).agg(macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"), acc=("acc", "mean"),
        prec=("macro_prec", "mean"), rec=("macro_rec", "mean"), n=("seed", "size")).round(4).reset_index()
rec = r.groupby(["protocol", "model"])["recall"].apply(lambda s: np.mean(np.array(s.tolist()), 0).round(3).tolist()).reset_index()
agg = agg.merge(rec); agg.to_csv("results/benchmark_summary.csv", index=False)
pd.set_option("display.width", 250); print(agg.to_string()); print("features:", len(F))
