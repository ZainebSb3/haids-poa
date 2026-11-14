"""Paper figures (vector PDF). Colours: validated categorical slots; hatching keeps them print-safe."""
import sys, json, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from sklearn.metrics import confusion_matrix
from haids.data import load_raw, add_features, splits, CLASSES
from haids.models import fit_supervised
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.axisbelow": True})
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#dedcd3"
C = ["#2a78d6", "#eb6834", "#1baf7a"]; H = ["", "////", "...."]
OUT = "paper/figures/"
# ---- Fig. protocol effect
b = pd.read_csv("results/benchmark_summary.csv")
models = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]; lab = ["Logistic\nregression", "Random\nForest", "XGBoost", "LightGBM"]
protos = [("random", "Random row split"), ("within", "Within-session, chronological"), ("cross_day", "Cross-day (unseen days)")]
fig, ax = plt.subplots(figsize=(3.45, 2.25)); w = 0.26
for i, (p, name) in enumerate(protos):
    d = b[b.protocol == p].set_index("model").loc[models]
    x = np.arange(4) + (i - 1) * w
    ax.bar(x, d.macro_f1, w - 0.03, color=C[i], hatch=H[i], edgecolor="white", linewidth=0.4, label=name,
           yerr=d.sd, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK))
    if p == "cross_day":
        for xi, v, s_ in zip(x, d.macro_f1, d.sd): ax.text(xi, v + s_ + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab); ax.set_ylim(0.5, 1.12); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylabel("Macro-F1 (5 classes)"); ax.grid(axis="y", color=GRID, lw=0.5); ax.tick_params(length=2, colors=INK)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=1, handlelength=1.4, labelspacing=0.25)
fig.savefig(OUT + "protocol_effect.pdf", bbox_inches="tight"); plt.close(fig)
# ---- Fig. feature-layer ablation
a = pd.read_csv("results/ablation_features_LightGBM.csv").set_index("config")
order = ["all", "- host", "- os_net", "- p2p", "- chain", "- rpc", "only host", "only os_net", "only p2p", "only chain", "only rpc",
         "only host+os_net (OS telemetry)", "only p2p+chain+rpc (node telemetry)"]
names = ["All 49 features", "without Host", "without OS-network", "without P2P", "without Chain", "without RPC", "Host only", "OS-network only",
         "P2P only", "Chain only", "RPC only", "OS telemetry (Host + OS-network)", "Node telemetry (P2P + Chain + RPC)"]
d = a.loc[order]; y = np.arange(len(order))[::-1]
fig, ax = plt.subplots(figsize=(3.45, 3.0))
ax.barh(y, d.f1_cross_day, 0.62, color=C[0], xerr=d.sd_cross_day, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK), label="Cross-day")
ax.plot(d.f1_within, y, "D", ms=3.2, mfc="white", mec=INK, mew=0.7, label="Within-session", zorder=3)
for yi, v in zip(y, d.f1_cross_day): ax.text(0.405, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.3, color="white")
for s in (6.5, 1.5, 11.5): ax.axhline(s, color=GRID, lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.8); ax.set_xlim(0.4, 1.02); ax.set_xlabel("Macro-F1 (LightGBM, 5 seeds)")
ax.grid(axis="x", color=GRID, lw=0.5); ax.tick_params(length=2, colors=INK)
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0)
fig.savefig(OUT + "ablation_layers.pdf", bbox_inches="tight"); plt.close(fig)
# ---- Fig. cross-day confusion matrices, all vs node telemetry (LightGBM, pooled over folds and seeds)
df, groups = add_features(load_raw("data/raw")); yv = df["label"].values
sets = {"All 49 features": [f for g in groups.values() for f in g], "Node telemetry (28 features)": groups["p2p"] + groups["chain"] + groups["rpc"]}
cms = {}
for name, fs in sets.items():
    X = df[fs].values; cm = np.zeros((5, 5))
    for seed in range(5):
        for _, tr, te in splits(df, "cross_day", seed):
            p = np.asarray(fit_supervised("LightGBM", seed, X[tr], yv[tr]).predict(X[te])).ravel().astype(int)
            cm += confusion_matrix(yv[te], p, labels=range(5))
    cms[name] = cm / cm.sum(1, keepdims=True)
json.dump({k: v.round(4).tolist() for k, v in cms.items()}, open("results/confusion_cross_day.json", "w"), indent=1)
short = ["Normal", "Eclipse", "Flood.", "Sybil", "RPC"]
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
for ax, (name, cm) in zip(axs, cms.items()):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            if cm[i, j] >= 0.005: ax.text(j, i, f"{cm[i,j]:.2f}".lstrip("0") if cm[i, j] < 1 else "1.0", ha="center", va="center", fontsize=5.8, color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=6.3); ax.set_yticks(range(5)); ax.set_yticklabels(short, fontsize=6.3)
    ax.set_title(name, fontsize=7); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel("True class", fontsize=7); fig.supxlabel("Predicted class", fontsize=7, y=-0.06)
fig.subplots_adjust(wspace=0.08); fig.savefig(OUT + "confusion_cross_day.pdf", bbox_inches="tight"); plt.close(fig)
print({k: np.diag(v).round(3).tolist() for k, v in cms.items()})
