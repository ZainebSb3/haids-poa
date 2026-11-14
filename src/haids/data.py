"""Loading, session segmentation, causal feature engineering and split protocols.

Design rules (each one fixes a flaw of the first pipeline):
  * rows are never shuffled before feature engineering;
  * every derived feature is causal and computed inside one recording session;
  * the class label is never used to build a feature;
  * chain-configuration identifiers (block height, block period) and the
    collector's polling interval are not model inputs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

CLASSES = ["Normal", "Eclipse", "Flooding", "Sybil", "RPC_Abuse"]
FILES = {0: "normal.csv", 1: "eclipse.csv", 2: "flooding.csv", 3: "sybil.csv", 4: "rpc_abuse.csv"}
SESSION_GAP_S = 60          # a gap longer than this starts a new recording session
WINDOW = 5                  # rolling window (samples) = 10 s at the nominal 2 s cadence
PURGE = WINDOW              # rows dropped at a train/test boundary inside a session

# 12 raw metrics kept as model inputs, grouped by telemetry layer
GROUPS = {
    "host": ["cpu_percent", "ram_percent"],
    "os_net": ["net_bytes_recv_rate", "net_bytes_sent_rate", "net_connections"],
    "p2p": ["peer_count", "subnet_diversity", "peer_churn", "p2p_ingress_rate"],
    "chain": ["gas_used_last_block", "pending_tx_count"],
    "rpc": ["rpc_latency_ms"],
}
RAW = [c for g in GROUPS.values() for c in g]
LOG1P = ["net_bytes_recv_rate", "net_bytes_sent_rate", "net_connections", "p2p_ingress_rate",
         "gas_used_last_block", "pending_tx_count", "rpc_latency_ms"]
# never model inputs: identify the chain instance / configuration / collector, not the behaviour
EXCLUDED = ["block_number", "block_time", "block_time_delta", "blocks_behind",
            "tx_invalid_total", "tx_invalid_delta", "timestamp"]


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Concatenate the five class files in chronological order and tag sessions."""
    parts = []
    for label, fname in FILES.items():
        d = pd.read_csv(Path(raw_dir) / fname)
        d["label"] = label
        d["t"] = pd.to_datetime(d["timestamp"], format="mixed")
        dt = d["t"].diff().dt.total_seconds()
        new = (dt > SESSION_GAP_S) | (dt < 0) | (d["block_number"].diff() < 0)
        d["session"] = [f"{CLASSES[label]}-{i:02d}" for i in new.cumsum()]
        d["day"] = d["t"].dt.strftime("%m-%d")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["pos"] = df.groupby("session").cumcount()
    df["sess_len"] = df.groupby("session")["pos"].transform("size")
    return df


def add_features(df: pd.DataFrame, rolling: bool = True) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Causal per-session features. Returns the frame and {group: [feature names]}."""
    out = df.copy()
    feats: dict[str, list[str]] = {g: [] for g in GROUPS}
    for c in LOG1P:
        out[c] = np.log1p(out[c].clip(lower=0))
    out["net_asymmetry"] = out["net_bytes_sent_rate"] - out["net_bytes_recv_rate"]  # log-ratio
    by = out.groupby("session", sort=False)
    for g, cols in GROUPS.items():
        for c in cols:
            feats[g].append(c)
            if rolling:
                r = by[c].rolling(WINDOW, min_periods=1)
                out[f"{c}_mean{WINDOW}"] = r.mean().reset_index(level=0, drop=True)
                out[f"{c}_std{WINDOW}"] = r.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
                out[f"{c}_delta"] = by[c].diff().fillna(0.0)
                feats[g] += [f"{c}_mean{WINDOW}", f"{c}_std{WINDOW}", f"{c}_delta"]
    feats["os_net"].append("net_asymmetry")
    return out, feats


# --------------------------------------------------------------------------- splits
# Cross-day folds: side A / side B never share a recording day for a class.
# Sybil has a single session, so its two halves are used (flagged in every report).
DAY_SIDE = {
    "Normal": {"04-11": "A", "04-13": "B", "04-16": "B"},
    "Eclipse": {"04-08": "A", "04-09": "B"},
    "Flooding": {"04-10": "A", "04-11": "B"},
    "RPC_Abuse": {"04-13": "A", "04-16": "B"},
}


def cross_day_side(df: pd.DataFrame) -> pd.Series:
    side = pd.Series("", index=df.index)
    for cls, m in DAY_SIDE.items():
        k = df["label"] == CLASSES.index(cls)
        side[k] = df.loc[k, "day"].map(m)
    k = df["label"] == CLASSES.index("Sybil")
    half = df.loc[k, "pos"] < df.loc[k, "sess_len"] // 2
    side[k] = np.where(half, "A", "B")
    gap = k & ((df["pos"] - df["sess_len"] // 2).abs() < 6 * PURGE)   # purge around the cut
    side[gap] = "drop"
    return side


def splits(df: pd.DataFrame, protocol: str, seed: int = 0):
    """Yield (name, train_idx, test_idx).

    random       : shuffled 80/20 rows, stratified (what most IDS papers do; optimistic)
    within       : per-session chronological 80/20 with a purge gap
    cross_day    : two folds, test days never seen in training (Sybil: same-session halves)
    """
    idx = np.arange(len(df))
    if protocol == "random":
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(idx, test_size=0.2, stratify=df["label"], random_state=seed)
        yield "random", tr, te
    elif protocol == "within":
        cut = (df["sess_len"] * 0.8).astype(int)
        small = df["sess_len"] < 50
        tr = idx[(small | (df["pos"] < cut - PURGE)).values]
        te = idx[(~small & (df["pos"] >= cut)).values]
        yield "within", tr, te
    elif protocol == "cross_day":
        side = cross_day_side(df)
        a, b = idx[(side == "A").values], idx[(side == "B").values]
        yield "A->B", a, b
        yield "B->A", b, a
    else:
        raise ValueError(protocol)
