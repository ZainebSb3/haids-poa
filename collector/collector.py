"""Reference telemetry collector for one Geth node (reimplementation).

Polls JSON-RPC and host counters every INTERVAL seconds and appends one CSV row per poll,
with the same column names as the HAIDS-PoA raw files. Rebuilt from the paper's description
and the dataset schema; it is not the script that recorded the April 2026 sessions.

    python collector/collector.py --rpc http://127.0.0.1:8545 --out session.csv --label 0 --attack-type Normal
"""
import argparse, csv, time
from collections import deque
from datetime import datetime
import psutil, requests

COLS = ["peer_count", "subnet_diversity", "peer_churn", "block_number", "block_time", "blocks_behind",
        "gas_used_last_block", "pending_tx_count", "rpc_latency_ms", "cpu_percent", "ram_percent",
        "net_bytes_recv_rate", "net_bytes_sent_rate", "net_connections", "p2p_ingress_rate", "tx_invalid_total",
        "label", "attack_type", "peer_count_delta", "block_time_delta", "pending_tx_delta", "tx_invalid_delta",
        "gas_used_delta", "pending_tx_rate_30s", "peer_churn_rate_30s", "timestamp"]


class Rpc:
    def __init__(self, url):
        self.url, self.s, self.i = url, requests.Session(), 0
    def call(self, method, params=None):
        self.i += 1
        r = self.s.post(self.url, json={"jsonrpc": "2.0", "method": method, "params": params or [], "id": self.i}, timeout=5)
        return r.json().get("result")


def poll(rpc, prev, hist):
    t0 = time.perf_counter(); head = int(rpc.call("eth_blockNumber"), 16); latency = (time.perf_counter() - t0) * 1e3
    peers = rpc.call("admin_peers") or []
    ids = {p.get("id") for p in peers}
    subnets = {".".join(p.get("network", {}).get("remoteAddress", "0.0.0.0").split(":")[0].split(".")[:3]) for p in peers}
    blk = rpc.call("eth_getBlockByNumber", ["latest", False]); parent = rpc.call("eth_getBlockByNumber", [hex(max(head - 1, 0)), False])
    pool = rpc.call("txpool_status") or {}
    sync = rpc.call("eth_syncing")
    io = psutil.net_io_counters(); now = time.time()
    ingress = 0.0   # the original source of p2p_ingress_rate is not documented; left at 0 here
    row = dict(peer_count=len(peers), subnet_diversity=len(subnets),
               peer_churn=len(ids ^ prev["ids"]) if prev else 0, block_number=head,
               block_time=int(blk["timestamp"], 16) - int(parent["timestamp"], 16) if head else 0,
               blocks_behind=(int(sync["highestBlock"], 16) - int(sync["currentBlock"], 16)) if isinstance(sync, dict) else 0,
               gas_used_last_block=int(blk["gasUsed"], 16), pending_tx_count=int(pool.get("pending", "0x0"), 16),
               rpc_latency_ms=round(latency, 2), cpu_percent=psutil.cpu_percent(), ram_percent=psutil.virtual_memory().percent,
               net_connections=len(psutil.net_connections()), tx_invalid_total=0)
    dt = (now - prev["t"]) if prev else 0
    row["net_bytes_recv_rate"] = round((io.bytes_recv - prev["io"].bytes_recv) / dt, 2) if dt else 0.0
    row["net_bytes_sent_rate"] = round((io.bytes_sent - prev["io"].bytes_sent) / dt, 2) if dt else 0.0
    row["p2p_ingress_rate"] = ingress
    p = prev["row"] if prev else row
    row.update(peer_count_delta=row["peer_count"] - p["peer_count"], block_time_delta=row["block_time"] - p["block_time"],
               pending_tx_delta=row["pending_tx_count"] - p["pending_tx_count"], tx_invalid_delta=0,
               gas_used_delta=row["gas_used_last_block"] - p["gas_used_last_block"])
    hist.append((now, row["pending_tx_delta"], row["peer_churn"]))
    while hist and now - hist[0][0] > 30: hist.popleft()
    row["pending_tx_rate_30s"] = round(sum(h[1] for h in hist) / 30, 3)
    row["peer_churn_rate_30s"] = round(sum(h[2] for h in hist) / 30, 3)
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return row, dict(ids=ids, io=io, t=now, row=row)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--rpc", default="http://127.0.0.1:8545"); a.add_argument("--out", required=True)
    a.add_argument("--interval", type=float, default=2.0); a.add_argument("--label", type=int, default=0)
    a.add_argument("--attack-type", default="Normal")
    a = a.parse_args()
    rpc, prev, hist = Rpc(a.rpc), None, deque()
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if f.tell() == 0: w.writeheader()
        while True:
            t = time.time()
            try:
                row, prev = poll(rpc, prev, hist); row.update(label=a.label, attack_type=a.attack_type); w.writerow(row); f.flush()
            except Exception as e:
                print("poll failed:", e)
            time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
