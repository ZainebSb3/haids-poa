# HAIDS-PoA raw data

Five CSV files, one per class, in recording order. 29,472 records, 26 columns.

| File | Label | Records | Sessions | Days (April 2026) |
|---|---|---|---|---|
| `normal.csv` | 0 Normal | 15,301 | 8 | 11, 13, 16 |
| `eclipse.csv` | 1 Eclipse | 3,416 | 2 | 8, 9 |
| `flooding.csv` | 2 Flooding | 3,531 | 6 | 10, 11 |
| `sybil.csv` | 3 Sybil | 3,541 | 1 | 9 |
| `rpc_abuse.csv` | 4 RPC Abuse | 3,683 | 18 | 13, 16 |

A session is a run of records with no gap longer than 60 s and no block-height reset
(`src/haids/data.py`). `results/sessions.csv` lists all 35 sessions.

## Columns

Raw metrics (16): `peer_count`, `subnet_diversity`, `peer_churn`, `block_number`, `block_time`,
`blocks_behind`, `gas_used_last_block`, `pending_tx_count`, `rpc_latency_ms`, `cpu_percent`,
`ram_percent`, `net_bytes_recv_rate`, `net_bytes_sent_rate`, `net_connections`,
`p2p_ingress_rate`, `tx_invalid_total`.

Collector-derived (7): `peer_count_delta`, `block_time_delta`, `pending_tx_delta`,
`tx_invalid_delta`, `gas_used_delta`, `pending_tx_rate_30s`, `peer_churn_rate_30s`.

Other: `label`, `attack_type`, `timestamp`.

## Read this before training a model

* **Do not use `block_number`, `block_time`, `block_time_delta` or the polling interval as
  features.** Eclipse and Sybil were recorded with a 5 s block period, the other classes with
  15 s, on a re-initialised chain. These columns identify the chain instance, not the attack.
* `blocks_behind`, `tx_invalid_total`, `tx_invalid_delta` are zero in every record.
* **Sybil has a single session**, recorded with a different collector configuration
  (millisecond timestamps, about 2 ms probe latency). It cannot be tested on an unseen day.
* **Split by session or by day, never by row.** A random split gives a macro-F1 of 1.00 that
  drops to 0.80-0.87 on unseen days.
* `net_connections` and the other operating-system network counters fingerprint the recording
  day. See the feature-layer ablation in the paper.
