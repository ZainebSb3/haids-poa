# Response to the supervisor's review

| Request | Where it is addressed | What was done |
|---|---|---|
| Introduction: "What new scientific insights does HAIDS-PoA enable?" | Section I-A (new subsection with that title); abstract; conclusion | Four measured findings: protocol effect (1.00 vs 0.80-0.87), node vs OS telemetry (0.95 vs 0.66), detection vs typing (0.99 binary; 0.89 two-stage vs 0.83), unseen attacks (0.4-49% vs 88-100%). |
| Ablation analysis | Section VI-B (Table IX, Fig. 4): feature layers. Section VI-D (Table XII): detection pipeline. | 16 feature-set configurations (leave-one-layer-out, single layer, grouped, rolling vs point-in-time) under two protocols; single-stage vs two-stage pipeline; 4 temporal-vote settings. LightGBM, 5 seeds. |
| Justify the choice of models | Section V-C | One paragraph per model family tied to the data (tabular, 49 features, about 29k rows) and to the two operator questions (known vs unknown attacks). The LSTM was removed and the reason is stated. |
| Complete hyperparameters (optimizer, batch size, learning rate, scheduler, seeds, hardware, training time) | Section V-D (Table VI); Table XIV | Every model's configuration; Adam, lr 1e-3, no scheduler, batch 256, early stopping for the autoencoder; seeds 0-4; hardware and library versions; training time mean and std over 5 fits. |
| Complexity analysis (training and inference time) | Section VI-E (Tables XIII, XIV) | Dominant complexity terms; measured training time, single-record latency (median, p95), batch throughput, model size, feature-construction time. |

## What changed compared with the first draft, and why

* **Rolling features.** They were computed after shuffling and grouped by the class label.
  They are now causal, per session, and label-free.
* **Feature selection.** The seven features of the first draft were not produced by the
  selection procedure the text described. Feature sets are now defined a priori by telemetry
  layer, and the ablation shows the effect of each layer.
* **Train/test split.** The per-class 80/20 split kept training and test data in the same
  sessions. The paper now reports three protocols, including unseen days.
* **Autoencoder and Isolation Forest.** The first versions used the block height and the
  block period as inputs and tuned thresholds with attack labels. They are now retrained
  without chain identifiers and with label-free thresholds.
* **LSTM.** Removed. It was trained on windows of shuffled rows.
* **Scores.** The first draft reported macro-F1 0.92-0.93 (supervised) and F1 0.99
  (autoencoder). Those numbers are replaced; they are not comparable with the new ones
  because the features, the split and the protocol all differ.
* **Text corrected to match the data:** 16 raw metrics (not 18); 18.6 h recorded (not 16.4);
  RPC Abuse uses 230 threads everywhere; block period 5 s for Eclipse and Sybil sessions;
  placeholder references [21]-[23] removed; BNaT and TON_IoT references corrected.

## Open items before submission (marked in red in the PDF)

1. Public repository URL (planned: https://github.com/ZainebSb3/haids-poa).
2. Confirm the 5-second block period on 8-9 April (read from the data).
3. Port the LaTeX to the IEEE Access template (the draft uses IEEEtran).
4. Confirm the supervisor's affiliation line and the final author order with all co-authors.
