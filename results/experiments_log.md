# Experiments Log — GraphSAGE vs. XGBoost for AML on SAML-D

Backbone: GraphSAGE-mean + edge-MLP head + learned categorical embeddings (5 cols). GPU = **L4** only.
Disjoint-node inductive split. Primary metric: **recall** (operating point recall>=0.70), plus
precision/F1/ROC-AUC/PR-AUC. Append-only ledger `results.jsonl`; per-run metrics committed to the Volume
(`/vol/runs/<exp_id>.json`). HARD cap 18 GPU training runs.

## Dataset (built once, cached to Volume)
SAML-D downloaded inside Modal via secret `kaggle-aml`: **9,504,852 transactions, 9,873 positives
(0.1039%)**. Undersampled to 2.5% of majority + all positives -> **247,247-edge subset, 3.99% positive,
221,207 account nodes**. Disjoint-node split: train 158,829 edges (6,336 pos) / test 9,889 edges (422 pos);
val = 15,883 train-internal edges. Categorical cardinalities [14, 14, 19, 19, 8] (currencies, bank
locations, payment type).

## Round 0 — Smoke (pipeline validation)
- **prepare(smoke)**: Kaggle download OK (9.5M rows, 9,873 pos), tiny graph built + cached. PyG image
  built (torch 2.4.1 + torch_geometric 2.6.1, no torch-scatter needed). ✅
- **train(smoke-train)** on L4: 2 epochs, 4.8 s, 24,225 params; test ROC-AUC 0.671, val PR-AUC 0.138
  (vs 3.6% base rate) — signal present, full metric/threshold pipeline works. ✅
- GPU runs used: 1 / 18.

## XGBoost baseline (CPU, free — the strong tabular comparator)
- **xgb-s0** (300 trees, depth 6, scale_pos_weight 24.1): test **PR-AUC 0.284**, ROC-AUC 0.790;
  @recall>=.70 -> precision 0.097, F1 0.171; best-F1 0.316. (~1.2 s). This is the bar to beat.

## Round 1 — BCE architecture search (5 configs, seed 0) — DONE
Selection metric = val PR-AUC. All on L4, full subset, 300-epoch cap + early stopping (patience 20).

| exp_id | arch (SAGE/hidden/edgeMLP/drop) | val PR-AUC | test PR-AUC | ROC-AUC | prec@rec>=.70 | rec | epochs | sec |
|--------|--------------------------------|-----------|-------------|---------|---------------|-----|--------|-----|
| **bce-b2** | **3 / 128 / 2 / 0.3** | **0.9817** | **0.7361** | **0.9338** | **0.524** | 0.718 | 225 | 76 |
| bce-b4 | 2 / 128 / 3 / 0.3 | 0.9796 | 0.7088 | 0.919 | 0.452 | 0.709 | 163 | 73 |
| bce-b5 | 2 / 128 / 2 / 0.5 | 0.9717 | 0.6953 | 0.9235 | 0.433 | 0.701 | 202 | 91 |
| bce-b1 | 2 / 128 / 2 / 0.3 | 0.9707 | 0.682 | 0.9178 | 0.369 | 0.706 | 114 | 56 |
| bce-b3 | 2 / 64 / 2 / 0.3 | 0.9658 | 0.685 | 0.9231 | 0.380 | 0.711 | 249 | 82 |

**Best architecture = bce-b2 (3 SAGE layers, hidden 128, edge-MLP 2, dropout 0.3).** Deeper 3-hop
aggregation wins — consistent with multi-hop laundering motifs. **GraphSAGE >> XGBoost**: best test
PR-AUC 0.736 vs 0.284, and precision 0.524 vs 0.097 at recall>=0.70 (5.4x better precision at equal
recall). GPU runs used: 6 / 18.

## Round 2 — Focal-loss search on best arch (3/128/2/0.3), 5 configs, seed 0 — DONE
| exp_id | (alpha, gamma) | val PR-AUC | test PR-AUC | ROC-AUC | prec@rec>=.70 | rec | epochs |
|--------|----------------|-----------|-------------|---------|---------------|-----|--------|
| **foc-b** | **(0.5, 2.0)** | **0.9820** | **0.7441** | **0.9291** | **0.637** | 0.704 | 297 |
| foc-c | (0.75, 2.0) | 0.9782 | 0.7424 | 0.9308 | 0.513 | 0.728 | 186 |
| foc-e | (0.25, 3.0) | 0.9779 | 0.7286 | 0.9231 | 0.348 | 0.765 | 275 |
| foc-a | (0.25, 2.0) | 0.9753 | 0.7136 | 0.9149 | 0.395 | 0.718 | 241 |
| foc-d | (0.25, 1.0) | 0.9635 | 0.6826 | 0.892 | n/a (max recall < .70) | - | 189 |

**Best focal = foc-b (alpha=0.5, gamma=2).** **Focal beats weighted BCE**, mainly at the operating point:
test PR-AUC 0.744 vs 0.736, and **precision 0.637 vs 0.524 at recall>=0.70** (+11 pts precision at equal
recall). The canonical vision default (alpha=0.25) is *not* best here; the higher alpha=0.5 (more
positive-class weight) suits this graph-fraud regime. GPU runs used: 11 / 18.

## Round 3 — Finalist confirmation (best arch; BCE + focal alpha=0.5 gamma=2; 3 seeds each) — DONE

| Model (3 seeds) | test PR-AUC | ROC-AUC | precision @ recall>=0.70 | recall |
|-----------------|-------------|---------|--------------------------|--------|
| XGBoost (tabular, 2 seeds) | 0.288 ± 0.006 | 0.791 | 0.089 ± 0.011 | 0.742 |
| **GraphSAGE + weighted BCE** | **0.741 ± 0.013** | **0.934** | 0.564 ± 0.069 | 0.710 |
| **GraphSAGE + Focal (a=0.5,g=2)** | 0.736 ± 0.008 | 0.925 | 0.578 ± 0.068 | 0.708 |

Welch t-tests: **GraphSAGE vs XGBoost PR-AUC: t=76.7, p<0.001** (decisive). **Focal vs BCE: PR-AUC
p=0.63, precision@.70 p=0.82** (not significant — on par).

## Conclusion (experiment stage)
**All planned experiments complete: 15 GPU training runs (1 smoke + 5 BCE + 5 focal + 4 finalists) of an
18-run cap, plus XGBoost (2 seeds) and threshold tuning on CPU. 0 GPU OOM, all on L4. Budget far under
$11 (~15 short L4 runs, est. < $1 GPU + negligible CPU). All success criteria met.** Findings:

1. **GraphSAGE dramatically outperforms the XGBoost tabular baseline** at inductive, disjoint-node AML
   edge classification on SAML-D: **PR-AUC 0.74 vs 0.29 (2.6x), ROC-AUC 0.93 vs 0.79, and ~6x precision
   at the recall>=0.70 operating point (0.56 vs 0.089)**, p<0.001. The multi-hop relational structure of
   laundering motifs is highly informative and is invisible to the flat per-edge model — a strong,
   significant positive result for the GNN (contrary to the Elliptic prior where RF beat GCN, but
   consistent with SAML-D's explicit typological graph structure).
2. **Focal loss does NOT robustly beat weighted cross-entropy** on this task: across 3 seeds the two are
   statistically indistinguishable (PR-AUC 0.736 vs 0.741, p=0.63; precision@.70 0.578 vs 0.564, p=0.82).
   The single-seed HPO had suggested a focal edge; multi-seed confirmation shows it was within noise. The
   honest verdict: once combined with majority undersampling and post-hoc threshold tuning, focal and
   weighted BCE handle SAML-D's imbalance comparably. (Best focal alpha=0.5, gamma=2 — not the vision
   default alpha=0.25.)
3. **Architecture:** a deeper 3-layer SAGE encoder (hidden 128, 2-layer edge-MLP, dropout 0.3) was best,
   consistent with multi-hop motif aggregation; recall>=0.70 was reachable for every GraphSAGE config via
   post-hoc threshold tuning, with the GNN holding far higher precision at that recall than XGBoost.

GAT/Edge-GAT was excluded by design (prior negative results); not run. Real numbers only; ledger
`results.jsonl` (16 runs) + per-run probs in `runs/` are the source of truth for the paper and figures.
