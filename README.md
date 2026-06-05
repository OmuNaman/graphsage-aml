# Graph Structure Beats Tabular Boosting for Money-Laundering Detection

An inductive **GraphSAGE** study on **SAML-D**: modeling transactions as a graph more than
doubles money-laundering detection quality over a strong XGBoost baseline — for under $1 of compute.

📄 **Paper:** [paper/paper.pdf](paper/paper.pdf) ·
🌐 **Project page:** https://omunaman.github.io/graphsage-aml/ ·
💻 reproducible on [Modal](https://modal.com)

## Summary
- **Problem:** money laundering is relational and multi-hop, but most monitoring scores transactions independently.
- **Approach:** accounts = nodes, transactions = edges; an inductive GraphSAGE-mean encoder + edge-MLP head, evaluated on a **disjoint-node** split (test accounts unseen); benchmarked against XGBoost on identical features; recall ≥ 0.70 as the primary, regulator-relevant metric; focal loss vs weighted cross-entropy.
- **Key result:** GraphSAGE decisively beats XGBoost (PR-AUC **0.74 vs 0.29**, ~**6×** precision at recall ≥ 0.70, Welch p < 0.001). Focal loss is statistically on par with weighted BCE (p = 0.63). Structure, not the loss, is the lever.

## Results
| Model | PR-AUC | ROC-AUC | Precision @ recall≥0.70 | F1 |
|-------|:------:|:-------:|:-----------------------:|:--:|
| XGBoost (tabular) | 0.288 | 0.791 | 0.089 | 0.159 |
| **GraphSAGE + weighted BCE** | **0.741** | **0.934** | 0.564 | 0.627 |
| GraphSAGE + Focal (α=0.5, γ=2) | 0.736 | 0.925 | **0.578** | **0.635** |

Disjoint-node test set, mean over 3 seeds. GraphSAGE vs XGBoost PR-AUC: p < 0.001. Focal vs weighted BCE: not significant (p = 0.63).

## Repository layout
- `src/` — training code (`modal_app.py`), experiment `configs/`, config generator (`gen_configs.py`), figure code (`make_figures.py`)
- `results/` — append-only `results.jsonl` ledger, `experiments_log.md`, `analysis/` (summary + comparison), best-config records
- `paper/` — LaTeX source, figures, compiled `paper.pdf`

## Reproduce
Experiments run on Modal serverless GPUs (L4); the SAML-D dataset is pulled from Kaggle.
```bash
pip install -r requirements.txt
modal setup
modal secret create kaggle-aml KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
modal run src/modal_app.py --config-json "@src/configs/configs_finalists.json"
python src/make_figures.py
```

## Citation
```bibtex
@misc{kudale2026graphsageaml,
  title  = {Graph Structure Beats Tabular Boosting for Money-Laundering Detection:
            An Inductive GraphSAGE Study on SAML-D},
  author = {Kudale, Prit and Dwivedi, Naman and Dandekar, Raj and Dandekar, Rajat and Panat, Sreedath},
  year   = {2026},
  note   = {Vizuara AI Labs}
}
```

_Produced by the Vizuara autonomous research pipeline. All results are real, computed on a single Modal L4 GPU and logged to an append-only ledger._
