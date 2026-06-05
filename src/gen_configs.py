"""Generate experiment config batches for modal_app.py (graphsage-aml).
Writes train_batch JSON files; focal/finalist batches are generated after the BCE arch is chosen.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def base(exp_id, typ, loss="wbce", seed=0, **kw):
    c = {"exp_id": exp_id, "type": typ, "loss": loss, "seed": seed,
         "epochs": 300, "patience": 20, "lr": 1e-3, "weight_decay": 5e-4, "embed_dim": 16,
         "sage_layers": 2, "hidden_dim": 128, "edge_mlp_layers": 2, "dropout": 0.3}
    c.update(kw)
    return c


# ---- Phase 1: BCE architecture search (seed 0) ----
bce = [
    base("bce-b1", "hpo_bce", sage_layers=2, hidden_dim=128, edge_mlp_layers=2, dropout=0.3),
    base("bce-b2", "hpo_bce", sage_layers=3, hidden_dim=128, edge_mlp_layers=2, dropout=0.3),
    base("bce-b3", "hpo_bce", sage_layers=2, hidden_dim=64,  edge_mlp_layers=2, dropout=0.3),
    base("bce-b4", "hpo_bce", sage_layers=2, hidden_dim=128, edge_mlp_layers=3, dropout=0.3),
    base("bce-b5", "hpo_bce", sage_layers=2, hidden_dim=128, edge_mlp_layers=2, dropout=0.5),
]
json.dump({"action": "train_batch", "cells": bce}, open(os.path.join(HERE, "configs_bce.json"), "w"), indent=2)
print(f"wrote configs_bce.json: {len(bce)} BCE HPO cells")


def gen_focal(arch):
    """arch = dict with sage_layers/hidden_dim/edge_mlp_layers/dropout chosen from BCE search."""
    grid = [("foc-a", 0.25, 2.0), ("foc-b", 0.50, 2.0), ("foc-c", 0.75, 2.0),
            ("foc-d", 0.25, 1.0), ("foc-e", 0.25, 3.0)]
    cells = [base(eid, "hpo_focal", loss="focal", focal_alpha=a, focal_gamma=g, **arch)
             for eid, a, g in grid]
    json.dump({"action": "train_batch", "cells": cells},
              open(os.path.join(HERE, "configs_focal.json"), "w"), indent=2)
    print(f"wrote configs_focal.json: {len(cells)} focal cells on arch {arch}")


def gen_finalists(arch, focal_alpha, focal_gamma):
    cells = []
    for s in [1, 2]:
        cells.append(base(f"fin-bce-s{s}", "finalist", loss="wbce", seed=s, **arch))
        cells.append(base(f"fin-foc-s{s}", "finalist", loss="focal", seed=s,
                          focal_alpha=focal_alpha, focal_gamma=focal_gamma, **arch))
    json.dump({"action": "train_batch", "cells": cells},
              open(os.path.join(HERE, "configs_finalists.json"), "w"), indent=2)
    print(f"wrote configs_finalists.json: {len(cells)} finalist cells")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "focal":
        # usage: gen_configs.py focal '{"sage_layers":2,"hidden_dim":128,"edge_mlp_layers":2,"dropout":0.3}'
        gen_focal(json.loads(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "finalists":
        # usage: gen_configs.py finalists '<arch-json>' <alpha> <gamma>
        gen_finalists(json.loads(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
