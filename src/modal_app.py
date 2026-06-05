"""
Modal app for: Inductive GraphSAGE vs XGBoost for money-laundering edge classification on SAML-D.

Actions (via --config-json {"action": ...}):
  prepare : CPU. Download SAML-D (Kaggle secret 'kaggle-aml') -> Volume, build the 2.5%-undersampled
            account graph with a disjoint-node inductive split, cache /vol/processed/graph.pt.
  train   : L4 GPU. Load cached graph, train GraphSAGE-mean + edge-MLP (wbce or focal), early-stop on
            val PR-AUC, evaluate on the disjoint-node test set, threshold-sweep, return + cache metrics.
  xgb     : CPU. XGBoost on the identical per-edge features (flat tabular baseline).

GraphSAGE training is the only GPU work (gpu="L4"); prepare + xgb are CPU and do not count against the
GPU-run cap. Per-run metrics are committed to the Volume (/vol/runs/<exp_id>.json) for crash-safety.
"""
import json
import os
import time

import modal

SLUG = "graphsage-aml"
APP_NAME = f"research-{SLUG}"
VOL_NAME = f"research-{SLUG}"
DATA_ROOT = "/vol"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torch_geometric==2.6.1",
        "pandas==2.2.2",
        "scikit-learn==1.5.2",
        "xgboost==2.1.1",
        "kaggle==1.6.17",
        "numpy<2",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
KAGGLE_SECRET = modal.Secret.from_name("kaggle-aml")

CSV_PATH = f"{DATA_ROOT}/data/SAML-D.csv"
GRAPH_PATH = f"{DATA_ROOT}/processed/graph.pt"
CAT_COLS = ["Payment_currency", "Received_currency", "Sender_bank_location",
            "Receiver_bank_location", "Payment_type"]


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ============================ DATA PREP (CPU) ============================
@app.function(image=image, volumes={DATA_ROOT: vol}, secrets=[KAGGLE_SECRET],
              cpu=4.0, memory=32768, timeout=60 * 60)
def prepare_data(config: dict) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    force = bool(config.get("force", False))
    if os.path.exists(GRAPH_PATH) and not force:
        meta = torch.load(GRAPH_PATH, map_location="cpu")["meta"]
        print("graph already cached:", json.dumps(meta))
        return {"status": "cached", **meta}

    # ---- 1. download CSV once ----
    if not os.path.exists(CSV_PATH):
        os.makedirs(f"{DATA_ROOT}/data", exist_ok=True)
        # kaggle reads KAGGLE_USERNAME / KAGGLE_KEY from env (provided by the secret)
        import subprocess
        print("downloading SAML-D from Kaggle ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "berkanoztas/synthetic-transaction-monitoring-dataset-aml",
             "-p", f"{DATA_ROOT}/data", "--unzip"],
            check=True,
        )
        vol.commit()
    # locate the CSV (name may vary)
    if not os.path.exists(CSV_PATH):
        cands = [f for f in os.listdir(f"{DATA_ROOT}/data") if f.lower().endswith(".csv")]
        assert cands, f"no CSV found in {DATA_ROOT}/data: {os.listdir(f'{DATA_ROOT}/data')}"
        os.rename(f"{DATA_ROOT}/data/{cands[0]}", CSV_PATH)
        vol.commit()

    smoke = bool(config.get("smoke", False))
    subset_frac = float(config.get("subset_frac", 0.025))
    seed = int(config.get("prep_seed", 42))
    rng = np.random.RandomState(seed)

    print("reading CSV ...")
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    # normalize label column name
    lab = "Is_laundering" if "Is_laundering" in df.columns else (
        "is_laundering" if "is_laundering" in df.columns else None)
    assert lab is not None, f"no label col in {list(df.columns)}"
    print(f"loaded {len(df):,} rows; columns={list(df.columns)}; positives={int(df[lab].sum()):,}")

    # ---- 2. undersample majority to subset_frac, keep ALL positives ----
    pos = df[df[lab] == 1]
    neg = df[df[lab] == 0]
    if smoke:
        pos = pos.sample(n=min(len(pos), 1500), random_state=seed)
        neg = neg.sample(n=min(len(neg), 40000), random_state=seed)
    else:
        neg = neg.sample(frac=subset_frac, random_state=seed)
    sub = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos_rate = float(sub[lab].mean())
    print(f"subset: {len(sub):,} edges, positive rate {pos_rate:.4%}")

    # ---- 3. temporal features from Date/Time ----
    def parse_temporal(d):
        dt = pd.to_datetime(d["Date"].astype(str) + " " + d["Time"].astype(str), errors="coerce")
        out = pd.DataFrame({
            "hour": dt.dt.hour.fillna(0).astype(int),
            "day_of_week": dt.dt.dayofweek.fillna(0).astype(int),
            "day": dt.dt.day.fillna(1).astype(int),
            "month": dt.dt.month.fillna(1).astype(int),
            "year": dt.dt.year.fillna(2022).astype(int),
        })
        return out
    temporal = parse_temporal(sub)

    # ---- 4. categorical -> integer ids (vocab fit on subset; 0 = pad/unknown) ----
    cat_ids = np.zeros((len(sub), len(CAT_COLS)), dtype=np.int64)
    cat_cards = []
    cat_vocabs = {}
    for j, c in enumerate(CAT_COLS):
        vals = sub[c].astype(str).fillna("NA")
        uniq = {v: i + 1 for i, v in enumerate(sorted(vals.unique()))}
        cat_ids[:, j] = vals.map(uniq).values
        cat_cards.append(len(uniq) + 1)
        cat_vocabs[c] = uniq

    # ---- 5. edge numeric features: log-amount + temporal (z-scored later per train) ----
    amount = sub["Amount"].astype(float).values
    log_amt = np.log1p(np.clip(amount, 0, None))
    edge_num_raw = np.column_stack([
        log_amt,
        temporal["hour"].values, temporal["day_of_week"].values,
        temporal["day"].values, temporal["month"].values, temporal["year"].values,
    ]).astype(np.float32)  # [E, 6]

    # ---- 6. account node ids ----
    senders = sub["Sender_account"].astype(str).values
    receivers = sub["Receiver_account"].astype(str).values
    accounts = np.unique(np.concatenate([senders, receivers]))
    acc_id = {a: i for i, a in enumerate(accounts)}
    src = np.array([acc_id[a] for a in senders], dtype=np.int64)
    dst = np.array([acc_id[a] for a in receivers], dtype=np.int64)
    y = sub[lab].values.astype(np.float32)
    N = len(accounts)
    print(f"graph: {N:,} account nodes, {len(sub):,} transaction edges")

    # ---- 7. disjoint-node split: stratify accounts by 'touches any laundering edge' ----
    acc_pos = np.zeros(N, dtype=bool)
    acc_pos[src[y == 1]] = True
    acc_pos[dst[y == 1]] = True
    test_frac = float(config.get("test_frac", 0.2))
    is_test_acc = np.zeros(N, dtype=bool)
    for grp in [np.where(acc_pos)[0], np.where(~acc_pos)[0]]:
        rng.shuffle(grp)
        k = int(round(test_frac * len(grp)))
        is_test_acc[grp[:k]] = True
    train_acc = ~is_test_acc

    def build_split(mask_acc):
        # edges with BOTH endpoints in this account set
        em = mask_acc[src] & mask_acc[dst]
        # remap node ids to 0..n-1 within the split
        nodes = np.where(mask_acc)[0]
        remap = -np.ones(N, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        s = remap[src[em]]
        d = remap[dst[em]]
        return em, len(nodes), s, d

    tr_em, tr_N, tr_s, tr_d = build_split(train_acc)
    te_em, te_N, te_s, te_d = build_split(is_test_acc)
    print(f"train edges {int(tr_em.sum()):,} (pos {int(y[tr_em].sum()):,}); "
          f"test edges {int(te_em.sum()):,} (pos {int(y[te_em].sum()):,})")

    # ---- 8. node features per split (from incident edges within the split) ----
    def node_feats(n_nodes, s, d, enum, yy):
        log_amt_col = enum[:, 0]
        out_deg = np.bincount(s, minlength=n_nodes).astype(np.float32)
        in_deg = np.bincount(d, minlength=n_nodes).astype(np.float32)
        sent_amt = np.bincount(s, weights=log_amt_col, minlength=n_nodes).astype(np.float32)
        recv_amt = np.bincount(d, weights=log_amt_col, minlength=n_nodes).astype(np.float32)
        mean_sent = sent_amt / np.clip(out_deg, 1, None)
        mean_recv = recv_amt / np.clip(in_deg, 1, None)
        feats = np.column_stack([
            np.log1p(out_deg), np.log1p(in_deg),
            np.log1p(sent_amt), np.log1p(recv_amt),
            mean_sent, mean_recv, np.log1p(out_deg + in_deg),
        ]).astype(np.float32)  # [n_nodes, 7]
        return feats

    x_tr = node_feats(tr_N, tr_s, tr_d, edge_num_raw[tr_em], y[tr_em])
    x_te = node_feats(te_N, te_s, te_d, edge_num_raw[te_em], y[te_em])

    # ---- 9. standardize numeric features using TRAIN stats only (inductive correctness) ----
    def fit_scaler(a):
        mu = a.mean(0); sd = a.std(0); sd[sd == 0] = 1.0
        return mu, sd
    nmu, nsd = fit_scaler(x_tr)
    x_tr = (x_tr - nmu) / nsd
    x_te = (x_te - nmu) / nsd
    emu, esd = fit_scaler(edge_num_raw[tr_em])
    enum_tr = (edge_num_raw[tr_em] - emu) / esd
    enum_te = (edge_num_raw[te_em] - emu) / esd

    # ---- 10. validation split = 10% of TRAIN supervised edges ----
    n_tr_edges = int(tr_em.sum())
    perm = rng.permutation(n_tr_edges)
    n_val = int(round(0.10 * n_tr_edges))
    val_idx = np.zeros(n_tr_edges, dtype=bool); val_idx[perm[:n_val]] = True

    # adjacency for message passing = undirected (add reverse) transaction edges within split
    def undirected(s, d):
        ei = np.vstack([np.concatenate([s, d]), np.concatenate([d, s])])
        return ei.astype(np.int64)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    data = {
        "train": {
            "x": t(x_tr), "edge_index": t(undirected(tr_s, tr_d)),
            "sup_src": t(tr_s), "sup_dst": t(tr_d),
            "cat": t(cat_ids[tr_em]), "num": t(enum_tr), "y": t(y[tr_em]),
            "val_mask": t(val_idx),
        },
        "test": {
            "x": t(x_te), "edge_index": t(undirected(te_s, te_d)),
            "sup_src": t(te_s), "sup_dst": t(te_d),
            "cat": t(cat_ids[te_em]), "num": t(enum_te), "y": t(y[te_em]),
        },
        "cat_cards": cat_cards,
        "n_node_feat": x_tr.shape[1],
        "n_edge_num": enum_tr.shape[1],
        "meta": {
            "n_edges_subset": int(len(sub)), "subset_pos_rate": pos_rate,
            "n_accounts": int(N), "n_train_nodes": int(tr_N), "n_test_nodes": int(te_N),
            "n_train_edges": int(tr_em.sum()), "n_test_edges": int(te_em.sum()),
            "n_train_pos": int(y[tr_em].sum()), "n_test_pos": int(y[te_em].sum()),
            "train_pos_rate": float(y[tr_em].mean()), "test_pos_rate": float(y[te_em].mean()),
            "n_val_edges": int(n_val), "cat_cards": cat_cards, "smoke": smoke,
        },
    }
    os.makedirs(f"{DATA_ROOT}/processed", exist_ok=True)
    save_path = GRAPH_PATH if not smoke else f"{DATA_ROOT}/processed/graph_smoke.pt"
    torch.save(data, save_path)
    vol.commit()
    print("saved", save_path, json.dumps(data["meta"]))
    return {"status": "ok", **data["meta"]}


# ============================ MODEL ============================
def _build_model(cfg, cat_cards, n_node_feat, n_edge_num):
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv

    hidden = int(cfg.get("hidden_dim", 128))
    sage_layers = int(cfg.get("sage_layers", 2))
    edge_mlp_layers = int(cfg.get("edge_mlp_layers", 2))
    dropout = float(cfg.get("dropout", 0.3))
    embed_dim = int(cfg.get("embed_dim", 16))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            d = n_node_feat
            for _ in range(sage_layers):
                self.convs.append(SAGEConv(d, hidden, aggr="mean"))
                d = hidden
            self.embeds = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cards])
            self.drop = nn.Dropout(dropout)
            edge_in = hidden * 2 + embed_dim * len(cat_cards) + n_edge_num
            mlp = []
            d = edge_in
            for _ in range(edge_mlp_layers - 1):
                mlp += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            mlp += [nn.Linear(d, 1)]
            self.edge_mlp = nn.Sequential(*mlp)

        def encode(self, x, edge_index):
            import torch.nn.functional as F
            h = x
            for conv in self.convs:
                h = self.drop(F.relu(conv(h, edge_index)))
            return h

        def forward(self, x, edge_index, sup_src, sup_dst, cat, num):
            import torch
            h = self.encode(x, edge_index)
            cat_e = torch.cat([emb(cat[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
            e = torch.cat([h[sup_src], h[sup_dst], cat_e, num], dim=1)
            return self.edge_mlp(e).squeeze(-1)

    return Net()


def _metrics(y_true, prob):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    prauc = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    # threshold sweep 0.10..0.90
    sweep = []
    best_op = None
    for thr in [round(0.10 + 0.05 * i, 2) for i in range(17)]:
        pred = (prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        row = {"thr": thr, "recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4)}
        sweep.append(row)
        # operating point: smallest threshold (highest recall) with recall>=0.70, max precision among those
        if rec >= 0.70:
            if best_op is None or prec > best_op["precision"]:
                best_op = row
    best_f1 = max(sweep, key=lambda r: r["f1"])
    return {"roc_auc": round(roc, 4), "pr_auc": round(prauc, 4),
            "op_recall70": best_op, "best_f1": best_f1, "sweep": sweep}


# ============================ TRAIN (L4 GPU) ============================
@app.function(image=image, volumes={DATA_ROOT: vol}, gpu="L4", timeout=2 * 60 * 60)
def train(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    exp_id = cfg["exp_id"]
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    smoke = bool(cfg.get("smoke", False))
    path = f"{DATA_ROOT}/processed/graph_smoke.pt" if smoke else GRAPH_PATH
    blob = torch.load(path, map_location="cpu")
    tr, te = blob["train"], blob["test"]
    cat_cards = blob["cat_cards"]

    def to_dev(d):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    tr = to_dev(tr); te = to_dev(te)

    model = _build_model(cfg, cat_cards, blob["n_node_feat"], blob["n_edge_num"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                            weight_decay=float(cfg.get("weight_decay", 5e-4)))

    y = tr["y"]; val_mask = tr["val_mask"].bool(); train_mask = ~val_mask
    loss_kind = cfg.get("loss", "wbce")
    n_pos = float(y[train_mask].sum()); n_neg = float(train_mask.sum() - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=dev)
    alpha = float(cfg.get("focal_alpha", 0.25)); gamma = float(cfg.get("focal_gamma", 2.0))

    def compute_loss(logits, target):
        if loss_kind == "focal":
            p = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p_t = p * target + (1 - p) * (1 - target)
            a_t = alpha * target + (1 - alpha) * (1 - target)
            return (a_t * (1 - p_t) ** gamma * ce).mean()
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    max_epochs = 2 if smoke else int(cfg.get("epochs", 300))
    patience = int(cfg.get("patience", 20))
    best_val = -1.0; best_state = None; bad = 0
    run_dir = f"{DATA_ROOT}/runs"; os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    from sklearn.metrics import average_precision_score
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        logits = model(tr["x"], tr["edge_index"], tr["sup_src"], tr["sup_dst"], tr["cat"], tr["num"])
        loss = compute_loss(logits[train_mask], y[train_mask])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(logits[val_mask]).detach().cpu().numpy()
            vy = y[val_mask].cpu().numpy()
            val_prauc = float(average_precision_score(vy, vp)) if vy.sum() > 0 else 0.0
        if val_prauc > best_val:
            best_val = val_prauc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or bad == 0:
            _write_json(f"{run_dir}/{exp_id}.progress.json",
                        {"exp_id": exp_id, "epoch": epoch + 1, "max_epochs": max_epochs,
                         "loss": float(loss), "val_pr_auc": round(val_prauc, 4),
                         "best_val_pr_auc": round(best_val, 4), "done": False})
            vol.commit()
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    # ---- test (disjoint-node, inductive) ----
    model.eval()
    with torch.no_grad():
        te_logits = model(te["x"], te["edge_index"], te["sup_src"], te["sup_dst"], te["cat"], te["num"])
        te_prob = torch.sigmoid(te_logits).cpu().numpy()
        te_y = te["y"].cpu().numpy()
        # val probs at best state for threshold selection reference
        tr_logits = model(tr["x"], tr["edge_index"], tr["sup_src"], tr["sup_dst"], tr["cat"], tr["num"])
        va_prob = torch.sigmoid(tr_logits[val_mask]).cpu().numpy(); va_y = y[val_mask].cpu().numpy()
    test_m = _metrics(te_y, te_prob)
    val_m = _metrics(va_y, va_prob)
    n_params = int(sum(p.numel() for p in model.parameters()))
    result = {
        "exp_id": exp_id, "type": cfg.get("type", ""), "loss": loss_kind,
        "sage_layers": cfg.get("sage_layers"), "hidden_dim": cfg.get("hidden_dim"),
        "edge_mlp_layers": cfg.get("edge_mlp_layers"), "dropout": cfg.get("dropout"),
        "focal_alpha": alpha if loss_kind == "focal" else None,
        "focal_gamma": gamma if loss_kind == "focal" else None,
        "seed": seed, "epochs_ran": epoch + 1, "best_val_pr_auc": round(best_val, 4),
        "n_params": n_params, "train_time_sec": round(time.time() - t0, 1),
        "test": test_m, "val": val_m, "meta": blob["meta"], "smoke": smoke,
        # convenience top-level numbers (test, at recall>=0.70 operating point)
        "test_pr_auc": test_m["pr_auc"], "test_roc_auc": test_m["roc_auc"],
        "test_recall70": (test_m["op_recall70"] or {}).get("recall"),
        "test_precision70": (test_m["op_recall70"] or {}).get("precision"),
        "test_f1_at70": (test_m["op_recall70"] or {}).get("f1"),
        "test_best_f1": test_m["best_f1"]["f1"],
    }
    _write_json(f"{run_dir}/{exp_id}.json", result)
    _write_json(f"{run_dir}/{exp_id}.progress.json", {"exp_id": exp_id, "done": True, **result["test"]})
    # also dump raw test probs for figures
    np.savez_compressed(f"{run_dir}/{exp_id}_probs.npz", y=te_y, prob=te_prob)
    vol.commit()
    print(f"{exp_id}: PR-AUC={test_m['pr_auc']} ROC={test_m['roc_auc']} "
          f"op@rec>=.70 -> rec={result['test_recall70']} prec={result['test_precision70']} "
          f"({result['train_time_sec']}s, {epoch+1} ep)")
    return result


# ============================ XGBOOST (CPU) ============================
@app.function(image=image, volumes={DATA_ROOT: vol}, cpu=8.0, memory=16384, timeout=60 * 60)
def xgboost_baseline(cfg: dict) -> dict:
    import numpy as np
    import torch
    import xgboost as xgb

    exp_id = cfg.get("exp_id", "xgb")
    seed = int(cfg.get("seed", 0))
    blob = torch.load(GRAPH_PATH, map_location="cpu")
    tr, te = blob["train"], blob["test"]

    def feats(split):
        cat = split["cat"].numpy().astype(np.float32)
        num = split["num"].numpy().astype(np.float32)
        return np.concatenate([cat, num], axis=1)
    Xtr = feats(tr); ytr = tr["y"].numpy()
    Xte = feats(te); yte = te["y"].numpy()
    val_mask = tr["val_mask"].numpy().astype(bool)
    spw = float((ytr[~val_mask] == 0).sum()) / max(float((ytr[~val_mask] == 1).sum()), 1.0)

    clf = xgb.XGBClassifier(
        n_estimators=int(cfg.get("n_estimators", 300)), max_depth=int(cfg.get("max_depth", 6)),
        learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="aucpr", n_jobs=8, random_state=seed, tree_method="hist",
    )
    t0 = time.time()
    clf.fit(Xtr[~val_mask], ytr[~val_mask])
    te_prob = clf.predict_proba(Xte)[:, 1]
    test_m = _metrics(yte, te_prob)
    result = {
        "exp_id": exp_id, "type": "baseline_xgb", "model": "xgboost", "seed": seed,
        "scale_pos_weight": round(spw, 2), "train_time_sec": round(time.time() - t0, 1),
        "test": test_m, "test_pr_auc": test_m["pr_auc"], "test_roc_auc": test_m["roc_auc"],
        "test_recall70": (test_m["op_recall70"] or {}).get("recall"),
        "test_precision70": (test_m["op_recall70"] or {}).get("precision"),
        "test_f1_at70": (test_m["op_recall70"] or {}).get("f1"),
        "test_best_f1": test_m["best_f1"]["f1"], "meta": blob["meta"],
    }
    os.makedirs(f"{DATA_ROOT}/runs", exist_ok=True)
    _write_json(f"{DATA_ROOT}/runs/{exp_id}.json", result)
    np.savez_compressed(f"{DATA_ROOT}/runs/{exp_id}_probs.npz", y=yte, prob=te_prob)
    vol.commit()
    print(f"{exp_id} (xgb): PR-AUC={test_m['pr_auc']} ROC={test_m['roc_auc']} "
          f"op@rec>=.70 prec={result['test_precision70']}")
    return result


@app.function(image=image, volumes={DATA_ROOT: vol}, cpu=2.0, memory=8192, timeout=20 * 60)
def export_viz(cfg: dict) -> dict:
    """Export a small laundering-motif subgraph from the cached graph for visualization."""
    import numpy as np
    import torch
    blob = torch.load(GRAPH_PATH, map_location="cpu")
    tr = blob["train"]
    s = tr["sup_src"].numpy(); d = tr["sup_dst"].numpy(); y = tr["y"].numpy().astype(int)
    # adjacency for neighborhood expansion
    from collections import defaultdict
    nbr = defaultdict(set)
    for a, b in zip(s, d):
        nbr[int(a)].add(int(b)); nbr[int(b)].add(int(a))
    # seed from a few laundering edges, grow a connected node set up to a cap
    pos_e = np.where(y == 1)[0]
    rng = np.random.RandomState(int(cfg.get("seed", 7)))
    rng.shuffle(pos_e)
    cap = int(cfg.get("max_nodes", 70))
    keep = set()
    for ei in pos_e:
        a, b = int(s[ei]), int(d[ei])
        frontier = [a, b]
        for node in frontier:
            keep.add(node)
            for nb in list(nbr[node])[:6]:
                keep.add(nb)
        if len(keep) >= cap:
            break
    keep = set(list(keep)[:cap])
    # collect edges fully inside 'keep'
    edges = []
    for a, b, lab in zip(s, d, y):
        if int(a) in keep and int(b) in keep:
            edges.append([int(a), int(b), int(lab)])
    nodes = sorted({n for e in edges for n in e[:2]})
    remap = {n: i for i, n in enumerate(nodes)}
    edges = [[remap[a], remap[b], lab] for a, b, lab in edges]
    out = {"n_nodes": len(nodes), "n_edges": len(edges),
           "n_laundering_edges": int(sum(e[2] for e in edges)), "edges": edges}
    _write_json(f"{DATA_ROOT}/runs/viz_subgraph.json", out)
    vol.commit()
    print(f"viz subgraph: {out['n_nodes']} nodes, {out['n_edges']} edges, "
          f"{out['n_laundering_edges']} laundering")
    return out


@app.local_entrypoint()
def main(config_json: str = "{}"):
    if config_json.startswith("@"):
        with open(config_json[1:]) as f:
            config_json = f.read()
    cfg = json.loads(config_json)
    action = cfg.get("action", "train")
    if action == "prepare":
        res = prepare_data.remote(cfg)
    elif action == "xgb":
        res = xgboost_baseline.remote(cfg)
    elif action == "export_viz":
        res = export_viz.remote(cfg)
    elif action == "train_batch":
        # cfg["cells"] = list of train configs -> run on parallel L4 containers via .map
        res = list(train.map(cfg["cells"]))
    else:
        res = train.remote(cfg)
    print("RESULT_JSON:" + json.dumps(res))
