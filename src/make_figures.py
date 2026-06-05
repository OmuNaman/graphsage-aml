"""Generate publication figures for graphsage-aml from REAL data (ledger, probs, exported subgraph).
Fig 1 (real SAML-D subgraph), Fig 3 (main bars), Fig 4 (PR curves), Fig 5 (arch search),
Fig 6 (focal ablation), Fig 7 (threshold). Fig 2 (schematic) is done via PaperBanana separately."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/graphsage-aml
EXP = os.path.join(ROOT, "experiments")
RUNS = os.path.join(EXP, "runs")
ANA = os.path.join(EXP, "analysis")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
C = {"xgb": "#7f7f7f", "bce": "#2166ac", "foc": "#d6604d", "launder": "#d62728", "normal": "#b0b0b0"}


def load_json(p):
    return json.load(open(p))


def run(eid):
    return load_json(os.path.join(RUNS, eid + ".json"))


# ============ FIG 1: real SAML-D laundering subgraph (networkx) ============
def fig1():
    import networkx as nx
    sg = load_json(os.path.join(RUNS, "viz_subgraph.json"))
    # aggregate multi-edges: count per (u,v); laundering if any laundering tx on the pair
    from collections import defaultdict
    cnt = defaultdict(int); lab = defaultdict(int)
    for u, v, y in sg["edges"]:
        key = tuple(sorted((u, v))); cnt[key] += 1; lab[key] = max(lab[key], y)
    G = nx.Graph()
    for (u, v), c in cnt.items():
        G.add_edge(u, v, count=c, laundering=lab[(u, v)])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    pos = nx.spring_layout(G, seed=3, k=0.55, iterations=120)
    deg = dict(G.degree())
    node_sizes = [60 + 90 * deg[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#eaf2fb",
                           edgecolors="#2166ac", linewidths=1.0, ax=ax)
    laund = [(u, v) for u, v, d in G.edges(data=True) if d["laundering"] == 1]
    norm = [(u, v) for u, v, d in G.edges(data=True) if d["laundering"] == 0]
    nx.draw_networkx_edges(G, pos, edgelist=norm, edge_color=C["normal"], width=1.0, alpha=0.6, ax=ax)
    widths = [1.2 + 0.6 * G[u][v]["count"] for u, v in laund]
    nx.draw_networkx_edges(G, pos, edgelist=laund, edge_color=C["launder"], width=widths, alpha=0.9, ax=ax)
    ax.set_title("A real SAML-D subgraph: laundering forms multi-hop, repeated-transfer motifs\n"
                 f"({G.number_of_nodes()} accounts, {sg['n_edges']} transactions, "
                 f"{sg['n_laundering_edges']} laundering)", fontsize=11)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=C["launder"], lw=3, label="laundering transaction"),
                       Line2D([0], [0], color=C["normal"], lw=2, label="normal transaction"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="#eaf2fb",
                              markeredgecolor="#2166ac", markersize=10, label="account (size = degree)")],
              loc="lower left", frameon=True, framealpha=0.95)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_saml_subgraph.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============ FIG 3: main results bars ============
def fig3():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ANA, "main_comparison.csv"))))
    names = [r["model"] for r in rows]
    prauc = [float(r["pr_auc_mean"]) for r in rows]; prauc_sd = [float(r["pr_auc_std"]) for r in rows]
    prec = [float(r["prec70_mean"]) for r in rows]; prec_sd = [float(r["prec70_std"]) for r in rows]
    colors = [C["xgb"], C["bce"], C["foc"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    x = np.arange(len(names))
    axes[0].bar(x, prauc, yerr=prauc_sd, capsize=4, color=colors)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=12, ha="right")
    axes[0].set_ylabel("Test PR-AUC"); axes[0].set_title("PR-AUC (higher = better)")
    for i, v in enumerate(prauc):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].bar(x, prec, yerr=prec_sd, capsize=4, color=colors)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=12, ha="right")
    axes[1].set_ylabel("Precision @ recall ≥ 0.70"); axes[1].set_title("Precision at recall ≥ 0.70")
    for i, v in enumerate(prec):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("GraphSAGE vs. XGBoost on SAML-D (disjoint-node test, mean ± std)",
                 fontsize=12.5, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_main_results.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============ FIG 4: precision-recall curves ============
def fig4():
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    series = [("xgb-s0", "XGBoost", C["xgb"], "-"),
              ("bce-b2", "GraphSAGE + weighted BCE", C["bce"], "-"),
              ("foc-b", "GraphSAGE + Focal", C["foc"], "-")]
    for eid, lab, col, ls in series:
        z = np.load(os.path.join(RUNS, eid + "_probs.npz"))
        y, p = z["y"], z["prob"]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, color=col, ls=ls, lw=2, label=f"{lab} (PR-AUC {ap:.3f})")
    base = float(np.load(os.path.join(RUNS, "xgb-s0_probs.npz"))["y"].mean())
    ax.axhline(base, color="k", ls=":", lw=1, label=f"base rate ({base:.3f})")
    ax.axvspan(0.70, 1.0, color="#1a9850", alpha=0.08)
    ax.text(0.85, 0.92, "recall ≥ 0.70\n(target region)", fontsize=8.5, color="#1a9850", ha="center")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall on the disjoint-node test set")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_pr_curves.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============ FIG 5: BCE architecture search ============
def fig5():
    s = load_json(os.path.join(ANA, "summary.json"))["bce_arch_search"]
    s = sorted(s, key=lambda r: r["exp_id"])
    labels = [f"{r['sage_layers']}L/{r['hidden']}h\n{r['exp_id']}" for r in s]
    val = [r["val_pr_auc"] for r in s]; test = [r["test_pr_auc"] for r in s]
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - w / 2, val, w, color="#92c5de", label="val PR-AUC")
    ax.bar(x + w / 2, test, w, color="#2166ac", label="test PR-AUC")
    best = max(range(len(s)), key=lambda i: s[i]["val_pr_auc"])
    ax.scatter([best], [val[best] + 0.02], marker="*", s=200, color="#fdae61", edgecolor="k", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1.05)
    ax.set_title("GraphSAGE architecture search (weighted BCE)\n★ = best by val PR-AUC (3-layer encoder)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_arch_search.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============ FIG 6: focal-vs-BCE ablation ============
def fig6():
    summ = load_json(os.path.join(ANA, "summary.json"))
    foc = sorted(summ["focal_search"], key=lambda r: (r["alpha"], r["gamma"]))
    labels = [f"α={r['alpha']}\nγ={r['gamma']}" for r in foc]
    prec = [r["prec70"] if r["prec70"] is not None else 0 for r in foc]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(len(foc))
    ax.bar(x, prec, 0.6, color=C["foc"], label="focal: precision @ recall ≥ 0.70")
    for i, (r, pv) in enumerate(zip(foc, prec)):
        if r["prec70"] is None:
            ax.text(i, 0.02, "n/a\n(recall<0.70)", ha="center", va="bottom", fontsize=7.5, color="#555")
    bce_prec = summ["graphsage_bce"]["test_precision70"]["mean"]
    bce_sd = summ["graphsage_bce"]["test_precision70"]["std"]
    ax.axhline(bce_prec, color=C["bce"], ls="--", lw=1.8, label=f"weighted BCE ({bce_prec:.3f})")
    ax.axhspan(bce_prec - bce_sd, bce_prec + bce_sd, color=C["bce"], alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Precision @ recall ≥ 0.70")
    p = summ["focal_vs_bce_prec70"]["p"]
    ax.set_title(f"Focal-loss grid vs. weighted-BCE reference\n(focal ≈ BCE; difference not significant, p={p})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_focal_vs_bce.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============ FIG 7: threshold operating-point curve (best GraphSAGE) ============
def fig7():
    sweep = run("bce-b2")["test"]["sweep"]
    thr = [r["thr"] for r in sweep]; rec = [r["recall"] for r in sweep]
    prec = [r["precision"] for r in sweep]; f1 = [r["f1"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(thr, rec, "-o", ms=4, color="#1a9850", label="recall")
    ax.plot(thr, prec, "-s", ms=4, color=C["bce"], label="precision")
    ax.plot(thr, f1, "-^", ms=4, color=C["foc"], label="F1")
    ax.axhline(0.70, color="k", ls=":", lw=1)
    # operating point = max precision among thresholds with recall>=0.70 (matches reported precision@.70)
    cand = [(t, p) for t, r, p in zip(thr, rec, prec) if r >= 0.70]
    if cand:
        op, op_prec = max(cand, key=lambda tp: tp[1])
        ax.axvline(op, color="#fdae61", ls="--", lw=1.5)
        ax.text(op - 0.21, 0.22, f"recall≥0.70 operating\npoint (thr={op},\nprecision={op_prec:.2f})",
                fontsize=8.5, color="#b8860b")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold sweep — best GraphSAGE (3-layer, weighted BCE)")
    ax.legend(loc="center right"); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_threshold.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for fn in [fig1, fig3, fig4, fig5, fig6, fig7]:
        fn(); print(fn.__name__, "ok")
    print("FIGURES ->", OUT)
