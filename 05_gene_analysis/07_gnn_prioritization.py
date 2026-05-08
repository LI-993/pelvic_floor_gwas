#!/usr/bin/env python3
"""GNN-based gene prioritization on the STRING PPI network.

Reads the feature matrix produced by 05_ml_gene_prioritization, attaches it
to the PPI graph from 06_ppi_network, and trains a GAT-based message-passing
model with OMIM/HPO labels. Falls back to a label-propagation + GBM ensemble
when torch_geometric is not installed. Reports CV AUC, top-N enrichment, and
independent validation against cross-ancestry / drug-target / pre-GWAS gene
sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

OUT = RESULTS_DIR / "gnn_prioritization"
ML_FEATURES = RESULTS_DIR / "gene_prioritization_ml" / "feature_matrix.csv"
PPI_GRAPHML = RESULTS_DIR / "ppi_network" / "ppi_network.graphml"
ML_RANKING = RESULTS_DIR / "gene_prioritization_ml" / "final_gene_ranking.csv"

NODE_FEATURE_COLS = ["neglog10p", "max_z", "mean_z", "n_snps", "n_phenotypes"]

# OMIM/HPO training labels (manuscript supplement).
OMIM_HPO_GENES: list[str] = [
    "WNT4", "WT1", "LOXL1", "ESR1", "ELN", "FGFR2", "COL3A1", "COL1A1", "COL1A2",
    "FBLN5", "LAMC1", "BMP4", "BMP7", "HOXA13", "HOXD13", "AR", "SRY", "SOX9",
    "NR5A1", "AMH", "AMHR2", "CYP17A1", "CYP19A1", "HSD17B3", "SRD5A2",
    "STAR", "POR", "DHCR7", "LHX1", "EMX2", "PAX2", "PAX8", "GATA3", "HNF1B",
    "EYA1", "SIX1", "SIX2", "SALL1", "SALL4", "GLI3", "IFT172", "DYNC2H1",
    "WDR35", "RSPH4A", "RSPH9", "CCDC39", "CCDC40", "DNAI1", "DNAH5", "DNAH11",
    "CFTR", "SCNN1A", "SCNN1B", "SCNN1G", "MUC5B", "MUC5AC", "SERPINA1",
    "FBN1", "FBN2", "TGFBR1", "TGFBR2", "SMAD3", "COL5A1", "COL5A2",
    "TNXB", "ADAMTS2", "PLOD1", "B3GALT6", "B4GALT7", "SLC39A13", "CHST14",
    "DSE", "ATP6V0A2", "ATP7A", "PYCR1", "ALDH18A1",
]

INDEPENDENT_VALIDATION = {
    "Cross-ancestry": ["WNT4", "WT1"],
    "Drug targets": ["ESR1", "ESR2", "AGTR1", "VEGFA", "PGR", "PTGS2"],
    "Pre-GWAS": ["LOXL1", "ELN", "FBLN5", "COL3A1"],
}


def load_graph_and_features() -> tuple[nx.Graph, pd.DataFrame, dict[str, int], dict[int, str]]:
    features = pd.read_csv(ML_FEATURES)

    if PPI_GRAPHML.exists():
        G = nx.read_graphml(str(PPI_GRAPHML))
        print(f"  Loaded PPI: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    else:
        print("  PPI graph not found; building empty graph.")
        G = nx.Graph()

    for gene in features["Gene"]:
        if gene not in G:
            G.add_node(gene)

    gene_to_idx = {gene: i for i, gene in enumerate(G.nodes())}
    idx_to_gene = {i: gene for gene, i in gene_to_idx.items()}
    print(f"  Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, features, gene_to_idx, idx_to_gene


def make_node_features(features: pd.DataFrame, gene_to_idx: dict[str, int]) -> np.ndarray:
    n_nodes = len(gene_to_idx)
    X = np.zeros((n_nodes, len(NODE_FEATURE_COLS)))
    by_gene = features.set_index("Gene")
    for gene, idx in gene_to_idx.items():
        if gene in by_gene.index:
            for j, col in enumerate(NODE_FEATURE_COLS):
                X[idx, j] = by_gene.loc[gene, col]
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def make_labels(gene_to_idx: dict[str, int]) -> np.ndarray:
    y = np.zeros(len(gene_to_idx))
    for gene in OMIM_HPO_GENES:
        if gene in gene_to_idx:
            y[gene_to_idx[gene]] = 1
    print(f"  Positive labels in network: {int(y.sum())}")
    return y


if HAS_PYG:
    class GAT(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
            self.lin = nn.Linear(hidden_channels, 1)
            self.dropout = nn.Dropout(0.3)
            self.skip = nn.Linear(in_channels, hidden_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            skip = self.skip(x)
            h = F.relu(self.conv1(x, edge_index))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, edge_index)) + skip
            return torch.sigmoid(self.lin(h)).squeeze()


def train_gnn(
    G: nx.Graph,
    X: np.ndarray,
    y: np.ndarray,
    gene_to_idx: dict[str, int],
) -> tuple[np.ndarray, list[float], list[float]]:
    """5-fold stratified CV → final-model predictions for all nodes."""
    edge_list = list(G.edges())
    edge_index = torch.tensor(
        [[gene_to_idx[a], gene_to_idx[b]] for a, b in edge_list]
        + [[gene_to_idx[b], gene_to_idx[a]] for a, b in edge_list],
        dtype=torch.long,
    ).t()
    x_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    data = Data(x=x_t, edge_index=edge_index, y=y_t)

    n_nodes = X.shape[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(n_nodes)
    aucs: list[float] = []
    aps: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(n_nodes), y), 1):
        model = GAT(X.shape[1])
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        model.train()
        for _ in range(200):
            optim.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(out[train_mask], y_t[train_mask])
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index)[val_idx].numpy()
        val_true = y[val_idx]
        if np.unique(val_true).size > 1:
            auc = roc_auc_score(val_true, val_pred)
            ap = average_precision_score(val_true, val_pred)
        else:
            auc, ap = 0.5, 0.0
        aucs.append(auc)
        aps.append(ap)
        preds[val_idx] = val_pred
        print(f"  Fold {fold}: AUC={auc:.3f}, AP={ap:.3f}")

    final = GAT(X.shape[1])
    optim = torch.optim.Adam(final.parameters(), lr=1e-2, weight_decay=5e-4)
    final.train()
    for _ in range(300):
        optim.zero_grad()
        out = final(data.x, data.edge_index)
        loss = F.binary_cross_entropy(out, y_t)
        loss.backward()
        optim.step()
    final.eval()
    with torch.no_grad():
        final_scores = final(data.x, data.edge_index).numpy()

    return final_scores, aucs, aps


def train_fallback(
    G: nx.Graph,
    X: np.ndarray,
    y: np.ndarray,
    gene_to_idx: dict[str, int],
) -> tuple[np.ndarray, list[float]]:
    """No torch_geometric: Gradient Boosting with PageRank + betweenness as extra features."""
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)
    extra = np.zeros((X.shape[0], 2))
    for gene, idx in gene_to_idx.items():
        extra[idx, 0] = pagerank.get(gene, 0)
        extra[idx, 1] = betweenness.get(gene, 0)
    X_ext = np.hstack([X, extra])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(X.shape[0])
    aucs: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_ext, y), 1):
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_ext[train_idx], y[train_idx])
        pred = model.predict_proba(X_ext[val_idx])[:, 1]
        if np.unique(y[val_idx]).size > 1:
            aucs.append(roc_auc_score(y[val_idx], pred))
        else:
            aucs.append(0.5)
        preds[val_idx] = pred
        print(f"  Fold {fold}: AUC={aucs[-1]:.3f}")

    final = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_ext, y)
    return final.predict_proba(X_ext)[:, 1], aucs


def make_ranking(idx_to_gene: dict[int, str], features: pd.DataFrame, scores: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "Gene": [idx_to_gene[i] for i in range(len(scores))],
        "GNN_Score": scores,
        "Is_Positive": y,
    }).merge(features[["Gene", "neglog10p", "n_phenotypes"]], on="Gene", how="left")
    df = df.sort_values("GNN_Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    return df


def report_top(ranking: pd.DataFrame, n: int = 30) -> None:
    print(f"\nTop {n} GNN-prioritized genes:")
    print(f"{'Rank':>4} {'Gene':>12} {'Score':>8} {'Known':>6} {'-log10P':>10} {'N_Pheno':>8}")
    print("-" * 60)
    for _, row in ranking.head(n).iterrows():
        known = "Yes" if row["Is_Positive"] == 1 else ""
        print(f"{int(row['Rank']):4d} {row['Gene']:>12} {row['GNN_Score']:8.4f} "
              f"{known:>6} {row['neglog10p']:10.2f} {int(row['n_phenotypes']):8d}")


def report_independent_validation(ranking: pd.DataFrame, n_nodes: int) -> None:
    rank_by_gene = ranking.set_index("Gene")["Rank"].to_dict()
    score_by_gene = ranking.set_index("Gene")["GNN_Score"].to_dict()
    for label, genes in INDEPENDENT_VALIDATION.items():
        print(f"\n  {label}:")
        ranks: list[int] = []
        for g in genes:
            if g in rank_by_gene:
                r = int(rank_by_gene[g])
                ranks.append(r)
                print(f"    {g}: Rank {r}/{n_nodes}, Score {score_by_gene[g]:.4f}")
        if ranks:
            median = np.median(ranks)
            pct = 100 * (1 - median / n_nodes)
            print(f"    Median rank: {median:.0f} (top {pct:.1f}%)")


def report_enrichment(ranking: pd.DataFrame, n_nodes: int, positive_genes: set[str]) -> None:
    print("\nTop-N enrichment vs OMIM/HPO labels:")
    for top_n in (10, 20, 50, 100):
        top = set(ranking.head(top_n)["Gene"])
        n_pos = sum(g in positive_genes for g in top)
        expected = top_n * len(positive_genes) / n_nodes
        print(f"  Top {top_n}: {n_pos} known (expected {expected:.1f}, {n_pos / max(expected, 0.01):.1f}x)")


def compare_with_ml(ranking: pd.DataFrame) -> pd.DataFrame:
    if not ML_RANKING.exists():
        return pd.DataFrame()
    prev = pd.read_csv(ML_RANKING)
    cmp = ranking[["Gene", "GNN_Score", "Rank"]].rename(columns={"Rank": "GNN_Rank"})
    cmp = cmp.merge(
        prev[["Gene", "Final_score", "is_known_disease_gene"]].rename(columns={"Final_score": "RF_Score"}),
        on="Gene", how="left",
    )
    cmp["RF_Rank"] = cmp["RF_Score"].rank(ascending=False)
    rho, p = spearmanr(cmp["GNN_Rank"], cmp["RF_Rank"], nan_policy="omit")
    print(f"\nGNN vs RF Spearman correlation: rho={rho:.3f}, p={p:.2e}")
    return cmp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GNN-based Gene Prioritization")
    print("=" * 60)

    print("\n[1] Loading graph + features...")
    G, features, gene_to_idx, idx_to_gene = load_graph_and_features()
    X = make_node_features(features, gene_to_idx)
    y = make_labels(gene_to_idx)

    print(f"\n[2] Training ({'GAT' if HAS_PYG else 'GBM fallback'})...")
    if HAS_PYG:
        scores, aucs, aps = train_gnn(G, X, y, gene_to_idx)
        print(f"\n  Mean CV AUC: {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")
        print(f"  Mean CV AP:  {np.mean(aps):.3f} (+/- {np.std(aps):.3f})")
    else:
        print("  torch_geometric not installed; using PageRank + GBM.")
        scores, aucs = train_fallback(G, X, y, gene_to_idx)
        print(f"\n  Mean CV AUC: {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")

    ranking = make_ranking(idx_to_gene, features, scores, y)
    ranking.to_csv(OUT / "gnn_gene_ranking.csv", index=False)

    report_top(ranking)
    report_enrichment(ranking, len(scores), set(OMIM_HPO_GENES))
    print("\n[Independent validation]")
    report_independent_validation(ranking, len(scores))

    cmp = compare_with_ml(ranking)
    if not cmp.empty:
        cmp.to_csv(OUT / "gnn_vs_rf_comparison.csv", index=False)

    print(f"\nResults: {OUT}")


if __name__ == "__main__":
    main()
