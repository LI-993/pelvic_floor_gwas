#!/usr/bin/env python3
"""GAT + GraphSAGE + GCN ensemble for gene prioritization.

Trains three GNN architectures with skip connections, averages their
predictions, then blends with a GradientBoosting classifier (70/30 split)
that gets PageRank + betweenness as additional features. Reports 5-fold CV
AUC and writes the ensemble ranking. Requires torch + torch_geometric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

OUT = RESULTS_DIR / "gnn_prioritization"
ML_FEATURES = RESULTS_DIR / "gene_prioritization_ml" / "feature_matrix.csv"
PPI_GRAPHML = RESULTS_DIR / "ppi_network" / "ppi_network.graphml"

NODE_FEATURE_COLS = ["neglog10p", "max_z", "mean_z", "n_snps", "n_phenotypes"]

# Same labels as 07_gnn_prioritization.
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

INDEPENDENT_SETS = {
    "Cross-ancestry": ["WNT4", "WT1"],
    "Drug targets": ["ESR1", "ESR2", "AGTR1", "VEGFA", "PGR", "PTGS2"],
    "Pre-GWAS": ["LOXL1", "ELN", "FBLN5", "COL3A1"],
}


class GAT_Skip(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int = 64) -> None:
        super().__init__()
        self.conv1 = GATConv(in_ch, hid_ch, heads=4, concat=False)
        self.conv2 = GATConv(hid_ch, hid_ch, heads=4, concat=False)
        self.skip = nn.Linear(in_ch, hid_ch)
        self.lin = nn.Linear(hid_ch, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index)) + skip
        return torch.sigmoid(self.lin(x)).squeeze()


class SAGE_Skip(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int = 64) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid_ch)
        self.conv2 = SAGEConv(hid_ch, hid_ch)
        self.skip = nn.Linear(in_ch, hid_ch)
        self.lin = nn.Linear(hid_ch, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index)) + skip
        return torch.sigmoid(self.lin(x)).squeeze()


class GCN_Deep(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int = 64) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, hid_ch)
        self.conv3 = GCNConv(hid_ch, hid_ch // 2)
        self.skip = nn.Linear(in_ch, hid_ch // 2)
        self.lin = nn.Linear(hid_ch // 2, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index)) + skip
        return torch.sigmoid(self.lin(x)).squeeze()


MODELS: list[tuple[str, type[nn.Module]]] = [
    ("GAT", GAT_Skip),
    ("SAGE", SAGE_Skip),
    ("GCN", GCN_Deep),
]


def load_data() -> tuple[nx.Graph, pd.DataFrame, dict[str, int], dict[int, str]]:
    features = pd.read_csv(ML_FEATURES)
    G = nx.read_graphml(str(PPI_GRAPHML))
    for gene in features["Gene"]:
        if gene not in G:
            G.add_node(gene)
    gene_to_idx = {gene: i for i, gene in enumerate(G.nodes())}
    idx_to_gene = {i: gene for gene, i in gene_to_idx.items()}
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, features, gene_to_idx, idx_to_gene


def build_inputs(
    G: nx.Graph,
    features: pd.DataFrame,
    gene_to_idx: dict[str, int],
) -> tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = G.number_of_nodes()
    X = np.zeros((n_nodes, len(NODE_FEATURE_COLS)))
    by_gene = features.set_index("Gene")
    for gene, idx in gene_to_idx.items():
        if gene in by_gene.index:
            for j, col in enumerate(NODE_FEATURE_COLS):
                X[idx, j] = by_gene.loc[gene, col]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    y = np.zeros(n_nodes)
    for gene in OMIM_HPO_GENES:
        if gene in gene_to_idx:
            y[gene_to_idx[gene]] = 1
    print(f"Positive samples: {int(y.sum())}")

    edge_list = list(G.edges())
    edge_index = torch.tensor(
        [[gene_to_idx[a], gene_to_idx[b]] for a, b in edge_list]
        + [[gene_to_idx[b], gene_to_idx[a]] for a, b in edge_list],
        dtype=torch.long,
    ).t()
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))

    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)
    extra = np.zeros((n_nodes, 2))
    for gene, idx in gene_to_idx.items():
        extra[idx, 0] = pagerank.get(gene, 0)
        extra[idx, 1] = betweenness.get(gene, 0)
    X_gb = np.hstack([X, extra])

    return data, X, X_gb, y


def train_one_gnn(model_cls: type[nn.Module], data: Data, train_mask: torch.Tensor, epochs: int = 400, seed: int = 42) -> np.ndarray:
    torch.manual_seed(seed)
    model = model_cls(data.x.shape[1], 64)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index).numpy()


def cross_validate(data: Data, X_gb: np.ndarray, y: np.ndarray) -> tuple[list[float], np.ndarray]:
    n_nodes = X_gb.shape[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_aucs: list[float] = []
    last_predictions = np.zeros(n_nodes)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(n_nodes), y), 1):
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        gnn_scores = [
            train_one_gnn(cls, data, train_mask, seed=42 + fold)
            for _, cls in MODELS
        ]
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42).fit(X_gb[train_idx], y[train_idx])
        gb_pred = gb.predict_proba(X_gb)[:, 1]

        ensemble = 0.7 * np.mean(gnn_scores, axis=0) + 0.3 * gb_pred
        last_predictions[val_idx] = ensemble[val_idx]

        val_true = y[val_idx]
        if np.unique(val_true).size > 1:
            auc = roc_auc_score(val_true, ensemble[val_idx])
            print(f"  Fold {fold}: AUC = {auc:.3f}")
            all_aucs.append(auc)

    return all_aucs, last_predictions


def train_final(data: Data, X_gb: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_nodes = X_gb.shape[0]
    full_mask = torch.ones(n_nodes, dtype=torch.bool)
    gnn_scores = [train_one_gnn(cls, data, full_mask) for _, cls in MODELS]
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42).fit(X_gb, y)
    gb_pred = gb.predict_proba(X_gb)[:, 1]
    return 0.7 * np.mean(gnn_scores, axis=0) + 0.3 * gb_pred


def report_independent(ranking: pd.DataFrame, n_nodes: int) -> None:
    rank_by_gene = ranking.set_index("Gene")["Rank"].to_dict()
    for label, genes in INDEPENDENT_SETS.items():
        print(f"\n  {label}:")
        for g in genes:
            if g in rank_by_gene:
                print(f"    {g}: Rank {int(rank_by_gene[g])}/{n_nodes}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GNN Ensemble (GAT + SAGE + GCN) + GBM")
    print("=" * 60)

    G, features, gene_to_idx, idx_to_gene = load_data()
    data, _X, X_gb, y = build_inputs(G, features, gene_to_idx)

    print("\n[1] Cross-validation...")
    aucs, _ = cross_validate(data, X_gb, y)
    print(f"\nMean CV AUC: {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")

    print("\n[2] Final model on full data...")
    final_scores = train_final(data, X_gb, y)

    ranking = pd.DataFrame({
        "Gene": [idx_to_gene[i] for i in range(len(final_scores))],
        "Ensemble_Score": final_scores,
        "Is_Positive": y,
    }).merge(features[["Gene", "neglog10p", "n_phenotypes"]], on="Gene", how="left")
    ranking = ranking.sort_values("Ensemble_Score", ascending=False)
    ranking["Rank"] = range(1, len(ranking) + 1)

    print("\nTop 15:")
    print(f"{'Rank':>4} {'Gene':>12} {'Score':>8} {'Known':>6} {'-log10P':>10}")
    print("-" * 50)
    for _, row in ranking.head(15).iterrows():
        marker = "Yes" if row["Is_Positive"] == 1 else ""
        print(f"{int(row['Rank']):4d} {row['Gene']:>12} {row['Ensemble_Score']:8.4f} {marker:>6} {row['neglog10p']:10.2f}")

    print("\n[Independent validation]")
    report_independent(ranking, G.number_of_nodes())

    out_path = OUT / "ensemble_ranking.csv"
    ranking.to_csv(out_path, index=False)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
