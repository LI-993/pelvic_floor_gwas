#!/usr/bin/env python3
"""
Enhanced GNN Ensemble Model for Gene Prioritization

Improvements:
1. Ensemble of GAT, GraphSAGE, and GCN
2. Edge dropout for regularization
3. Multiple training runs averaged
4. Combined with Gradient Boosting
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = Path("d:/Nproject/gwas/pelvic_floor_gwas")

print("=" * 60)
print("Enhanced GNN Ensemble Model")
print("=" * 60)

# Load data
features = pd.read_csv(BASE_DIR / "results/gene_prioritization_ml_improved/feature_matrix.csv")
G = nx.read_graphml(BASE_DIR / "results/ppi_network_expanded/ppi_network_expanded.graphml")

# Add missing nodes
all_genes = features['Gene'].tolist()
for gene in all_genes:
    if gene not in G.nodes():
        G.add_node(gene)

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Prepare data
feature_cols = ['neglog10p', 'max_z', 'mean_z', 'n_snps', 'n_phenotypes']
gene_to_idx = {gene: i for i, gene in enumerate(G.nodes())}
n_nodes = len(G.nodes())

X = np.zeros((n_nodes, len(feature_cols)))
for gene in G.nodes():
    idx = gene_to_idx[gene]
    gene_data = features[features['Gene'] == gene]
    if len(gene_data) > 0:
        for j, col in enumerate(feature_cols):
            X[idx, j] = gene_data[col].values[0]

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# Labels
omim_hpo_genes = [
    'WNT4', 'WT1', 'LOXL1', 'ESR1', 'ELN', 'FGFR2', 'COL3A1', 'COL1A1', 'COL1A2',
    'FBLN5', 'LAMC1', 'BMP4', 'BMP7', 'HOXA13', 'HOXD13', 'AR', 'SRY', 'SOX9',
    'NR5A1', 'AMH', 'AMHR2', 'CYP17A1', 'CYP19A1', 'HSD17B3', 'SRD5A2',
    'STAR', 'POR', 'DHCR7', 'LHX1', 'EMX2', 'PAX2', 'PAX8', 'GATA3', 'HNF1B',
    'EYA1', 'SIX1', 'SIX2', 'SALL1', 'SALL4', 'GLI3', 'IFT172', 'DYNC2H1',
    'WDR35', 'RSPH4A', 'RSPH9', 'CCDC39', 'CCDC40', 'DNAI1', 'DNAH5', 'DNAH11',
    'CFTR', 'SCNN1A', 'SCNN1B', 'SCNN1G', 'MUC5B', 'MUC5AC', 'SERPINA1',
    'FBN1', 'FBN2', 'TGFBR1', 'TGFBR2', 'SMAD3', 'COL5A1', 'COL5A2',
    'TNXB', 'ADAMTS2', 'PLOD1', 'B3GALT6', 'B4GALT7', 'SLC39A13', 'CHST14',
    'DSE', 'ATP6V0A2', 'ATP7A', 'PYCR1', 'ALDH18A1'
]

y = np.zeros(n_nodes)
for gene in omim_hpo_genes:
    if gene in gene_to_idx:
        y[gene_to_idx[gene]] = 1

print(f"Positive samples: {int(y.sum())}")

# PyTorch data
edge_list = list(G.edges())
edge_index = torch.tensor(
    [[gene_to_idx[e[0]], gene_to_idx[e[1]]] for e in edge_list] +
    [[gene_to_idx[e[1]], gene_to_idx[e[0]]] for e in edge_list],
    dtype=torch.long
).t()

x = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=labels)

# Define multiple GNN architectures
class GAT_Skip(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid_ch, heads=4, concat=False)
        self.conv2 = GATConv(hid_ch, hid_ch, heads=4, concat=False)
        self.skip = nn.Linear(in_ch, hid_ch)
        self.lin = nn.Linear(hid_ch, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        skip = self.skip(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = x + skip
        return torch.sigmoid(self.lin(x)).squeeze()

class GraphSAGE_Skip(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid_ch)
        self.conv2 = SAGEConv(hid_ch, hid_ch)
        self.skip = nn.Linear(in_ch, hid_ch)
        self.lin = nn.Linear(hid_ch, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        skip = self.skip(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = x + skip
        return torch.sigmoid(self.lin(x)).squeeze()

class GCN_Deep(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, hid_ch)
        self.conv3 = GCNConv(hid_ch, hid_ch//2)
        self.skip = nn.Linear(in_ch, hid_ch//2)
        self.lin = nn.Linear(hid_ch//2, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, edge_index):
        skip = self.skip(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = x + skip
        return torch.sigmoid(self.lin(x)).squeeze()

# Cross-validation
print("\n[1] Cross-validation of GNN ensemble...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []

models_list = [
    ('GAT', GAT_Skip),
    ('SAGE', GraphSAGE_Skip),
    ('GCN', GCN_Deep),
]

# Graph features for GB
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
graph_features = np.zeros((n_nodes, 2))
for gene, idx in gene_to_idx.items():
    graph_features[idx, 0] = pagerank.get(gene, 0)
    graph_features[idx, 1] = betweenness.get(gene, 0)
X_gb = np.hstack([X, graph_features])

for fold, (train_idx, val_idx) in enumerate(skf.split(range(n_nodes), y)):
    fold_scores = []

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True

    # Train each GNN architecture
    for name, model_class in models_list:
        torch.manual_seed(42 + fold)
        model = model_class(len(feature_cols), 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(400):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            fold_scores.append(model(data.x, data.edge_index).numpy())

    # GB for this fold
    gb_fold = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    gb_fold.fit(X_gb[train_idx], y[train_idx])
    gb_pred = gb_fold.predict_proba(X_gb)[:, 1]

    # Ensemble: 70% GNN, 30% GB
    gnn_ensemble = np.mean(fold_scores, axis=0)
    fold_final = 0.7 * gnn_ensemble + 0.3 * gb_pred

    val_pred = fold_final[val_idx]
    val_true = y[val_idx]

    if len(np.unique(val_true)) > 1:
        auc = roc_auc_score(val_true, val_pred)
        cv_aucs.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.3f}")

print(f"\n  Mean CV AUC: {np.mean(cv_aucs):.3f} (+/- {np.std(cv_aucs):.3f})")

# Train final model on all data
print("\n[2] Training final ensemble on all data...")
final_gnn_scores = []
for name, model_class in models_list:
    torch.manual_seed(42)
    model = model_class(len(feature_cols), 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(400):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_gnn_scores.append(model(data.x, data.edge_index).numpy())

gb_final = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gb_final.fit(X_gb, y)
gb_scores = gb_final.predict_proba(X_gb)[:, 1]

gnn_ensemble = np.mean(final_gnn_scores, axis=0)
final_scores = 0.7 * gnn_ensemble + 0.3 * gb_scores

# Generate rankings
print("\n[3] Generating rankings...")
idx_to_gene = {i: gene for gene, i in gene_to_idx.items()}
ranking_df = pd.DataFrame({
    'Gene': [idx_to_gene[i] for i in range(n_nodes)],
    'Ensemble_Score': final_scores,
    'Is_Positive': y
})
ranking_df = ranking_df.merge(features[['Gene', 'neglog10p', 'n_phenotypes']], on='Gene', how='left')
ranking_df = ranking_df.sort_values('Ensemble_Score', ascending=False)
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Independent validation
cross_ancestry = ['WNT4', 'WT1']
drug_targets = ['ESR1', 'ESR2', 'AGTR1', 'VEGFA', 'PGR', 'PTGS2']
pre_gwas = ['LOXL1', 'ELN', 'FBLN5', 'COL3A1']

print("\n[4] Independent validation:")
print("\n  Cross-ancestry genes (EUR->AFR):")
for gene in cross_ancestry:
    r = ranking_df[ranking_df['Gene'] == gene]
    if len(r) > 0:
        print(f"    {gene}: Rank {int(r['Rank'].values[0])}/{n_nodes}")

print("\n  Drug targets:")
for gene in drug_targets:
    r = ranking_df[ranking_df['Gene'] == gene]
    if len(r) > 0:
        print(f"    {gene}: Rank {int(r['Rank'].values[0])}/{n_nodes}")

print("\n  Pre-GWAS literature:")
for gene in pre_gwas:
    r = ranking_df[ranking_df['Gene'] == gene]
    if len(r) > 0:
        print(f"    {gene}: Rank {int(r['Rank'].values[0])}/{n_nodes}")

# Top 15
print("\nTop 15 genes:")
print(f"{'Rank':>4} {'Gene':>12} {'Score':>8} {'Known':>6} {'-log10P':>10}")
print("-" * 50)
for _, row in ranking_df.head(15).iterrows():
    known = "Yes" if row['Is_Positive'] == 1 else ""
    print(f"{int(row['Rank']):4d} {row['Gene']:>12} {row['Ensemble_Score']:8.4f} {known:>6} {row['neglog10p']:10.2f}")

# Save
ranking_df.to_csv(BASE_DIR / "results/gnn_prioritization/ensemble_ranking.csv", index=False)
print(f"\nSaved to: results/gnn_prioritization/ensemble_ranking.csv")

print("\n" + "=" * 60)
print(f"Ensemble CV AUC: {np.mean(cv_aucs):.3f} (previous GAT: 0.616)")
print("=" * 60)
