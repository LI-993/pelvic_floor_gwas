#!/usr/bin/env python3
"""Drug-repurposing visualizations.

Generates: drug-gene interaction network (top 30 by priority), priority bar
chart, interaction-type pie + per-phenotype bar, gene-drug heatmap, source
distribution.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPE_COLORS, RESULTS_DIR
from utils.plotting import setup_publication_style

DRUG_DIR = RESULTS_DIR / "drug_repurposing"
OUT = FIGURES_DIR / "drug_repurposing"

INTERACTION_COLORS: dict[str, str] = {
    "inhibitor": "#E64B35",
    "agonist": "#4DBBD5",
    "antagonist": "#00A087",
    "modulator": "#3C5488",
    "other": "#888888",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    prioritized = pd.read_csv(DRUG_DIR / "prioritized_candidates.csv")
    inter_path = DRUG_DIR / "dgidb_interactions.csv"
    interactions = pd.read_csv(inter_path) if inter_path.exists() else None
    return prioritized, interactions


def plot_drug_gene_network(prioritized: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))
    G = nx.Graph()
    top = prioritized.nlargest(30, "priority_score")

    for gene in top["gene_symbol"].unique():
        G.add_node(gene, node_type="gene")
    for _, row in top.iterrows():
        G.add_node(row["drug"], node_type="drug", interaction=row.get("interaction_type", "other"))
        G.add_edge(row["drug"], row["gene_symbol"], weight=row["priority_score"])

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
    drug_colors = [
        INTERACTION_COLORS.get(G.nodes[n].get("interaction", "other") or "other", "#888888")
        for n in drug_nodes
    ]

    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color="#E64B35", node_size=1000, node_shape="s", alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color=drug_colors, node_size=500, node_shape="o", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[d.get("weight", 1) / 5 for _, _, d in G.edges(data=True)], alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: n for n in gene_nodes}, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_labels(G, pos, {n: n[:15] + "..." if len(n) > 15 else n for n in drug_nodes}, font_size=7, ax=ax)

    ax.legend(handles=[
        Patch(facecolor="#E64B35", label="Gene (target)", alpha=0.9),
        Patch(facecolor=INTERACTION_COLORS["inhibitor"], label="Inhibitor"),
        Patch(facecolor=INTERACTION_COLORS["agonist"], label="Agonist"),
        Patch(facecolor=INTERACTION_COLORS["other"], label="Other"),
    ], loc="upper left", fontsize=10)
    ax.set_title("Drug-Gene Interaction Network\n(Top 30 by Priority Score)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"drug_gene_network.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: drug_gene_network.png/pdf")


def plot_priority_bar(prioritized: pd.DataFrame) -> None:
    drug_priority = (
        prioritized.groupby("drug")
        .agg({"priority_score": "max", "gene_symbol": lambda x: ", ".join(x.unique()[:3]),
              "phenotypes": "first", "interaction_type": "first"})
        .reset_index()
        .nlargest(25, "priority_score")
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = [
        INTERACTION_COLORS.get(itype if pd.notna(itype) else "other", "#888888")
        for itype in drug_priority["interaction_type"]
    ]
    ax.barh(np.arange(len(drug_priority)), drug_priority["priority_score"], color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(np.arange(len(drug_priority)))
    ax.set_yticklabels(drug_priority["drug"])
    ax.invert_yaxis()
    for i, (score, genes) in enumerate(zip(drug_priority["priority_score"], drug_priority["gene_symbol"])):
        ax.text(score + 0.3, i, f"→ {genes}", va="center", fontsize=8, alpha=0.8)
    ax.set_xlabel("Priority Score", fontsize=12)
    ax.set_title("Top 25 Drug Repurposing Candidates", fontsize=14, fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor=INTERACTION_COLORS["inhibitor"], label="Inhibitor"),
        Patch(facecolor=INTERACTION_COLORS["agonist"], label="Agonist"),
        Patch(facecolor=INTERACTION_COLORS["other"], label="Other"),
    ], loc="lower right", fontsize=10)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"drug_priority_bar.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: drug_priority_bar.png/pdf")


def plot_interaction_distribution(prioritized: pd.DataFrame) -> None:
    interactions = prioritized["interaction_type"].fillna("unknown").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].pie(
        interactions.values,
        labels=interactions.index,
        colors=[INTERACTION_COLORS.get(t, "#888888") for t in interactions.index],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.8,
    )
    axes[0].set_title("Drug-Gene Interaction Types", fontweight="bold")

    pheno_counts: dict[str, int] = {}
    for phenos in prioritized["phenotypes"].dropna():
        for p in str(phenos).split(", "):
            p = p.strip()
            if p and p != "nan":
                pheno_counts[p] = pheno_counts.get(p, 0) + 1
    if pheno_counts:
        labels = list(pheno_counts)
        counts = [pheno_counts[p] for p in labels]
        colors = [PHENOTYPE_COLORS.get(p, "#888888") for p in labels]
        bars = axes[1].barh(labels, counts, color=colors, alpha=0.8)
        axes[1].set_xlabel("Number of Interactions", fontsize=12)
        axes[1].set_title("Drug Candidates by Target Phenotype", fontweight="bold")
        for bar, count in zip(bars, counts):
            axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=10)

    plt.suptitle("Drug Repurposing Analysis Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"drug_interaction_distribution.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: drug_interaction_distribution.png/pdf")


def plot_gene_drug_heatmap(prioritized: pd.DataFrame) -> None:
    top_genes = prioritized.groupby("gene_symbol")["priority_score"].sum().nlargest(15).index
    top_drugs = prioritized.groupby("drug")["priority_score"].sum().nlargest(20).index
    matrix = pd.DataFrame(0, index=top_genes, columns=top_drugs)
    for _, r in prioritized.iterrows():
        if r["gene_symbol"] in top_genes and r["drug"] in top_drugs:
            matrix.loc[r["gene_symbol"], r["drug"]] = r["priority_score"]
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    if matrix.empty:
        print("  Heatmap: no data")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Priority Score"}, ax=ax)
    ax.set_xlabel("Drug")
    ax.set_ylabel("Target Gene")
    ax.set_title("Gene-Drug Interaction Heatmap", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"drug_gene_heatmap.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: drug_gene_heatmap.png/pdf")


def plot_source_distribution(prioritized: pd.DataFrame) -> None:
    sources: list[str] = []
    for s in prioritized["sources"].dropna():
        sources.extend(t.strip()[:30] for t in str(s).split(", "))
    counts = dict(sorted(Counter(sources).items(), key=lambda x: -x[1])[:15])

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(np.arange(len(counts)), list(counts.values()), color="#3C5488", alpha=0.8, edgecolor="white")
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(list(counts))
    ax.invert_yaxis()
    ax.set_xlabel("Number of Interactions", fontsize=12)
    ax.set_title("Drug-Gene Interaction Sources (Top 15)", fontsize=14, fontweight="bold")
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"drug_source_distribution.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: drug_source_distribution.png/pdf")


def main() -> None:
    setup_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Drug Repurposing Visualization")
    print("=" * 60)

    prioritized, interactions = load_data()
    print(f"\nCandidates: {len(prioritized)}")
    if interactions is not None:
        print(f"Interactions: {len(interactions)}")

    plot_drug_gene_network(prioritized)
    plot_priority_bar(prioritized)
    plot_interaction_distribution(prioritized)
    plot_gene_drug_heatmap(prioritized)
    plot_source_distribution(prioritized)
    print(f"\nAll figures saved to: {OUT}")


if __name__ == "__main__":
    main()
