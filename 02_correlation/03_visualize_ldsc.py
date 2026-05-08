#!/usr/bin/env python3
"""LDSC genetic-correlation visualizations.

Produces a heatmap with significance stars, a phenotype-correlation network,
heritability bar chart, and a hierarchically clustered correlation map.
Inputs come from the parsed summary table written by 02_parse_ldsc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPE_COLORS, PHENOTYPE_SHORT, PHENOTYPES, RESULTS_DIR
from utils.plotting import setup_publication_style, significance_marker

PHENOTYPE_LABELS = {
    "POP": "POP",
    "BPH": "BPH",
    "Bladder": "Bladder\nDysfunction",
    "Constipation": "Constipation",
    "FemaleProlapse": "Female\nProlapse",
    "Incontinence": "Incontinence",
}


def load_data() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "ldsc" / "genetic_correlation_summary.tsv", sep="\t")


def build_matrices(rg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Symmetric rg and p-value matrices indexed by `PHENOTYPES`."""
    n = len(PHENOTYPES)
    rg = pd.DataFrame(np.eye(n), index=PHENOTYPES, columns=PHENOTYPES)
    p = pd.DataFrame(np.zeros((n, n)), index=PHENOTYPES, columns=PHENOTYPES)
    for _, row in rg_df.iterrows():
        a, b = row["phenotype1"], row["phenotype2"]
        rg.loc[a, b] = rg.loc[b, a] = row["rg"]
        p.loc[a, b] = p.loc[b, a] = row["p"]
    return rg, p


def extract_h2(rg_df: pd.DataFrame) -> pd.DataFrame:
    """One h² estimate per phenotype (taken from the first pair it appears in)."""
    h2: dict[str, dict[str, float]] = {}
    for _, row in rg_df.iterrows():
        h2.setdefault(row["phenotype1"], {"h2": row["h2_p1"], "se": row["h2_p1_se"]})
        h2.setdefault(row["phenotype2"], {"h2": row["h2_p2"], "se": row["h2_p2_se"]})
    return pd.DataFrame(h2).T.rename_axis("phenotype").reset_index()


def plot_heatmap(rg: pd.DataFrame, p: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [PHENOTYPE_SHORT[ph] for ph in rg.index]
    mask = np.triu(np.ones_like(rg, dtype=bool), k=1)

    sns.heatmap(
        rg,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Genetic Correlation (rg)", "shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    for i in range(len(rg)):
        for j in range(i):
            stars = significance_marker(p.iloc[i, j])
            if stars:
                ax.text(j + 0.5, i + 0.75, stars, ha="center", va="center", fontsize=8)

    ax.set_title("Genetic Correlation Matrix of Pelvic Floor Disorders\n(LDSC)", fontsize=14, fontweight="bold", pad=20)
    ax.text(1.02, -0.15, "*** p<0.001, ** p<0.01, * p<0.05", transform=ax.transAxes, fontsize=8, va="top")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ldsc_correlation_heatmap.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: ldsc_correlation_heatmap.png/pdf")


def plot_network(rg_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.Graph()
    G.add_nodes_from(PHENOTYPES)

    for _, row in rg_df.iterrows():
        if row["p"] < 0.05:
            G.add_edge(row["phenotype1"], row["phenotype2"], rg=row["rg"], p=row["p"])

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    for u, v, d in G.edges(data=True):
        rg = d["rg"]
        weight = abs(rg)
        color = "#E64B35" if rg > 0 else "#4DBBD5"
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=color,
            alpha=min(1.0, 0.3 + weight * 0.7),
            linewidth=1 + weight * 8,
            zorder=1,
        )
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ax.text(
            mx, my, f"{rg:.2f}", fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    for node in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=[PHENOTYPE_COLORS[node]], node_size=3000, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos, {p: PHENOTYPE_LABELS[p] for p in PHENOTYPES}, font_size=10, font_weight="bold", ax=ax)

    legend = [
        Line2D([0], [0], color="#E64B35", linewidth=4, label="Positive rg"),
        Line2D([0], [0], color="#4DBBD5", linewidth=4, label="Negative rg"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10)
    ax.set_title(
        "Genetic Correlation Network of Pelvic Floor Disorders\n"
        "(Edge width proportional to |rg|, only significant pairs shown)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ldsc_correlation_network.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: ldsc_correlation_network.png/pdf")


def plot_h2_bar(h2_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    h2_df = h2_df.sort_values("h2")
    colors = [PHENOTYPE_COLORS[p] for p in h2_df["phenotype"]]
    y = np.arange(len(h2_df))

    ax.barh(y, h2_df["h2"], xerr=h2_df["se"], color=colors, alpha=0.8,
            error_kw=dict(ecolor="gray", capsize=3, capthick=1))
    ax.set_yticks(y)
    ax.set_yticklabels([PHENOTYPE_SHORT[p] for p in h2_df["phenotype"]])

    for i, (h2, se) in enumerate(zip(h2_df["h2"], h2_df["se"])):
        ax.text(h2 + se + 0.002, i, f"{h2:.3f}", va="center", fontsize=9)

    ax.set_xlabel("SNP-based Heritability (h²)", fontsize=12)
    ax.set_title("SNP Heritability Estimates (LDSC)", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlim(0, (h2_df["h2"] + h2_df["se"]).max() * 1.3)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ldsc_heritability_bar.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: ldsc_heritability_bar.png/pdf")


def plot_clustered(rg: pd.DataFrame, p: pd.DataFrame, out_dir: Path) -> None:
    annot = rg.copy().astype(str)
    for i in range(len(rg)):
        for j in range(len(rg)):
            if i == j:
                annot.iloc[i, j] = "1.00"
            else:
                stars = significance_marker(p.iloc[i, j])
                annot.iloc[i, j] = f"{rg.iloc[i, j]:.2f}{stars}"

    labels = [PHENOTYPE_SHORT[ph] for ph in rg.index]
    g = sns.clustermap(
        rg,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        figsize=(10, 8),
        cbar_kws={"label": "Genetic Correlation (rg)"},
        xticklabels=labels,
        yticklabels=labels,
    )
    g.fig.suptitle("Hierarchically Clustered Genetic Correlation Matrix", fontsize=14, fontweight="bold", y=1.02)
    for ext in ("png", "pdf"):
        g.savefig(out_dir / f"ldsc_correlation_clustered.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: ldsc_correlation_clustered.png/pdf")


def main() -> None:
    setup_publication_style()
    out_dir = FIGURES_DIR / "ldsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LDSC Genetic Correlation Visualization")
    print("=" * 60)

    rg_df = load_data()
    print(f"\nLoaded {len(rg_df)} pairwise correlations")

    rg, p = build_matrices(rg_df)
    h2_df = extract_h2(rg_df)
    print(h2_df.to_string(index=False))

    print("\nPlots:")
    plot_heatmap(rg, p, out_dir)
    plot_network(rg_df, out_dir)
    plot_h2_bar(h2_df, out_dir)
    plot_clustered(rg, p, out_dir)
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
