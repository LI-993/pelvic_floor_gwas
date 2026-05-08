#!/usr/bin/env python3
"""Regenerate key paper figures with a colorblind-friendly palette.

Replaces the older hardcoded palette with tab10 (categorical) + viridis
(sequential). Generates revised versions of:
  Fig 2: LDSC heatmap + heritability bar
  Fig 3: MAGMA manhattan (multi-panel)
  Fig 4: GNN gene prioritization
  Fig 5: Cross-ancestry scatter
  Fig 6: Drug-gene network
Output goes to figures/revised/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHR_LENGTHS_GRCh38 as CHR_LENGTHS
from config import FIGURES_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

OUT = FIGURES_DIR / "revised"

sns.set_style("ticks")
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Colorblind-friendly palette (tab10).
_tab10 = plt.cm.tab10.colors
PHENOTYPE_COLORS = {
    "POP": _tab10[0],
    "BPH": _tab10[1],
    "Bladder": _tab10[2],
    "Constipation": _tab10[3],
    "FemaleProlapse": _tab10[4],
    "Incontinence": _tab10[5],
}
PHENOTYPE_COLORS_HEX = {k: matplotlib.colors.rgb2hex(v) for k, v in PHENOTYPE_COLORS.items()}

PHENOTYPE_SHORT = {
    "POP": "POP", "BPH": "BPH", "Bladder": "Bladder",
    "Constipation": "Constip.", "FemaleProlapse": "F.Prolapse", "Incontinence": "Incontin.",
}

INTERACTION_COLORS = {
    "inhibitor": _tab10[3],
    "agonist": _tab10[0],
    "antagonist": _tab10[2],
    "modulator": _tab10[4],
    "other": _tab10[7],
}


def check_file(path: Path) -> bool:
    exists = path.exists()
    print(f"  [{'OK' if exists else 'MISSING'}] {path}")
    return exists


def fig2_ldsc(ldsc_dir: Path) -> None:
    rg_file = ldsc_dir / "genetic_correlation_summary.tsv"
    if not check_file(rg_file):
        return
    rg_df = pd.read_csv(rg_file, sep="\t")
    phenotypes = ["POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence"]
    n = len(phenotypes)

    rg_matrix = pd.DataFrame(np.eye(n), index=phenotypes, columns=phenotypes)
    p_matrix = pd.DataFrame(np.zeros((n, n)), index=phenotypes, columns=phenotypes)
    for _, row in rg_df.iterrows():
        a, b = row["phenotype1"], row["phenotype2"]
        rg_matrix.loc[a, b] = rg_matrix.loc[b, a] = row["rg"]
        p_matrix.loc[a, b] = p_matrix.loc[b, a] = row["p"]

    h2_data: dict[str, dict] = {}
    for _, row in rg_df.iterrows():
        h2_data.setdefault(row["phenotype1"], {"h2": row["h2_p1"], "se": row["h2_p1_se"]})
        h2_data.setdefault(row["phenotype2"], {"h2": row["h2_p2"], "se": row["h2_p2_se"]})
    h2_df = pd.DataFrame(h2_data).T.reset_index().rename(columns={"index": "phenotype"}).sort_values("h2")

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]})

    labels = [PHENOTYPE_SHORT[p] for p in rg_matrix.index]
    mask = np.triu(np.ones_like(rg_matrix, dtype=bool), k=1)
    sns.heatmap(rg_matrix, mask=mask, annot=True, fmt=".2f", cmap="crest", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={"label": "Genetic Correlation (rg)", "shrink": 0.75},
                xticklabels=labels, yticklabels=labels, ax=ax_heat)

    for i in range(n):
        for j in range(i):
            pv = p_matrix.iloc[i, j]
            stars = "***" if pv < 1e-3 else ("**" if pv < 1e-2 else ("*" if pv < 5e-2 else ""))
            if stars:
                ax_heat.text(j + 0.5, i + 0.75, stars, ha="center", va="center", fontsize=8, color="black")

    ax_heat.set_title("A  Genetic Correlation Matrix (LDSC)", fontsize=13, fontweight="bold", loc="left")
    ax_heat.text(1.0, -0.08, "*** p<0.001  ** p<0.01  * p<0.05", transform=ax_heat.transAxes, fontsize=8, ha="right", va="top")

    colors = [PHENOTYPE_COLORS.get(p, (0.5, 0.5, 0.5)) for p in h2_df["phenotype"]]
    y_pos = np.arange(len(h2_df))
    ax_bar.barh(y_pos, h2_df["h2"], xerr=h2_df["se"], color=colors, alpha=0.85,
                error_kw=dict(ecolor="gray", capsize=3, capthick=1), edgecolor="white")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([PHENOTYPE_SHORT[p] for p in h2_df["phenotype"]])
    for i, (h2, se) in enumerate(zip(h2_df["h2"], h2_df["se"])):
        ax_bar.text(h2 + se + 0.001, i, f"{h2:.4f}", va="center", fontsize=9)
    ax_bar.set_xlabel(r"SNP-based Heritability ($h^2$)", fontsize=11)
    ax_bar.set_title("B  SNP Heritability Estimates", fontsize=13, fontweight="bold", loc="left")
    ax_bar.set_xlim(0, (h2_df["h2"] + h2_df["se"]).max() * 1.4)
    ax_bar.axvline(0, color="gray", linewidth=0.5)
    sns.despine(ax=ax_bar)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig2_ldsc_revised.{ext}", bbox_inches="tight")
    plt.close()
    print("  -> Fig2_ldsc_revised.png/pdf")


def fig3_magma(magma_dir: Path) -> None:
    phenotypes = ["POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence"]
    full: dict[str, pd.DataFrame] = {}
    for pheno in phenotypes:
        path = magma_dir / f"{pheno}_genes.genes.out.txt"
        if path.exists():
            df = pd.read_csv(path, sep=r"\s+", comment="#")
            df["Phenotype"] = pheno
            full[pheno] = df
    if not full:
        print("  No MAGMA results, skipping Fig 3")
        return

    offsets: dict[int, int] = {}
    cumulative = 0
    for c in range(1, 23):
        offsets[c] = cumulative
        cumulative += CHR_LENGTHS.get(c, 100_000_000) + 5_000_000

    chr_color_even = cm.viridis(0.3)
    chr_color_odd = cm.viridis(0.7)

    fig, axes = plt.subplots(3, 2, figsize=(18, 13))
    axes = axes.flatten()
    for ax, pheno in zip(axes, phenotypes):
        if pheno not in full:
            ax.set_title(f"{pheno} (No data)")
            continue
        df = full[pheno].copy()
        df = df[df["P"].notna() & (df["P"] > 0)].copy()
        df["neglog10p"] = -np.log10(df["P"])
        df["CHR_num"] = pd.to_numeric(df["CHR"], errors="coerce")
        df = df.dropna(subset=["CHR_num"])
        df["CHR_num"] = df["CHR_num"].astype(int)
        df = df[df["CHR_num"].between(1, 22)]
        df["plot_pos"] = df.apply(lambda r: offsets.get(r["CHR_num"], 0) + r["START"], axis=1)

        colors = [chr_color_even if c % 2 == 0 else chr_color_odd for c in df["CHR_num"]]
        ax.scatter(df["plot_pos"], df["neglog10p"], c=colors, alpha=0.5, s=10, edgecolors="none")
        bonf = -np.log10(0.05 / max(len(df), 1))
        ax.axhline(bonf, color=PHENOTYPE_COLORS["Constipation"], linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(-np.log10(1e-4), color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

        for _, row in df.nlargest(5, "neglog10p").iterrows():
            if row["neglog10p"] > 4:
                gene = str(int(row["GENE"])) if isinstance(row["GENE"], float) else str(row["GENE"])
                ax.annotate(gene, xy=(row["plot_pos"], row["neglog10p"]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, alpha=0.85, fontstyle="italic")

        ax.set_title(pheno, fontsize=12, fontweight="bold", color=PHENOTYPE_COLORS.get(pheno, (0, 0, 0)))
        ax.set_ylabel(r"$-\log_{10}(P)$", fontsize=10)
        ax.set_xlabel("Chromosome", fontsize=9)
        chr_centers = {c: offsets[c] + CHR_LENGTHS[c] / 2 for c in range(1, 23)}
        ax.set_xticks([chr_centers[c] for c in range(1, 23)])
        ax.set_xticklabels([str(c) for c in range(1, 23)], fontsize=6, rotation=0)
        sns.despine(ax=ax)

    fig.suptitle("Gene-based Association Analysis (MAGMA)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig3_magma_manhattan_revised.{ext}", bbox_inches="tight")
    plt.close()
    print("  -> Fig3_magma_manhattan_revised.png/pdf")


def fig4_gnn(gnn_dir: Path) -> None:
    ranking_file = gnn_dir / "ensemble_ranking.csv"
    if not check_file(ranking_file):
        return
    df = pd.read_csv(ranking_file)
    top = df.head(25).sort_values("Ensemble_Score").copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.3, 1]})

    y_pos = np.arange(len(top))
    colors_arr = [PHENOTYPE_COLORS["POP"] if pos else PHENOTYPE_COLORS["BPH"] for pos in top["Is_Positive"]]
    ax1.hlines(y_pos, 0, top["Ensemble_Score"], color="gray", alpha=0.4, linewidth=1)
    ax1.scatter(top["Ensemble_Score"], y_pos, c=colors_arr, s=80, zorder=5, edgecolors="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top["Gene"], fontsize=9)
    ax1.set_xlabel("Ensemble Score", fontsize=11)
    ax1.set_title("A  GNN Ensemble Gene Ranking (Top 25)", fontsize=13, fontweight="bold", loc="left")
    ax1.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PHENOTYPE_COLORS["POP"], markersize=8, label="Known positive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PHENOTYPE_COLORS["BPH"], markersize=8, label="Novel candidate"),
    ], loc="lower right", fontsize=9)
    sns.despine(ax=ax1)

    ax2.scatter(df["neglog10p"], df["Ensemble_Score"], c=df["n_phenotypes"],
                cmap="viridis", s=30, alpha=0.6, edgecolors="white", linewidth=0.3)
    top10 = df.head(10)
    ax2.scatter(top10["neglog10p"], top10["Ensemble_Score"], c="none", s=80,
                edgecolors=PHENOTYPE_COLORS_HEX["Constipation"], linewidth=1.5)

    try:
        from adjustText import adjust_text
        texts = [
            ax2.text(row["neglog10p"], row["Ensemble_Score"], row["Gene"], fontsize=7, fontstyle="italic")
            for _, row in top10.iterrows()
        ]
        adjust_text(
            texts, ax=ax2, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            expand=(2.0, 2.0), force_text=(1.5, 1.5), force_points=(1.5, 1.5),
            ensure_inside_axes=True, max_move=50,
        )
    except ImportError:  # adjustText is optional
        for _, row in top10.iterrows():
            ax2.text(row["neglog10p"], row["Ensemble_Score"], row["Gene"], fontsize=7, fontstyle="italic")

    ax2.set_xlabel(r"$-\log_{10}(P_{\mathrm{MAGMA}})$", fontsize=11)
    ax2.set_ylabel("Ensemble Score", fontsize=11)
    ax2.set_title("B  GNN Score vs MAGMA Significance", fontsize=13, fontweight="bold", loc="left")
    plt.colorbar(ax2.collections[0], ax=ax2, shrink=0.7, label="# Phenotypes")
    sns.despine(ax=ax2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig4_gnn_prioritization_revised.{ext}", bbox_inches="tight")
    plt.close()
    print("  -> Fig4_gnn_prioritization_revised.png/pdf")


def fig5_cross_ancestry(ca_dir: Path) -> None:
    snp_file = ca_dir / "top10_snps.csv"
    summary_file = ca_dir / "n81_summary.csv"
    if not check_file(snp_file):
        return
    snps = pd.read_csv(snp_file)
    summary = pd.read_csv(summary_file) if summary_file.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (eur_col, other_col, label) in zip(axes, [("beta_EUR", "beta_AFR", "AFR"), ("beta_EUR", "beta_CSA", "CSA")]):
        ax.scatter(snps[eur_col], snps[other_col], c=snps["neglog10_pval_EUR"],
                   cmap="viridis", s=60, edgecolors="white", linewidth=0.5, zorder=5)
        lims = [min(snps[eur_col].min(), snps[other_col].min()) * 1.2,
                max(snps[eur_col].max(), snps[other_col].max()) * 1.2]
        ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Effect size (EUR)", fontsize=11)
        ax.set_ylabel(f"Effect size ({label})", fontsize=11)
        title = "A  EUR vs AFR Effect Sizes" if label == "AFR" else "B  EUR vs CSA Effect Sizes"
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        plt.colorbar(ax.collections[0], ax=ax, shrink=0.7, label=r"$-\log_{10}(P_{\mathrm{EUR}})$")
        sns.despine(ax=ax)

    if summary is not None:
        for ax, col in zip(axes, ["corr_eur_afr_all", "corr_eur_csa_all"]):
            if col in summary.columns:
                r = summary[col].iloc[0]
                ax.text(0.05, 0.95, f"r = {r:.4f}", transform=ax.transAxes, fontsize=10, va="top",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.suptitle("Cross-Ancestry Comparison of Top GWAS Loci", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig5_cross_ancestry_revised.{ext}", bbox_inches="tight")
    plt.close()
    print("  -> Fig5_cross_ancestry_revised.png/pdf")


def fig6_drug_network(drug_dir: Path) -> None:
    pri_file = drug_dir / "prioritized_candidates.csv"
    if not check_file(pri_file):
        return
    prioritized = pd.read_csv(pri_file)
    top_drugs = prioritized.nlargest(30, "priority_score")

    G = nx.Graph()
    for gene in top_drugs["gene_symbol"].unique():
        G.add_node(gene, node_type="gene")
    for _, row in top_drugs.iterrows():
        itype = row.get("interaction_type", "other") or "other"
        G.add_node(row["drug"], node_type="drug", interaction=itype)
        G.add_edge(row["drug"], row["gene_symbol"], weight=row["priority_score"])

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]

    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color=[PHENOTYPE_COLORS["POP"]],
                           node_size=1000, node_shape="s", alpha=0.9, ax=ax)
    drug_colors = [INTERACTION_COLORS.get(G.nodes[n].get("interaction", "other") or "other", INTERACTION_COLORS["other"])
                   for n in drug_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color=drug_colors, node_size=500,
                           node_shape="o", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.35, edge_color="gray",
                           width=[d.get("weight", 1) / 5 for _, _, d in G.edges(data=True)], ax=ax)
    nx.draw_networkx_labels(G, pos, {n: n for n in gene_nodes}, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_labels(G, pos, {n: (n[:15] + "..." if len(n) > 15 else n) for n in drug_nodes}, font_size=7, ax=ax)

    ax.legend(handles=[
        Patch(facecolor=PHENOTYPE_COLORS["POP"], label="Gene (target)", alpha=0.9),
        Patch(facecolor=INTERACTION_COLORS["inhibitor"], label="Inhibitor"),
        Patch(facecolor=INTERACTION_COLORS["agonist"], label="Agonist"),
        Patch(facecolor=INTERACTION_COLORS["antagonist"], label="Antagonist"),
        Patch(facecolor=INTERACTION_COLORS["other"], label="Other"),
    ], loc="upper left", fontsize=10, framealpha=0.9, edgecolor="gray")
    ax.set_title("Drug-Gene Interaction Network\n(Top 30 by Priority Score)", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig6_drug_network_revised.{ext}", bbox_inches="tight")
    plt.close()
    print("  -> Fig6_drug_network_revised.png/pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Regenerating revised paper figures")
    print(f"Output directory: {OUT}")
    print("=" * 70)
    print("\nNew phenotype color palette (tab10):")
    for pheno, hex_color in PHENOTYPE_COLORS_HEX.items():
        print(f"  {pheno:20s} -> {hex_color}")

    print("\n[Fig 2] LDSC heatmap + heritability bar")
    fig2_ldsc(RESULTS_DIR / "ldsc")

    print("\n[Fig 3] MAGMA gene-based Manhattan plots")
    fig3_magma(RESULTS_DIR / "magma")

    print("\n[Fig 4] GNN gene prioritization")
    fig4_gnn(RESULTS_DIR / "gnn_prioritization")

    print("\n[Fig 5] Cross-ancestry comparison")
    fig5_cross_ancestry(RESULTS_DIR / "cross_ancestry")

    print("\n[Fig 6] Drug-gene network")
    fig6_drug_network(RESULTS_DIR / "drug_repurposing")

    print(f"\nAll revised figures saved to {OUT}")


if __name__ == "__main__":
    main()
