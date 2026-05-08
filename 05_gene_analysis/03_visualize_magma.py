#!/usr/bin/env python3
"""MAGMA gene-based analysis visualizations.

Produces a per-phenotype gene-level Manhattan, top-10-gene bar charts, a
shared-significant-gene UpSet-style plot, a top-30 gene-phenotype heatmap,
and a per-phenotype counts plot.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHR_LENGTHS_GRCh38, FIGURES_DIR, PHENOTYPE_COLORS, PHENOTYPES, RESULTS_DIR
from utils.plotting import bonferroni_threshold, setup_publication_style

DEFAULT_GENE_COUNT = 19_000  # ~ MAGMA gene universe; used for the bar-chart Bonferroni line.


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Top-genes table + per-phenotype gene-level results."""
    magma_dir = RESULTS_DIR / "magma"
    top_genes = pd.read_csv(magma_dir / "magma_top_genes.csv")

    full: dict[str, pd.DataFrame] = {}
    for p in PHENOTYPES:
        path = magma_dir / f"{p}_genes.genes.out.txt"
        if path.exists():
            df = pd.read_csv(path, sep=r"\s+", comment="#")
            df["Phenotype"] = p
            full[p] = df
    return top_genes, full


def _gene_offsets() -> dict[int, int]:
    offsets: dict[int, int] = {}
    cumulative = 0
    for chrom in range(1, 23):
        offsets[chrom] = cumulative
        cumulative += CHR_LENGTHS_GRCh38[chrom] + 5_000_000
    return offsets


def plot_manhattan(full: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not full:
        print("  Manhattan: no data")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    offsets = _gene_offsets()

    for ax, pheno in zip(axes, PHENOTYPES):
        if pheno not in full:
            ax.set_title(f"{pheno} (no data)")
            continue
        df = full[pheno].copy()
        df = df[df["P"].notna() & (df["P"] > 0)]
        df["neglog10p"] = -np.log10(df["P"])
        df["CHR_num"] = df["CHR"].replace({"X": 23, "Y": 24}).astype(int)
        df = df[df["CHR_num"] <= 22]
        df["plot_pos"] = df.apply(lambda r: offsets.get(r["CHR_num"], 0) + r["START"], axis=1)

        colors = ["#4DBBD5" if c % 2 == 0 else "#3C5488" for c in df["CHR_num"]]
        ax.scatter(df["plot_pos"], df["neglog10p"], c=colors, alpha=0.5, s=8, edgecolors="none")
        ax.axhline(-np.log10(bonferroni_threshold(len(df))), color="red", linestyle="--", linewidth=0.8, alpha=0.7)

        for _, row in df.nlargest(5, "neglog10p").iterrows():
            if row["neglog10p"] > 4:
                ax.annotate(row["GENE"], xy=(row["plot_pos"], row["neglog10p"]),
                            xytext=(3, 3), textcoords="offset points", fontsize=7, alpha=0.8)

        ax.set_title(pheno, fontsize=12, fontweight="bold", color=PHENOTYPE_COLORS[pheno])
        ax.set_xlabel("Chromosome", fontsize=9)
        ax.set_ylabel("-log10(P)", fontsize=9)
        chr_centers = {c: offsets[c] + CHR_LENGTHS_GRCh38[c] / 2 for c in range(1, 23)}
        ax.set_xticks([chr_centers[c] for c in (1, 5, 10, 15, 20)])
        ax.set_xticklabels([1, 5, 10, 15, 20])

    plt.suptitle("Gene-based Association Analysis (MAGMA)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magma_manhattan.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: magma_manhattan.png/pdf")


def plot_top_genes_bar(top_genes: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, pheno in zip(axes, top_genes["Phenotype"].unique()):
        sub = top_genes[top_genes["Phenotype"] == pheno].sort_values("P").copy()
        sub["neglog10p"] = -np.log10(sub["P"])
        y_pos = np.arange(len(sub))
        color = PHENOTYPE_COLORS.get(pheno, "#888888")

        ax.barh(y_pos, sub["neglog10p"], color=color, alpha=0.8, edgecolor="white")
        ax.axvline(-np.log10(bonferroni_threshold(DEFAULT_GENE_COUNT)), color="red", linestyle="--", linewidth=1, label="Bonferroni")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["Symbol"])
        ax.invert_yaxis()
        ax.set_xlabel("-log10(P)", fontsize=10)
        ax.set_title(pheno, fontsize=12, fontweight="bold", color=color)
        for i, (p_val, neglog) in enumerate(zip(sub["P"], sub["neglog10p"])):
            ax.text(neglog + 0.3, i, f"{p_val:.1e}", va="center", fontsize=8)

    plt.suptitle("Top 10 Genes per Phenotype (MAGMA)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magma_top_genes_bar.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: magma_top_genes_bar.png/pdf")


def plot_shared_upset(full: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not full:
        return

    sig: dict[str, set[str]] = {}
    for pheno, df in full.items():
        thr = bonferroni_threshold(len(df))
        s = set(df.loc[df["P"] < thr, "GENE"])
        if not s:  # fall back to suggestive
            s = set(df.loc[df["P"] < 1e-4, "GENE"])
        sig[pheno] = s

    all_genes = set().union(*sig.values())
    if not all_genes:
        print("  UpSet: no significant genes")
        return

    matrix = pd.DataFrame({pheno: [g in s for g in all_genes] for pheno, s in sig.items()}, index=list(all_genes)).astype(int)

    intersections: dict[tuple[str, ...], int] = {}
    for pheno in sig:
        if (count := matrix[pheno].sum()):
            intersections[(pheno,)] = int(count)
    for combo in combinations(sig, 2):
        if (count := matrix[list(combo)].all(axis=1).sum()):
            intersections[combo] = int(count)
    for combo in combinations(sig, 3):
        if (count := matrix[list(combo)].all(axis=1).sum()):
            intersections[combo] = int(count)

    fig, (ax_bar, ax_blank) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    sorted_int = sorted(intersections.items(), key=lambda x: -x[1])[:20]
    x_pos = np.arange(len(sorted_int))
    heights = [v for _, v in sorted_int]
    labels = [" & ".join(k) if len(k) <= 2 else f"{len(k)} traits" for k, _ in sorted_int]
    colors = [PHENOTYPE_COLORS.get(k[0], "#888888") if len(k) == 1 else "#3C5488" for k, _ in sorted_int]

    ax_bar.bar(x_pos, heights, color=colors, alpha=0.8, edgecolor="white")
    for i, h in enumerate(heights):
        ax_bar.text(i, h + 0.5, str(h), ha="center", fontsize=9)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_bar.set_ylabel("Number of Genes", fontsize=12)
    ax_bar.set_title("Shared Significant Genes Across Phenotypes (MAGMA)", fontsize=14, fontweight="bold")
    ax_blank.axis("off")

    legend = [Patch(facecolor=PHENOTYPE_COLORS[p], label=p, alpha=0.8) for p in sig]
    legend.append(Patch(facecolor="#3C5488", label="Multiple", alpha=0.8))
    ax_bar.legend(handles=legend, loc="upper right", ncol=3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magma_shared_genes_upset.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: magma_shared_genes_upset.png/pdf")


def plot_gene_phenotype_heatmap(full: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not full:
        return

    top_genes = (
        pd.concat([df.nsmallest(20, "P")[["GENE", "P"]].assign(Phenotype=p) for p, df in full.items()])
        .groupby("GENE")["P"]
        .min()
        .nsmallest(50)
        .index
    )

    matrix = pd.DataFrame(0.0, index=top_genes, columns=list(full))
    for pheno, df in full.items():
        gene_p = df.set_index("GENE")["P"]
        for gene in top_genes:
            if gene in gene_p.index:
                matrix.loc[gene, pheno] = -np.log10(gene_p[gene])

    matrix = matrix.assign(_max=matrix.max(axis=1)).sort_values("_max", ascending=False).drop(columns="_max").head(30)

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "-log10(P-value)"}, ax=ax)
    ax.set_xlabel("Phenotype", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.set_title("Gene-Phenotype Association Heatmap (MAGMA)\nTop 30 Genes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magma_gene_phenotype_heatmap.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: magma_gene_phenotype_heatmap.png/pdf")


def plot_significance_summary(full: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not full:
        return

    rows = []
    for pheno, df in full.items():
        bonf = bonferroni_threshold(len(df))
        rows.append({
            "Phenotype": pheno,
            "Bonferroni": int((df["P"] < bonf).sum()),
            "Suggestive": int(((df["P"] >= bonf) & (df["P"] < 1e-4)).sum()),
            "Nominal": int(((df["P"] >= 1e-4) & (df["P"] < 0.05)).sum()),
        })
    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.25
    layers = [
        ("Bonferroni", "#E64B35", -width),
        ("Suggestive", "#4DBBD5", 0.0),
        ("Nominal", "#00A087", width),
    ]
    for label, color, offset in layers:
        bars = ax.bar(x + offset, summary_df[label], width, label=f"{label}", color=color, alpha=0.8)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(str(int(h)), xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Phenotype"], rotation=45, ha="right")
    ax.set_xlabel("Phenotype", fontsize=12)
    ax.set_ylabel("Number of Significant Genes", fontsize=12)
    ax.set_title("Significant Genes per Phenotype (MAGMA)", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magma_significant_summary.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: magma_significant_summary.png/pdf")


def main() -> None:
    setup_publication_style()
    out_dir = FIGURES_DIR / "magma"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MAGMA Visualizations")
    print("=" * 60)

    top_genes, full = load_data()
    print(f"\nTop genes: {len(top_genes)} | full results for: {list(full)}")

    plot_manhattan(full, out_dir)
    plot_top_genes_bar(top_genes, out_dir)
    plot_shared_upset(full, out_dir)
    plot_gene_phenotype_heatmap(full, out_dir)
    plot_significance_summary(full, out_dir)
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
