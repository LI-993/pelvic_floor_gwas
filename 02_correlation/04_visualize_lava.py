#!/usr/bin/env python3
"""LAVA local-genetic-correlation visualizations.

Reads `results/lava/lava_bivariate.tsv` (produced by the LAVA R run, kept on
disk from the original analysis) and produces:
- genome-wide Manhattan plot of -log10(p)
- top significant-loci heatmap across phenotype pairs
- per-chromosome significant-locus bar chart
- rho distribution (overall + per pair)
- pair-specific Manhattan multi-panel plot
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, RESULTS_DIR
from utils.plotting import bonferroni_threshold, chr_offsets, setup_publication_style


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "lava" / "lava_bivariate.tsv", sep="\t")
    df["pair"] = df["phen1"] + " vs " + df["phen2"]
    df["neglog10p"] = -np.log10(df["p"].clip(lower=1e-300))
    return df


def plot_manhattan(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    offsets = chr_offsets(df, pos_col="stop")
    df = df.assign(plot_pos=df.apply(lambda r: offsets.get(r["chr"], 0) + (r["start"] + r["stop"]) / 2, axis=1))

    bonf = -np.log10(bonferroni_threshold(len(df)))
    suggestive = -np.log10(1e-4)

    colors = ["#4DBBD5" if c % 2 == 0 else "#3C5488" for c in df["chr"]]
    ax.scatter(df["plot_pos"], df["neglog10p"], c=colors, alpha=0.6, s=20, edgecolors="none")
    ax.axhline(bonf, color="red", linestyle="--", linewidth=1, label=f"Bonferroni (p={bonferroni_threshold(len(df)):.2e})")
    ax.axhline(suggestive, color="orange", linestyle="--", linewidth=1, label="Suggestive (p=1e-4)")

    for _, row in df.nlargest(10, "neglog10p").iterrows():
        if row["neglog10p"] > bonf:
            ax.annotate(
                f"chr{row['chr']}:{row['locus']}\n{row['phen1']}-{row['phen2']}",
                xy=(row["plot_pos"], row["neglog10p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
            )

    chr_centers: dict[int, float] = {}
    for chrom in range(1, 23):
        chr_data = df[df["chr"] == chrom]
        if len(chr_data):
            chr_centers[chrom] = chr_data["plot_pos"].mean()
    ax.set_xticks(list(chr_centers.values()))
    ax.set_xticklabels(list(chr_centers.keys()))

    ax.set_xlabel("Chromosome", fontsize=12)
    ax.set_ylabel("-log10(P-value)", fontsize=12)
    ax.set_title("LAVA Local Genetic Correlation Manhattan Plot\n(All Phenotype Pairs)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, df["neglog10p"].max() * 1.1)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"lava_manhattan.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: lava_manhattan.png/pdf")


def plot_top_loci_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    bonf = bonferroni_threshold(len(df))
    sig = df[df["p"] < bonf].copy()
    if not len(sig):
        print("  No Bonferroni-significant loci; using top 50 by p")
        sig = df.nsmallest(50, "p").copy()

    sig["locus_id"] = "chr" + sig["chr"].astype(str) + ":" + sig["locus"].astype(str)
    top_loci = sig.groupby("locus_id")["p"].min().nsmallest(30).index

    matrix = pd.DataFrame(0.0, index=top_loci, columns=sig["pair"].unique())
    for _, row in sig.iterrows():
        if row["locus_id"] in top_loci:
            matrix.loc[row["locus_id"], row["pair"]] = row["rho"]
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    if matrix.empty:
        print("  Heatmap: no data")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        matrix,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Local Genetic Correlation (ρ)"},
        ax=ax,
    )
    ax.set_xlabel("Phenotype Pair", fontsize=12)
    ax.set_ylabel("Genomic Locus", fontsize=12)
    ax.set_title("Top Significant Loci: Local Genetic Correlations\n(LAVA Bivariate Analysis)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"lava_top_loci_heatmap.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: lava_top_loci_heatmap.png/pdf")


def plot_chromosome_summary(df: pd.DataFrame, out_dir: Path) -> None:
    bonf = bonferroni_threshold(len(df))
    suggestive = 1e-4

    rows = []
    for chrom in range(1, 23):
        cd = df[df["chr"] == chrom]
        rows.append({
            "chr": chrom,
            "Bonferroni": (cd["p"] < bonf).sum(),
            "Suggestive": ((cd["p"] >= bonf) & (cd["p"] < suggestive)).sum(),
        })
    chr_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(chr_df))
    width = 0.35
    bars = ax.bar(x - width / 2, chr_df["Bonferroni"], width, label="Bonferroni significant", color="#E64B35", alpha=0.8)
    ax.bar(x + width / 2, chr_df["Suggestive"], width, label="Suggestive", color="#4DBBD5", alpha=0.8)

    ax.set_xlabel("Chromosome", fontsize=12)
    ax.set_ylabel("Number of Significant Loci", fontsize=12)
    ax.set_title("Significant Local Genetic Correlations by Chromosome (LAVA)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(chr_df["chr"])
    ax.legend()

    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"lava_chromosome_summary.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: lava_chromosome_summary.png/pdf")


def plot_rho_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["rho"], bins=50, kde=True, ax=axes[0], color="#3C5488", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Local Genetic Correlation (ρ)", fontsize=12)
    axes[0].set_title("Distribution of Local Genetic Correlations\n(All Tests)", fontsize=12, fontweight="bold")

    pair_order = df.groupby("pair")["rho"].median().sort_values(ascending=False).index[:10]
    sns.boxplot(data=df[df["pair"].isin(pair_order)], x="rho", y="pair", order=pair_order, palette="RdBu_r", ax=axes[1])
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Local Genetic Correlation (ρ)", fontsize=12)
    axes[1].set_title("Local Correlations by Phenotype Pair (Top 10)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"lava_rho_distribution.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: lava_rho_distribution.png/pdf")


def plot_pair_manhattan(df: pd.DataFrame, out_dir: Path) -> None:
    bonf = bonferroni_threshold(len(df))
    pair_sig = df[df["p"] < bonf].groupby("pair").size()
    top_pairs = pair_sig.nlargest(6).index.tolist()
    if len(top_pairs) < 6:
        remainder = [p for p in df["pair"].unique() if p not in top_pairs]
        top_pairs.extend(remainder[: 6 - len(top_pairs)])

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    offsets = chr_offsets(df, pos_col="stop")

    for idx, pair in enumerate(top_pairs[:6]):
        ax = axes[idx]
        pd_pair = df[df["pair"] == pair].copy()
        pd_pair["plot_pos"] = pd_pair.apply(lambda r: offsets.get(r["chr"], 0) + (r["start"] + r["stop"]) / 2, axis=1)
        colors = ["#4DBBD5" if c % 2 == 0 else "#3C5488" for c in pd_pair["chr"]]
        ax.scatter(pd_pair["plot_pos"], pd_pair["neglog10p"], c=colors, alpha=0.6, s=15, edgecolors="none")
        ax.axhline(-np.log10(bonferroni_threshold(len(pd_pair))), color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(pair, fontsize=10, fontweight="bold")
        ax.set_xlabel("Chromosome", fontsize=9)
        ax.set_ylabel("-log10(P)", fontsize=9)
        ax.set_xticks([])

    plt.suptitle("LAVA Local Genetic Correlations by Phenotype Pair", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"lava_pair_manhattan.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: lava_pair_manhattan.png/pdf")


def main() -> None:
    setup_publication_style()
    out_dir = FIGURES_DIR / "lava"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAVA Local Genetic Correlation Visualization")
    print("=" * 60)

    df = load_data()
    bonf = bonferroni_threshold(len(df))
    print(
        f"\nTests: {len(df):,} | pairs: {df['pair'].nunique()} | unique loci: {df['locus'].nunique()}"
        f"\nBonferroni threshold: {bonf:.2e}"
        f"\nBonferroni significant: {(df['p'] < bonf).sum()}"
        f"\nSuggestive (p<1e-4): {(df['p'] < 1e-4).sum()}"
    )

    plot_manhattan(df, out_dir)
    plot_top_loci_heatmap(df, out_dir)
    plot_chromosome_summary(df, out_dir)
    plot_rho_distribution(df, out_dir)
    plot_pair_manhattan(df, out_dir)
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
