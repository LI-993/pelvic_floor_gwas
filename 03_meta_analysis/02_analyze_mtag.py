#!/usr/bin/env python3
"""Summarize MTAG cross-phenotype results.

Reads `pelvic_floor_trait_{i}.txt` per phenotype, identifies genome-wide
significant SNPs, computes pairwise overlaps, the genetic-correlation matrix
implied by Omega_hat, and generates the comparison plots used in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPES, RESULTS_DIR
from utils.plotting import setup_publication_style

GWS_THRESHOLD = 5e-8

# Pre/post-MTAG GWS counts from the manuscript; used for the power-gain plot.
PRE_MTAG = {"POP": 308, "BPH": 622, "Bladder": 0, "Constipation": 1, "FemaleProlapse": 450, "Incontinence": 33}
POST_MTAG = {"POP": 625, "BPH": 507, "Bladder": 2, "Constipation": 12, "FemaleProlapse": 635, "Incontinence": 130}


def main() -> None:
    setup_publication_style()
    mtag_dir = RESULTS_DIR / "mtag"
    figures_dir = FIGURES_DIR / "mtag"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MTAG results...")
    results: dict[str, pd.DataFrame] = {}
    for i, p in enumerate(PHENOTYPES, 1):
        df = pd.read_csv(mtag_dir / f"pelvic_floor_trait_{i}.txt", sep="\t")
        results[p] = df
        print(f"  {p}: {len(df):,} SNPs, {(df['mtag_pval'] < GWS_THRESHOLD).sum()} GWS")

    gws_snps = {p: set(df.loc[df["mtag_pval"] < GWS_THRESHOLD, "SNP"]) for p, df in results.items()}

    # Pairwise shared GWS SNPs
    n = len(PHENOTYPES)
    shared = np.zeros((n, n))
    for i, p1 in enumerate(PHENOTYPES):
        for j, p2 in enumerate(PHENOTYPES):
            shared[i, j] = len(gws_snps[p1] & gws_snps[p2])
    print("\nShared GWS SNPs (pairs):")
    for i, p1 in enumerate(PHENOTYPES):
        for j, p2 in enumerate(PHENOTYPES):
            if i < j and shared[i, j]:
                print(f"  {p1} & {p2}: {int(shared[i, j])}")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(shared, dtype=bool), k=1)
    sns.heatmap(shared, mask=mask, annot=True, fmt=".0f", cmap="YlOrRd",
                xticklabels=PHENOTYPES, yticklabels=PHENOTYPES, ax=ax)
    ax.set_title("Shared Genome-wide Significant SNPs (MTAG)", fontsize=14)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"mtag_shared_snps_heatmap.{ext}", dpi=150)
    plt.close()
    print(f"  -> {figures_dir / 'mtag_shared_snps_heatmap.png'}")

    # Omega -> correlation matrix
    omega = pd.read_csv(mtag_dir / "pelvic_floor_omega_hat.txt", sep="\t", header=None).values
    diag_sqrt = np.sqrt(np.diag(omega))
    omega_corr = omega / np.outer(diag_sqrt, diag_sqrt)
    omega_corr_df = pd.DataFrame(omega_corr, index=PHENOTYPES, columns=PHENOTYPES)
    print("\nMTAG Omega correlation matrix:")
    print(omega_corr_df.round(3))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        omega_corr_df, mask=np.triu(np.ones_like(omega_corr, dtype=bool), k=1),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Genetic Correlation Matrix (MTAG Omega)", fontsize=14)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"mtag_genetic_correlation.{ext}", dpi=150)
    plt.close()
    print(f"  -> {figures_dir / 'mtag_genetic_correlation.png'}")

    # Multi-trait SNP table
    all_gws = set().union(*gws_snps.values())
    multi_rows = []
    for snp in all_gws:
        traits = [p for p in PHENOTYPES if snp in gws_snps[p]]
        if len(traits) >= 2:
            min_p = min(results[t].loc[results[t]["SNP"] == snp, "mtag_pval"].iloc[0] for t in traits)
            multi_rows.append({"SNP": snp, "n_traits": len(traits), "traits": ", ".join(traits), "min_pval": min_p})

    multi_df = pd.DataFrame(multi_rows).sort_values(["n_traits", "min_pval"], ascending=[False, True])
    multi_path = mtag_dir / "mtag_multi_trait_snps.csv"
    multi_df.to_csv(multi_path, index=False)
    print(f"\nMulti-trait SNPs: {len(multi_df)} -> {multi_path}")
    print(multi_df.head(20).to_string(index=False))

    # Power-gain comparison plot
    pre = [PRE_MTAG[p] for p in PHENOTYPES]
    post = [POST_MTAG[p] for p in PHENOTYPES]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(PHENOTYPES))
    width = 0.35
    b1 = ax.bar(x - width / 2, pre, width, label="Original GWAS", color="steelblue")
    b2 = ax.bar(x + width / 2, post, width, label="MTAG", color="coral")
    for b in (*b1, *b2):
        h = b.get_height()
        ax.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("Number of GWS SNPs (p < 5e-8)")
    ax.set_title("MTAG Power Gain: Genome-wide Significant SNPs")
    ax.set_xticks(x)
    ax.set_xticklabels(PHENOTYPES, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"mtag_power_comparison.{ext}", dpi=150)
    plt.close()

    summary = pd.DataFrame({
        "Phenotype": PHENOTYPES,
        "Original_GWS": pre,
        "MTAG_GWS": post,
        "Change": [b - a for a, b in zip(pre, post)],
        "Percent_Change": [100 * (b - a) / max(a, 1) for a, b in zip(pre, post)],
    })
    summary.to_csv(mtag_dir / "mtag_summary.csv", index=False)

    print("\nKey findings:")
    print(f"  Unique GWS SNPs across all traits: {len(all_gws)}")
    print(f"  SNPs significant in >=2 traits: {len(multi_df)}")


if __name__ == "__main__":
    main()
