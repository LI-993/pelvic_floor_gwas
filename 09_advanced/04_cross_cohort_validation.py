#!/usr/bin/env python3
"""Cross-cohort validation: FinnGen R12 vs GWAS Catalog phenotypes.

Within-cohort vs between-cohort genetic correlations, top-gene overlap
(Jaccard), POP/FemaleProlapse concordance, and a 4-panel summary figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR
from utils.plotting import setup_publication_style

OUT = RESULTS_DIR / "cross_cohort_validation"
FIG = FIGURES_DIR / "cross_cohort"

FINNGEN_PHENOS = ["BPH", "Bladder", "Constipation", "FemaleProlapse"]
GWAS_CATALOG_PHENOS = ["POP", "Incontinence"]
ALL_PHENOS = FINNGEN_PHENOS + GWAS_CATALOG_PHENOS


def load_ldsc() -> pd.DataFrame | None:
    path = RESULTS_DIR / "ldsc" / "genetic_correlation_summary.tsv"
    if not path.exists():
        print(f"  Missing: {path}")
        return None
    return pd.read_csv(path, sep="\t")


def load_magma() -> pd.DataFrame | None:
    path = RESULTS_DIR / "magma" / "magma_top_genes.csv"
    if not path.exists():
        print(f"  Missing: {path}")
        return None
    return pd.read_csv(path)


def heritability_table(ldsc: pd.DataFrame) -> dict[str, dict]:
    h2: dict[str, dict] = {}
    for _, row in ldsc.iterrows():
        h2.setdefault(row["phenotype1"], {"h2": row["h2_p1"], "se": row["h2_p1_se"]})
        h2.setdefault(row["phenotype2"], {"h2": row["h2_p2"], "se": row["h2_p2_se"]})
    return h2


def architecture_summary(ldsc: pd.DataFrame, h2: dict[str, dict]) -> dict:
    finn_h2 = [h2[p]["h2"] for p in FINNGEN_PHENOS if p in h2]
    cat_h2 = [h2[p]["h2"] for p in GWAS_CATALOG_PHENOS if p in h2]

    within_finn: list[float] = []
    within_cat: list[float] = []
    between: list[float] = []
    for _, row in ldsc.iterrows():
        p1, p2 = row["phenotype1"], row["phenotype2"]
        s1 = "FinnGen" if p1 in FINNGEN_PHENOS else "GWAS_Catalog"
        s2 = "FinnGen" if p2 in FINNGEN_PHENOS else "GWAS_Catalog"
        if s1 == s2 == "FinnGen":
            within_finn.append(row["rg"])
        elif s1 == s2 == "GWAS_Catalog":
            within_cat.append(row["rg"])
        else:
            between.append(row["rg"])

    return {
        "h2_comparison": {
            "finngen_mean": float(np.mean(finn_h2)) if finn_h2 else None,
            "finngen_std": float(np.std(finn_h2)) if finn_h2 else None,
            "gwas_catalog_mean": float(np.mean(cat_h2)) if cat_h2 else None,
            "gwas_catalog_std": float(np.std(cat_h2)) if cat_h2 else None,
        },
        "rg_comparison": {
            "within_finngen_mean": float(np.mean(within_finn)) if within_finn else None,
            "within_finngen_std": float(np.std(within_finn)) if within_finn else None,
            "between_cohort_mean": float(np.mean(between)) if between else None,
            "between_cohort_std": float(np.std(between)) if between else None,
            "within_catalog_mean": float(np.mean(within_cat)) if within_cat else None,
        },
    }


def gene_overlap(magma: pd.DataFrame) -> dict:
    by_pheno = {p: set(magma[magma["Phenotype"] == p]["Symbol"]) for p in ALL_PHENOS}
    matrix = pd.DataFrame(index=ALL_PHENOS, columns=ALL_PHENOS, dtype=float)
    for p1 in ALL_PHENOS:
        for p2 in ALL_PHENOS:
            g1, g2 = by_pheno[p1], by_pheno[p2]
            if g1 and g2:
                matrix.loc[p1, p2] = len(g1 & g2) / len(g1 | g2)

    finn_genes = set().union(*(by_pheno[p] for p in FINNGEN_PHENOS))
    cat_genes = set().union(*(by_pheno[p] for p in GWAS_CATALOG_PHENOS))
    shared = sorted(finn_genes & cat_genes)
    between = [matrix.loc[fp, gp] for fp in FINNGEN_PHENOS for gp in GWAS_CATALOG_PHENOS if pd.notna(matrix.loc[fp, gp])]

    return {
        "gene_overlap_matrix": matrix,
        "mean_between_overlap": float(np.mean(between)) if between else 0.0,
        "shared_genes": shared,
    }


def pop_concordance(ldsc: pd.DataFrame, magma: pd.DataFrame) -> dict:
    res: dict = {}
    pair = ldsc[
        ((ldsc["phenotype1"] == "POP") & (ldsc["phenotype2"] == "FemaleProlapse")) |
        ((ldsc["phenotype1"] == "FemaleProlapse") & (ldsc["phenotype2"] == "POP"))
    ]
    if not pair.empty:
        row = pair.iloc[0]
        res["pop_femaleprolapse_rg"] = {"rg": float(row["rg"]), "se": float(row["rg_se"]), "p": float(row["p"])}
    pop_genes = set(magma[magma["Phenotype"] == "POP"]["Symbol"])
    fp_genes = set(magma[magma["Phenotype"] == "FemaleProlapse"]["Symbol"])
    res["pop_fp_gene_overlap"] = sorted(pop_genes & fp_genes)
    return res


def plot_summary(arch: dict, h2: dict[str, dict], gene_res: dict, concordance: dict, ldsc: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    phenos = list(h2)
    h2_vals = [h2[p]["h2"] for p in phenos]
    h2_ses = [h2[p]["se"] for p in phenos]
    colors = ["#E64B35" if p in FINNGEN_PHENOS else "#4DBBD5" for p in phenos]
    ax1.barh(np.arange(len(phenos)), h2_vals, xerr=h2_ses, color=colors, alpha=0.8, capsize=3)
    ax1.set_yticks(np.arange(len(phenos)))
    ax1.set_yticklabels(phenos)
    ax1.set_xlabel("SNP Heritability (h²)")
    ax1.set_title("Heritability by Cohort\n(Red=FinnGen, Blue=GWAS Catalog)", fontweight="bold")

    ax2 = fig.add_subplot(2, 2, 2)
    rg_matrix = pd.DataFrame(index=ALL_PHENOS, columns=ALL_PHENOS, dtype=float)
    np.fill_diagonal(rg_matrix.values, 1.0)
    for _, row in ldsc.iterrows():
        if row["phenotype1"] in ALL_PHENOS and row["phenotype2"] in ALL_PHENOS:
            rg_matrix.loc[row["phenotype1"], row["phenotype2"]] = row["rg"]
            rg_matrix.loc[row["phenotype2"], row["phenotype1"]] = row["rg"]
    sns.heatmap(rg_matrix.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax2, square=True,
                cbar_kws={"label": "Genetic Correlation (rg)"})
    ax2.set_title("Genetic Correlations by Source", fontweight="bold")
    ax2.axhline(y=len(FINNGEN_PHENOS), color="black", linewidth=2)
    ax2.axvline(x=len(FINNGEN_PHENOS), color="black", linewidth=2)

    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(gene_res["gene_overlap_matrix"].astype(float), annot=True, fmt=".2f",
                cmap="YlOrRd", vmin=0, vmax=1, ax=ax3, square=True,
                cbar_kws={"label": "Jaccard"})
    ax3.set_title("Top Gene Overlap (Jaccard)", fontweight="bold")
    ax3.axhline(y=len(FINNGEN_PHENOS), color="black", linewidth=2)
    ax3.axvline(x=len(FINNGEN_PHENOS), color="black", linewidth=2)

    ax4 = fig.add_subplot(2, 2, 4)
    rg_pop = concordance.get("pop_femaleprolapse_rg", {})
    cats = ["Within FinnGen", "Between Cohorts", "POP↔F.Prolapse"]
    vals = [
        arch["rg_comparison"].get("within_finngen_mean") or 0.0,
        arch["rg_comparison"].get("between_cohort_mean") or 0.0,
        rg_pop.get("rg", 0.0),
    ]
    errs = [
        arch["rg_comparison"].get("within_finngen_std") or 0.0,
        arch["rg_comparison"].get("between_cohort_std") or 0.0,
        rg_pop.get("se", 0.0),
    ]
    bars = ax4.bar(cats, vals, yerr=errs, color=["#E64B35", "#00A087", "#3C5488"], alpha=0.8, capsize=5)
    ax4.set_ylabel("Mean Genetic Correlation")
    ax4.set_title("Cross-Cohort Validation", fontweight="bold")
    ax4.axhline(y=0, color="gray", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"cross_cohort_validation.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved figures -> {FIG}")


def main() -> None:
    setup_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cross-Cohort Validation (FinnGen vs GWAS Catalog)")
    print("=" * 60)

    ldsc = load_ldsc()
    magma = load_magma()
    if ldsc is None or magma is None:
        return

    h2 = heritability_table(ldsc)
    arch = architecture_summary(ldsc, h2)
    gene_res = gene_overlap(magma)
    conc = pop_concordance(ldsc, magma)

    print(f"\n  FinnGen mean h2: {arch['h2_comparison']['finngen_mean']}")
    print(f"  GWAS-Catalog mean h2: {arch['h2_comparison']['gwas_catalog_mean']}")
    print(f"  Within-FinnGen mean rg: {arch['rg_comparison']['within_finngen_mean']}")
    print(f"  Between-cohort mean rg: {arch['rg_comparison']['between_cohort_mean']}")
    if "pop_femaleprolapse_rg" in conc:
        rg = conc["pop_femaleprolapse_rg"]
        print(f"  POP-FemaleProlapse rg: {rg['rg']:.3f} (p={rg['p']:.2e})")
    print(f"  Mean cross-cohort gene Jaccard: {gene_res['mean_between_overlap']:.3f}")
    print(f"  Shared genes: {len(gene_res['shared_genes'])}")

    h2_df = pd.DataFrame([
        {"Phenotype": p, "h2": v["h2"], "se": v["se"], "Source": "FinnGen" if p in FINNGEN_PHENOS else "GWAS_Catalog"}
        for p, v in h2.items()
    ])
    h2_df.to_csv(OUT / "heritability_comparison.csv", index=False)
    gene_res["gene_overlap_matrix"].to_csv(OUT / "gene_overlap_matrix.csv")
    pd.DataFrame([{
        "within_finngen_rg": arch["rg_comparison"]["within_finngen_mean"],
        "between_cohort_rg": arch["rg_comparison"]["between_cohort_mean"],
        "pop_femaleprolapse_rg": conc.get("pop_femaleprolapse_rg", {}).get("rg"),
        "mean_gene_overlap": gene_res["mean_between_overlap"],
        "n_shared_genes": len(gene_res["shared_genes"]),
    }]).to_csv(OUT / "validation_summary.csv", index=False)

    plot_summary(arch, h2, gene_res, conc, ldsc)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rg = conc.get("pop_femaleprolapse_rg", {})
    (LOGS_DIR / "17_cross_cohort_validation.md").write_text(
        "# Cross-Cohort Validation\n\n"
        f"**Date**: {pd.Timestamp.now():%Y-%m-%d}\n\n"
        f"FinnGen phenotypes: {', '.join(FINNGEN_PHENOS)}\n"
        f"GWAS-Catalog phenotypes: {', '.join(GWAS_CATALOG_PHENOS)}\n\n"
        f"- Within-FinnGen mean rg: {arch['rg_comparison']['within_finngen_mean']}\n"
        f"- Between-cohort mean rg: {arch['rg_comparison']['between_cohort_mean']}\n"
        f"- POP-FemaleProlapse rg: {rg.get('rg', 'NA')} (p={rg.get('p', 'NA')})\n"
        f"- Cross-cohort gene Jaccard: {gene_res['mean_between_overlap']:.3f}\n"
        f"- Shared genes: {', '.join(gene_res['shared_genes']) or 'none'}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
