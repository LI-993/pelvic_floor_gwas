#!/usr/bin/env python3
"""Bayesian shrinkage on top of the P+T weights from 01_prs_development.

A pure-Python alternative to LDpred-inf: each SNP's effect is shrunk under a
N(0, h2/M) prior, with the posterior mean used as the new weight. Outputs
phenotype-level Bayesian PRS files plus a multi-trait female PRS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, LOGS_DIR, PHENOTYPES, RESULTS_DIR
from utils.plotting import setup_publication_style

OUT = RESULTS_DIR / "prs_bayesian"
PRS_DIR = RESULTS_DIR / "prs"
FIG_DIR = FIGURES_DIR / "prs"

THRESHOLDS = ["5e-08", "1e-05", "0.0001", "0.001", "0.01", "0.05", "0.1", "0.5", "1.0"]
OPTIMAL_THRESHOLD = "0.01"  # default working threshold for shrinkage
DEFAULT_H2 = 0.02
DEFAULT_N = 500_000
FEMALE_PHENOS = ("POP", "FemaleProlapse", "Incontinence")


def load_pt_prs() -> dict[str, dict[str, pd.DataFrame]]:
    """Load all P+T PRS files written by 01_prs_development."""
    prs: dict[str, dict[str, pd.DataFrame]] = {}
    for pheno in PHENOTYPES:
        prs[pheno] = {}
        for thresh in THRESHOLDS:
            path = PRS_DIR / f"{pheno}_PRS_p{thresh}.txt"
            if path.exists():
                prs[pheno][thresh] = pd.read_csv(path, sep="\t")
    return prs


def bayesian_shrinkage(beta: np.ndarray, se: np.ndarray, h2: float, M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and shrinkage factor under a N(0, h2/M) prior."""
    sigma2 = h2 / max(M, 1)
    posterior_var = 1 / (1 / sigma2 + N / se**2)
    posterior_mean = posterior_var * (N * beta / se**2)
    shrinkage = posterior_var / (posterior_var + 1 / sigma2)
    return posterior_mean, shrinkage


def apply_shrinkage(prs_df: pd.DataFrame, h2: float = DEFAULT_H2, n: int = DEFAULT_N) -> pd.DataFrame:
    if prs_df.empty:
        return prs_df
    beta = prs_df["BETA"].values
    se = prs_df["SE"].values if "SE" in prs_df.columns else np.abs(beta) * 0.1
    M = max(len(prs_df) * 10, 1)  # rough effective-SNP estimate
    post_mean, shrink = bayesian_shrinkage(beta, se, h2=h2, M=M, N=n)

    out = prs_df.copy()
    out["BETA_original"] = out["BETA"]
    out["BETA"] = post_mean
    out["shrinkage_factor"] = shrink
    return out


def comparison_row(pt_prs: pd.DataFrame, bayes_prs: pd.DataFrame, pheno: str) -> dict:
    return {
        "phenotype": pheno,
        "pt_n_snps": len(pt_prs),
        "bayes_n_snps": len(bayes_prs),
        "pt_var_beta": float(pt_prs["BETA"].var()) if "BETA" in pt_prs.columns else 0.0,
        "bayes_var_beta": float(bayes_prs["BETA"].var()) if "BETA" in bayes_prs.columns else 0.0,
        "mean_shrinkage": float(bayes_prs["shrinkage_factor"].mean()) if "shrinkage_factor" in bayes_prs.columns else None,
    }


def make_multi_trait_prs(prs_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight averaging across traits at the SNP level."""
    weight = 1 / len(prs_dict)
    snp_data: dict[str, dict] = {}
    for pheno, df in prs_dict.items():
        for _, row in df.iterrows():
            rec = snp_data.setdefault(row["SNP"], {"A1": row["A1"], "beta_sum": 0.0, "n_traits": 0})
            rec["beta_sum"] += row["BETA"] * weight
            rec["n_traits"] += 1
    return pd.DataFrame([
        {"SNP": snp, "A1": v["A1"], "BETA": v["beta_sum"] / v["n_traits"], "N_TRAITS": v["n_traits"]}
        for snp, v in snp_data.items()
    ])


def plot_summary(comparisons: list[dict], pt_prs: dict[str, dict[str, pd.DataFrame]]) -> None:
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(comp_df))
        width = 0.35
        axes[0].bar(x - width / 2, comp_df["pt_var_beta"], width, label="P+T", color="#E64B35", alpha=0.8)
        axes[0].bar(x + width / 2, comp_df["bayes_var_beta"], width, label="Bayesian", color="#4DBBD5", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comp_df["phenotype"], rotation=45, ha="right")
        axes[0].set_ylabel("Variance of β weights")
        axes[0].set_title("Effect-size variance: P+T vs Bayesian", fontweight="bold")
        axes[0].legend()

        if comp_df["mean_shrinkage"].notna().any():
            axes[1].bar(comp_df["phenotype"], comp_df["mean_shrinkage"], color="#00A087", alpha=0.8)
            axes[1].set_ylabel("Mean shrinkage factor")
            axes[1].set_title("Bayesian shrinkage by phenotype", fontweight="bold")
            axes[1].set_xticklabels(comp_df["phenotype"], rotation=45, ha="right")

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"prs_method_comparison.{ext}", bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    thresholds = ["5e-08", "1e-05", "0.0001", "0.001", "0.01"]
    for pheno in pt_prs:
        counts = [len(pt_prs[pheno].get(t, [])) for t in thresholds]
        ax.plot(thresholds, counts, "o-", label=pheno, markersize=8)
    ax.set_xlabel("P-value Threshold")
    ax.set_ylabel("Number of SNPs")
    ax.set_yscale("log")
    ax.set_title("PRS SNP Count by P-value Threshold", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"prs_snp_counts.{ext}", bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bayesian PRS Shrinkage")
    print("=" * 60)

    pt_prs = load_pt_prs()

    print("\nApplying Bayesian shrinkage at p<{0}...".format(OPTIMAL_THRESHOLD))
    bayes: dict[str, pd.DataFrame] = {}
    comparisons: list[dict] = []
    for pheno, thresholds in pt_prs.items():
        df = thresholds.get(OPTIMAL_THRESHOLD)
        if df is None or df.empty:
            continue
        out_df = apply_shrinkage(df)
        bayes[pheno] = out_df
        comparisons.append(comparison_row(df, out_df, pheno))
        out_path = OUT / f"{pheno}_bayesian_prs.txt"
        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"  {pheno}: {len(out_df)} SNPs, mean shrinkage={out_df['shrinkage_factor'].mean():.4f}")

    print("\nMulti-trait female PRS...")
    female = {p: bayes[p] for p in FEMALE_PHENOS if p in bayes}
    if female:
        multi = make_multi_trait_prs(female)
        multi_path = OUT / "multi_trait_female_prs.txt"
        multi.to_csv(multi_path, sep="\t", index=False)
        print(f"  {len(multi)} SNPs (multi-trait SNPs: {(multi['N_TRAITS'] > 1).sum()})")

    if comparisons:
        pd.DataFrame(comparisons).to_csv(OUT / "method_comparison.csv", index=False)

    plot_summary(comparisons, pt_prs)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "15_bayesian_prs.md").write_text(
        f"# Bayesian PRS Development\n\n"
        f"**Date**: {pd.Timestamp.now():%Y-%m-%d}\n\n"
        "Pure-Python Bayesian shrinkage on top of P+T weights.\n"
        f"Phenotypes processed: {len(comparisons)}\n"
        f"Multi-trait female PRS: {'yes' if female else 'no'}\n",
        encoding="utf-8",
    )

    print(f"\nResults: {OUT}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
