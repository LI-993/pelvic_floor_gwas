#!/usr/bin/env python3
"""Single-phenotype + cross-phenotype PRS via P+T (P-value thresholding).

Generates PRS weights at a panel of p-value thresholds for each phenotype,
plus a combined cross-phenotype PRS for the female pelvic-floor traits
(POP + FemaleProlapse + Incontinence) using inverse-variance weighted
meta-analysis. Distance-based clumping is from utils.clumping; for proper
LD-aware clumping use PLINK.

Also writes PRSice-2 / PLINK validation shell templates so external cohorts
can score the weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR, PHENOTYPES, RESULTS_DIR
from utils.clumping import simple_clump
from utils.io import load_sumstats

OUT = RESULTS_DIR / "prs"
P_THRESHOLDS = [5e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
CROSS_THRESHOLDS = [5e-8, 1e-5, 1e-3, 0.01, 0.05]
FEMALE_TRAITS = ["POP", "FemaleProlapse", "Incontinence"]


def thresh_str(p: float) -> str:
    return f"{p:.0e}".replace("-", "m").replace("+", "p")


def generate_single_phenotype(pheno: str, df: pd.DataFrame) -> list[dict]:
    print(f"\n  {pheno}")
    df = df.dropna(subset=["SNP", "BETA", "P"])
    rows: list[dict] = []
    for p_thr in P_THRESHOLDS:
        sub = df[df["P"] <= p_thr]
        if sub.empty:
            rows.append({"phenotype": pheno, "p_threshold": p_thr, "n_snps_raw": 0, "n_snps_clumped": 0})
            continue
        clumped = simple_clump(sub)
        weights = clumped[["SNP", "CHR", "POS", "A1", "A2", "BETA", "SE", "P"]].copy()
        weights["WEIGHT"] = weights["BETA"]
        out_path = OUT / f"{pheno}_PRS_p{thresh_str(p_thr)}.txt"
        weights.to_csv(out_path, sep="\t", index=False)
        rows.append({
            "phenotype": pheno,
            "p_threshold": p_thr,
            "n_snps_raw": len(sub),
            "n_snps_clumped": len(clumped),
            "output_file": out_path.name,
        })
        print(f"    P<{p_thr:.0e}: {len(sub):,} -> {len(clumped):,} -> {out_path.name}")
    return rows


def generate_cross_phenotype(sumstats: dict[str, pd.DataFrame]) -> list[dict]:
    print("\nCross-phenotype PRS (POP + FemaleProlapse + Incontinence)...")
    merged: pd.DataFrame | None = None
    for pheno in FEMALE_TRAITS:
        df = sumstats[pheno][["SNP", "CHR", "POS", "A1", "A2", "BETA", "SE", "P"]].copy()
        df.columns = ["SNP", "CHR", "POS", "A1", "A2", f"BETA_{pheno}", f"SE_{pheno}", f"P_{pheno}"]
        merged = df if merged is None else merged.merge(
            df[["SNP", f"BETA_{pheno}", f"SE_{pheno}", f"P_{pheno}"]], on="SNP", how="inner"
        )
    print(f"  SNPs in common: {len(merged):,}")

    se_cols = [f"SE_{p}" for p in FEMALE_TRAITS]
    beta_cols = [f"BETA_{p}" for p in FEMALE_TRAITS]
    weights = [1 / merged[c] ** 2 for c in se_cols]
    numerator = sum(merged[beta_cols[i]] * weights[i] for i in range(len(beta_cols)))
    denominator = sum(weights)

    merged["BETA_combined"] = numerator / denominator
    merged["SE_combined"] = np.sqrt(1 / denominator)
    merged["Z_combined"] = merged["BETA_combined"] / merged["SE_combined"]
    merged["P_combined"] = 2 * stats.norm.sf(np.abs(merged["Z_combined"]))

    rows: list[dict] = []
    for p_thr in CROSS_THRESHOLDS:
        sub = merged[merged["P_combined"] <= p_thr]
        if sub.empty:
            continue
        sub = sub.rename(columns={"P_combined": "P"})
        clumped = simple_clump(sub)
        out = clumped[["SNP", "CHR", "POS", "A1", "A2", "BETA_combined", "SE_combined", "P"]].copy()
        out.columns = ["SNP", "CHR", "POS", "A1", "A2", "BETA", "SE", "P"]
        out["WEIGHT"] = out["BETA"]
        path = OUT / f"CrossPhenotype_Female_PRS_p{thresh_str(p_thr)}.txt"
        out.to_csv(path, sep="\t", index=False)
        rows.append({"p_threshold": p_thr, "n_snps": len(clumped), "output_file": path.name})
        print(f"  P<{p_thr:.0e}: {len(clumped):,} -> {path.name}")
    return rows


def write_validation_scripts() -> None:
    prsice = """#!/bin/bash
# PRSice-2 validation script for pelvic-floor PRS
# Requires: PRSice-2, individual-level genotype data
TARGET_GENO="path/to/target_genotypes"
TARGET_PHENO="path/to/phenotype_file.txt"
PRS_WEIGHTS="path/to/prs_weights.txt"
OUTPUT_PREFIX="pelvic_floor_prs_validation"

PRSice_linux \\
    --base ${PRS_WEIGHTS} \\
    --target ${TARGET_GENO} \\
    --pheno ${TARGET_PHENO} \\
    --snp SNP --chr CHR --bp POS --A1 A1 --A2 A2 \\
    --stat WEIGHT --pvalue P --beta \\
    --binary-target T \\
    --out ${OUTPUT_PREFIX} \\
    --thread 4

echo "PRS validation complete!"
"""

    plink = """#!/bin/bash
# PLINK PRS scoring script
TARGET_GENO="path/to/target_genotypes"
PRS_WEIGHTS="path/to/prs_weights.txt"
OUTPUT_PREFIX="pelvic_floor_prs_scores"

plink \\
    --bfile ${TARGET_GENO} \\
    --score ${PRS_WEIGHTS} 1 4 8 header sum \\
    --out ${OUTPUT_PREFIX}

echo "PRS scoring complete!"
"""
    (OUT / "validate_prs_prsice2.sh").write_text(prsice)
    (OUT / "calculate_prs_plink.sh").write_text(plink)
    print(f"\nWrote validation scripts under {OUT}")


def write_log(single_results: list[dict], cross_results: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    n_files = len(list(OUT.glob("*_PRS_*.txt")))
    log = f"""# Polygenic Risk Score Development

**Date**: {pd.Timestamp.now():%Y-%m-%d}

## Summary
- Total PRS weight files: {n_files}
- Single-phenotype PRS rows: {len(single_results)}
- Cross-phenotype PRS rows: {len(cross_results)}

PRS weights at thresholds {P_THRESHOLDS} are written to `results/prs/`.
The cross-phenotype female PRS combines POP + FemaleProlapse + Incontinence
via inverse-variance-weighted meta-analysis.

## Limitations
- Distance-based clumping (no LD reference); use PLINK for proper clumping.
- No individual-level validation cohort available.
- European ancestry only.
"""
    (LOGS_DIR / "11_prs_development.md").write_text(log)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRS Development (P+T method)")
    print("=" * 60)

    sumstats: dict[str, pd.DataFrame] = {}
    print("\nLoading sumstats...")
    for pheno in PHENOTYPES:
        sumstats[pheno] = load_sumstats(pheno)

    print("\nSingle-phenotype PRS:")
    single_results: list[dict] = []
    for pheno in PHENOTYPES:
        single_results.extend(generate_single_phenotype(pheno, sumstats[pheno]))

    cross_results = generate_cross_phenotype(sumstats)
    write_validation_scripts()
    write_log(single_results, cross_results)

    print(f"\nTotal PRS files: {len(list(OUT.glob('*_PRS_*.txt')))}")


if __name__ == "__main__":
    main()
