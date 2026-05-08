#!/usr/bin/env python3
"""
Setup and run Mendelian Randomization analysis for pelvic floor phenotypes.

MR Analysis Plan:
1. Bidirectional MR between pelvic floor phenotypes
2. MR with potential risk factors (BMI, age at menarche, etc.)
3. MR with potential outcomes (quality of life, depression, etc.)

Using TwoSampleMR approach with summary statistics.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
SUMSTATS_DIR = BASE_DIR / "data/processed"
RESULTS_DIR = BASE_DIR / "results/mr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Our phenotypes
PHENOTYPES = {
    "POP": "POP_GRCh38.tsv.gz",
    "BPH": "BPH_GRCh38.tsv.gz",
    "Bladder": "Bladder_GRCh38.tsv.gz",
    "Constipation": "Constipation_GRCh38.tsv.gz",
    "FemaleProlapse": "FemaleProlapse_GRCh38.tsv.gz",
    "Incontinence": "Incontinence_GRCh38.tsv.gz",
}

# Sample sizes for each phenotype
SAMPLE_SIZES = {
    "POP": 574377,
    "BPH": 501137,
    "Bladder": 503550,
    "Constipation": 501956,
    "FemaleProlapse": 503074,
    "Incontinence": 430019,
}

# Significance threshold for instrument selection
P_THRESHOLD = 5e-8


def load_sumstats(pheno):
    """Load summary statistics for a phenotype."""
    filepath = SUMSTATS_DIR / PHENOTYPES[pheno]
    df = pd.read_csv(filepath, sep='\t', compression='gzip')
    return df


def select_instruments(df, p_threshold=P_THRESHOLD):
    """Select genetic instruments (significant SNPs)."""
    instruments = df[df['P'] < p_threshold].copy()
    return instruments


def ld_clump_simple(df, r2_threshold=0.001, window_kb=10000):
    """
    Simple LD clumping by selecting top SNPs per region.
    For proper clumping, should use PLINK with LD reference.
    This is a simplified version based on distance only.
    """
    df_sorted = df.sort_values('P')
    selected = []
    used_positions = set()

    for _, row in df_sorted.iterrows():
        pos_key = (row['CHR'], row['POS'] // (window_kb * 1000))
        if pos_key not in used_positions:
            selected.append(row)
            used_positions.add(pos_key)

    return pd.DataFrame(selected)


def harmonize_data(exposure_df, outcome_df):
    """
    Harmonize exposure and outcome data.
    Align effect alleles and flip effects if needed.
    """
    # Merge on SNP
    merged = exposure_df.merge(
        outcome_df,
        on='SNP',
        suffixes=('_exp', '_out')
    )

    if len(merged) == 0:
        return None

    # Check allele alignment
    harmonized = []
    for _, row in merged.iterrows():
        # Simple case: same alleles
        if (row['A1_exp'] == row['A1_out'] and row['A2_exp'] == row['A2_out']):
            harmonized.append({
                'SNP': row['SNP'],
                'beta_exp': row['BETA_exp'],
                'se_exp': row['SE_exp'],
                'beta_out': row['BETA_out'],
                'se_out': row['SE_out'],
                'eaf_exp': row.get('EAF_exp', 0.5),
            })
        # Flipped alleles
        elif (row['A1_exp'] == row['A2_out'] and row['A2_exp'] == row['A1_out']):
            harmonized.append({
                'SNP': row['SNP'],
                'beta_exp': row['BETA_exp'],
                'se_exp': row['SE_exp'],
                'beta_out': -row['BETA_out'],  # Flip effect
                'se_out': row['SE_out'],
                'eaf_exp': row.get('EAF_exp', 0.5),
            })

    return pd.DataFrame(harmonized)


def ivw_mr(beta_exp, se_exp, beta_out, se_out):
    """
    Inverse Variance Weighted MR.
    Returns: beta, se, p-value
    """
    # Wald ratios
    ratio = beta_out / beta_exp
    ratio_se = np.abs(se_out / beta_exp)

    # IVW (fixed effects)
    weights = 1 / (ratio_se ** 2)
    beta_ivw = np.sum(weights * ratio) / np.sum(weights)
    se_ivw = np.sqrt(1 / np.sum(weights))

    # Z-score and p-value
    z = beta_ivw / se_ivw
    p = 2 * stats.norm.sf(np.abs(z))

    return beta_ivw, se_ivw, p


def weighted_median_mr(beta_exp, se_exp, beta_out, se_out):
    """
    Weighted Median MR (more robust to pleiotropy).
    """
    ratio = beta_out / beta_exp
    ratio_se = np.abs(se_out / beta_exp)
    weights = 1 / (ratio_se ** 2)

    # Sort by ratio
    idx = np.argsort(ratio)
    ratio_sorted = ratio[idx]
    weights_sorted = weights[idx]

    # Cumulative weights
    cum_weights = np.cumsum(weights_sorted) / np.sum(weights_sorted)

    # Find median
    median_idx = np.searchsorted(cum_weights, 0.5)
    beta_wm = ratio_sorted[median_idx]

    # Bootstrap SE (simplified)
    se_wm = np.std(ratio) / np.sqrt(len(ratio))

    z = beta_wm / se_wm
    p = 2 * stats.norm.sf(np.abs(z))

    return beta_wm, se_wm, p


def mr_egger(beta_exp, se_exp, beta_out, se_out):
    """
    MR-Egger regression (tests for pleiotropy).
    Returns: beta, se, p-value, intercept, intercept_p
    """
    # Egger regression: beta_out = intercept + slope * beta_exp
    weights = 1 / (se_out ** 2)

    # Weighted linear regression
    X = np.column_stack([np.ones(len(beta_exp)), beta_exp])
    W = np.diag(weights)

    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    coeffs = XtWX_inv @ X.T @ W @ beta_out

    intercept = coeffs[0]
    slope = coeffs[1]

    # Standard errors
    residuals = beta_out - X @ coeffs
    mse = np.sum(weights * residuals**2) / (len(beta_out) - 2)
    se_coeffs = np.sqrt(np.diag(XtWX_inv) * mse)

    se_intercept = se_coeffs[0]
    se_slope = se_coeffs[1]

    # P-values
    z_slope = slope / se_slope
    p_slope = 2 * stats.norm.sf(np.abs(z_slope))

    z_intercept = intercept / se_intercept
    p_intercept = 2 * stats.norm.sf(np.abs(z_intercept))

    return slope, se_slope, p_slope, intercept, p_intercept


def run_mr_analysis(exposure, outcome, exposure_df, outcome_df):
    """Run full MR analysis between two phenotypes."""
    print(f"\n{'='*60}")
    print(f"MR: {exposure} -> {outcome}")
    print(f"{'='*60}")

    # Select instruments
    instruments = select_instruments(exposure_df)
    print(f"Genome-wide significant SNPs: {len(instruments)}")

    if len(instruments) == 0:
        print("No significant instruments found!")
        return None

    # LD clumping (simplified)
    instruments_clumped = ld_clump_simple(instruments)
    print(f"After clumping: {len(instruments_clumped)}")

    if len(instruments_clumped) < 3:
        print("Insufficient instruments for MR!")
        return None

    # Harmonize
    harmonized = harmonize_data(instruments_clumped, outcome_df)
    if harmonized is None or len(harmonized) < 3:
        print("Insufficient harmonized SNPs!")
        return None

    print(f"Harmonized SNPs: {len(harmonized)}")

    # Extract arrays
    beta_exp = harmonized['beta_exp'].values
    se_exp = harmonized['se_exp'].values
    beta_out = harmonized['beta_out'].values
    se_out = harmonized['se_out'].values

    # Run MR methods
    results = {'exposure': exposure, 'outcome': outcome, 'n_snps': len(harmonized)}

    # IVW
    beta_ivw, se_ivw, p_ivw = ivw_mr(beta_exp, se_exp, beta_out, se_out)
    results['ivw_beta'] = beta_ivw
    results['ivw_se'] = se_ivw
    results['ivw_p'] = p_ivw
    print(f"IVW: beta={beta_ivw:.4f}, SE={se_ivw:.4f}, P={p_ivw:.2e}")

    # Weighted Median
    beta_wm, se_wm, p_wm = weighted_median_mr(beta_exp, se_exp, beta_out, se_out)
    results['wm_beta'] = beta_wm
    results['wm_se'] = se_wm
    results['wm_p'] = p_wm
    print(f"Weighted Median: beta={beta_wm:.4f}, SE={se_wm:.4f}, P={p_wm:.2e}")

    # MR-Egger
    if len(harmonized) >= 5:
        beta_egger, se_egger, p_egger, intercept, p_intercept = mr_egger(
            beta_exp, se_exp, beta_out, se_out
        )
        results['egger_beta'] = beta_egger
        results['egger_se'] = se_egger
        results['egger_p'] = p_egger
        results['egger_intercept'] = intercept
        results['egger_intercept_p'] = p_intercept
        print(f"MR-Egger: beta={beta_egger:.4f}, P={p_egger:.2e}")
        print(f"  Intercept={intercept:.4f}, P={p_intercept:.2e} (pleiotropy test)")

    return results


def main():
    print("="*60)
    print("Mendelian Randomization Analysis")
    print("Pelvic Floor GWAS - Phase 6")
    print("="*60)

    # Load all summary statistics
    print("\nLoading summary statistics...")
    sumstats = {}
    for pheno in PHENOTYPES:
        print(f"  {pheno}...")
        sumstats[pheno] = load_sumstats(pheno)

    # Bidirectional MR between phenotypes
    print("\n" + "="*60)
    print("Bidirectional MR Between Phenotypes")
    print("="*60)

    # Key pairs to test
    pairs = [
        # Female phenotypes
        ("POP", "FemaleProlapse"),
        ("POP", "Incontinence"),
        ("FemaleProlapse", "Incontinence"),

        # Male vs Female
        ("BPH", "POP"),
        ("BPH", "Incontinence"),

        # Constipation relationships
        ("Constipation", "POP"),
        ("Constipation", "FemaleProlapse"),
    ]

    all_results = []

    for exp, out in pairs:
        # Forward direction
        result = run_mr_analysis(exp, out, sumstats[exp], sumstats[out])
        if result:
            all_results.append(result)

        # Reverse direction
        result = run_mr_analysis(out, exp, sumstats[out], sumstats[exp])
        if result:
            all_results.append(result)

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_DIR / "mr_bidirectional_results.csv", index=False)
        print(f"\nResults saved to: {RESULTS_DIR / 'mr_bidirectional_results.csv'}")

        # Summary
        print("\n" + "="*60)
        print("Summary of Significant Results (IVW P < 0.05)")
        print("="*60)
        sig = results_df[results_df['ivw_p'] < 0.05]
        if len(sig) > 0:
            for _, row in sig.iterrows():
                direction = "+" if row['ivw_beta'] > 0 else "-"
                print(f"{row['exposure']} -> {row['outcome']}: "
                      f"beta={row['ivw_beta']:.3f} ({direction}), P={row['ivw_p']:.2e}")
        else:
            print("No significant causal relationships found.")

    print("\n" + "="*60)
    print("MR Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
