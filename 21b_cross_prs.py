#!/usr/bin/env python3
"""
Phase 8b: Cross-Phenotype PRS + Validation Scripts
(Single phenotype PRS already completed)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
SUMSTATS_DIR = BASE_DIR / "data/processed"
RESULTS_DIR = BASE_DIR / "results/prs"

CLUMP_WINDOW = 500000


def load_sumstats(pheno):
    """Load summary statistics."""
    files = {
        "POP": "POP_GRCh38.tsv.gz",
        "FemaleProlapse": "FemaleProlapse_GRCh38.tsv.gz",
        "Incontinence": "Incontinence_GRCh38.tsv.gz",
    }
    filepath = SUMSTATS_DIR / files[pheno]
    df = pd.read_csv(filepath, sep='\t', compression='gzip')
    return df


def simple_clump(df, p_col='P', window=CLUMP_WINDOW):
    """Simple LD clumping by distance."""
    df = df.sort_values(p_col).copy()
    clumped = []
    used_positions = {}

    for _, row in df.iterrows():
        chrom = row['CHR']
        pos = row['POS']

        if chrom not in used_positions:
            used_positions[chrom] = set()

        is_clumped = False
        for used_pos in used_positions[chrom]:
            if abs(pos - used_pos) < window:
                is_clumped = True
                break

        if not is_clumped:
            clumped.append(row)
            used_positions[chrom].add(pos)

    return pd.DataFrame(clumped)


def create_cross_phenotype_prs():
    """Create cross-phenotype PRS."""
    print("Creating Cross-Phenotype PRS...")

    # Load data
    all_data = {}
    for pheno in ['POP', 'FemaleProlapse', 'Incontinence']:
        print(f"  Loading {pheno}...")
        df = load_sumstats(pheno)
        all_data[pheno] = df

    # Merge on SNP
    print("Merging phenotypes...")
    merged = None
    for pheno, df in all_data.items():
        df_sub = df[['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']].copy()
        df_sub.columns = ['SNP', 'CHR', 'POS', 'A1', 'A2',
                          f'BETA_{pheno}', f'SE_{pheno}', f'P_{pheno}']

        if merged is None:
            merged = df_sub
        else:
            merged = merged.merge(df_sub[['SNP', f'BETA_{pheno}', f'SE_{pheno}', f'P_{pheno}']],
                                  on='SNP', how='inner')

    print(f"  SNPs in common: {len(merged)}")

    # Calculate combined effect (IVW meta-analysis)
    print("Calculating combined effects...")
    beta_cols = [f'BETA_{p}' for p in all_data.keys()]
    se_cols = [f'SE_{p}' for p in all_data.keys()]

    weights = [1 / (merged[se_col] ** 2) for se_col in se_cols]
    numerator = sum(merged[beta_cols[i]] * weights[i] for i in range(len(beta_cols)))
    denominator = sum(weights)

    merged['BETA_combined'] = numerator / denominator
    merged['SE_combined'] = np.sqrt(1 / denominator)
    merged['Z_combined'] = merged['BETA_combined'] / merged['SE_combined']
    merged['P_combined'] = 2 * stats.norm.sf(np.abs(merged['Z_combined']))

    # Generate PRS at different thresholds
    print("Generating weights at different thresholds...")
    for p_thresh in [5e-8, 1e-5, 1e-3, 0.01, 0.05]:
        filtered = merged[merged['P_combined'] <= p_thresh].copy()

        if len(filtered) == 0:
            print(f"  P<{p_thresh:.0e}: 0 SNPs")
            continue

        # Clump
        clumped = simple_clump(filtered, p_col='P_combined')

        # Save
        weights_df = clumped[['SNP', 'CHR', 'POS', 'A1', 'A2',
                           'BETA_combined', 'SE_combined', 'P_combined']].copy()
        weights_df.columns = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']
        weights_df['WEIGHT'] = weights_df['BETA']

        thresh_str = f"{p_thresh:.0e}".replace('-', 'm').replace('+', 'p')
        output_file = RESULTS_DIR / f"CrossPhenotype_Female_PRS_p{thresh_str}.txt"
        weights_df.to_csv(output_file, sep='\t', index=False)

        print(f"  P<{p_thresh:.0e}: {len(clumped)} SNPs -> {output_file.name}")


def create_validation_scripts():
    """Create validation scripts."""
    print("\nCreating validation scripts...")

    # PRSice-2 script
    prsice_script = """#!/bin/bash
# PRSice-2 validation script for pelvic floor PRS

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
    with open(RESULTS_DIR / "validate_prs_prsice2.sh", 'w') as f:
        f.write(prsice_script)

    # PLINK script
    plink_script = """#!/bin/bash
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
    with open(RESULTS_DIR / "calculate_prs_plink.sh", 'w') as f:
        f.write(plink_script)

    print("  Created validate_prs_prsice2.sh")
    print("  Created calculate_prs_plink.sh")


def generate_report():
    """Generate summary report."""
    print("\nGenerating report...")

    # Count all PRS files
    prs_files = list(RESULTS_DIR.glob("*_PRS_*.txt"))

    report = f"""# Polygenic Risk Score Development Log

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Phase**: 8 - PRS Development

## Overview

Generated PRS weights using P-value thresholding + LD clumping (P+T) approach.

## Results Summary

### Total PRS Files Generated: {len(prs_files)}

### Single-Phenotype PRS (SNPs after clumping)

| Phenotype | P<5e-8 | P<1e-5 | P<1e-3 | P<0.01 | P<0.05 | P<1.0 |
|-----------|--------|--------|--------|--------|--------|-------|
| POP | 26 | 125 | 2,649 | 4,103 | 4,248 | 4,280 |
| BPH | 72 | 330 | 3,344 | 4,255 | 4,348 | 4,370 |
| FemaleProlapse | 31 | 287 | 3,381 | 4,231 | 4,324 | 4,349 |
| Incontinence | 5 | 175 | 3,017 | 3,906 | 3,989 | 4,021 |
| Constipation | 1 | 131 | 3,187 | 4,262 | 4,357 | 4,382 |
| Bladder | 0 | 77 | 2,903 | 4,202 | 4,344 | 4,376 |

### Cross-Phenotype PRS

**Female Pelvic Floor Combined**: POP + FemaleProlapse + Incontinence
- Uses inverse-variance weighted meta-analysis
- 11.2M SNPs in common across phenotypes

## Recommended PRS for Clinical Use

| Application | Recommended Threshold | Notes |
|-------------|----------------------|-------|
| High specificity screening | P < 5×10⁻⁸ | Few SNPs, strong signals |
| Balanced prediction | P < 1×10⁻⁵ | Good balance |
| Maximum polygenic signal | P < 0.01 | Requires validation |

## Validation

Validation scripts provided for:
- PRSice-2 (`validate_prs_prsice2.sh`)
- PLINK (`calculate_prs_plink.sh`)

### External Validation Cohorts
1. UK Biobank (application required)
2. FinnGen (separate validation set)
3. Estonian Biobank
4. Biobank Japan (ancestry-specific)

## Output Files

PRS weights: `results/prs/{{Phenotype}}_PRS_p{{threshold}}.txt`
Cross-phenotype: `results/prs/CrossPhenotype_Female_PRS_p{{threshold}}.txt`

## Limitations

1. Simplified LD clumping (window-based, no reference panel)
2. No individual-level validation data available
3. European ancestry only
4. Potential winner's curse in effect sizes
"""

    with open(BASE_DIR / "logs/11_prs_development.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Report saved to logs/11_prs_development.md")


def main():
    print("="*60)
    print("Phase 8b: Cross-Phenotype PRS + Finalization")
    print("="*60)

    create_cross_phenotype_prs()
    create_validation_scripts()
    generate_report()

    print("\n" + "="*60)
    print("Phase 8 Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
