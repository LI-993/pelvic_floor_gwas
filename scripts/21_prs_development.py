#!/usr/bin/env python3
"""
Phase 8: Polygenic Risk Score (PRS) Development

Generate PRS weights from GWAS summary statistics using:
1. P+T (P-value thresholding + LD clumping) approach
2. Cross-phenotype PRS combining multiple traits
3. Prepare files for external validation

Note: Without individual-level genotype data, we can only generate weights.
Validation requires external cohorts (e.g., UK Biobank, FinnGen validation set).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
SUMSTATS_DIR = BASE_DIR / "data/processed"
RESULTS_DIR = BASE_DIR / "results/prs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Phenotypes
PHENOTYPES = {
    "POP": "POP_GRCh38.tsv.gz",
    "BPH": "BPH_GRCh38.tsv.gz",
    "FemaleProlapse": "FemaleProlapse_GRCh38.tsv.gz",
    "Incontinence": "Incontinence_GRCh38.tsv.gz",
    "Constipation": "Constipation_GRCh38.tsv.gz",
    "Bladder": "Bladder_GRCh38.tsv.gz",
}

# P-value thresholds for P+T approach
P_THRESHOLDS = [5e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]

# LD clumping parameters (simplified window-based)
CLUMP_WINDOW = 500000  # 500kb window
CLUMP_R2 = 0.1  # r2 threshold (approximated by distance)


def load_sumstats(pheno):
    """Load and prepare summary statistics."""
    filepath = SUMSTATS_DIR / PHENOTYPES[pheno]
    df = pd.read_csv(filepath, sep='\t', compression='gzip')

    # Ensure required columns
    required = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']
    if not all(col in df.columns for col in required):
        print(f"  Warning: Missing columns for {pheno}")
        return None

    # Remove missing values
    df = df.dropna(subset=['SNP', 'BETA', 'P'])

    return df


def simple_clump(df, window=CLUMP_WINDOW):
    """
    Simple LD clumping by distance (window-based).
    For proper clumping, use PLINK with LD reference panel.
    """
    # Sort by P-value
    df = df.sort_values('P').copy()

    clumped = []
    used_positions = {}  # chr -> set of used positions

    for _, row in df.iterrows():
        chrom = row['CHR']
        pos = row['POS']

        if chrom not in used_positions:
            used_positions[chrom] = set()

        # Check if position is within window of any used SNP
        is_clumped = False
        for used_pos in used_positions[chrom]:
            if abs(pos - used_pos) < window:
                is_clumped = True
                break

        if not is_clumped:
            clumped.append(row)
            used_positions[chrom].add(pos)

    return pd.DataFrame(clumped)


def generate_prs_weights(pheno, df):
    """Generate PRS weights at different P-value thresholds."""
    print(f"\n  Generating PRS weights for {pheno}...")

    results = []

    for p_thresh in P_THRESHOLDS:
        # Filter by P-value
        filtered = df[df['P'] <= p_thresh].copy()

        if len(filtered) == 0:
            results.append({
                'phenotype': pheno,
                'p_threshold': p_thresh,
                'n_snps_raw': 0,
                'n_snps_clumped': 0
            })
            continue

        # LD clumping
        clumped = simple_clump(filtered)

        # Save weights file
        weights = clumped[['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']].copy()
        weights['WEIGHT'] = weights['BETA']

        # Save to file
        thresh_str = f"{p_thresh:.0e}".replace('-', 'm').replace('+', 'p')
        output_file = RESULTS_DIR / f"{pheno}_PRS_p{thresh_str}.txt"
        weights.to_csv(output_file, sep='\t', index=False)

        results.append({
            'phenotype': pheno,
            'p_threshold': p_thresh,
            'n_snps_raw': len(filtered),
            'n_snps_clumped': len(clumped),
            'output_file': output_file.name
        })

        print(f"    P<{p_thresh:.0e}: {len(filtered)} SNPs -> {len(clumped)} after clumping")

    return results


def create_cross_phenotype_prs():
    """
    Create cross-phenotype PRS combining signals from multiple traits.
    Uses meta-analysis approach to combine effects.
    """
    print("\n" + "="*60)
    print("Creating Cross-Phenotype Pelvic Floor PRS")
    print("="*60)

    # Load all summary stats
    all_data = {}
    for pheno in ['POP', 'FemaleProlapse', 'Incontinence']:  # Female-relevant
        df = load_sumstats(pheno)
        if df is not None:
            all_data[pheno] = df

    if len(all_data) < 2:
        print("Not enough phenotypes for cross-phenotype PRS")
        return None

    # Merge on SNP
    print("\nMerging phenotypes...")
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

    # Calculate combined effect using inverse-variance weighted meta-analysis
    print("\nCalculating combined effects...")

    beta_cols = [f'BETA_{p}' for p in all_data.keys()]
    se_cols = [f'SE_{p}' for p in all_data.keys()]

    # Inverse variance weights
    weights = []
    for se_col in se_cols:
        w = 1 / (merged[se_col] ** 2)
        weights.append(w)

    # Combined beta
    numerator = sum(merged[beta_cols[i]] * weights[i] for i in range(len(beta_cols)))
    denominator = sum(weights)

    merged['BETA_combined'] = numerator / denominator
    merged['SE_combined'] = np.sqrt(1 / denominator)
    merged['Z_combined'] = merged['BETA_combined'] / merged['SE_combined']
    merged['P_combined'] = 2 * stats.norm.sf(np.abs(merged['Z_combined']))

    # Generate PRS at different thresholds
    print("\nGenerating cross-phenotype PRS weights...")

    results = []
    for p_thresh in [5e-8, 1e-5, 1e-3, 0.01, 0.05]:
        filtered = merged[merged['P_combined'] <= p_thresh].copy()

        if len(filtered) == 0:
            continue

        # Rename P_combined to P for clumping function
        filtered = filtered.rename(columns={'P_combined': 'P'})

        # Clump
        clumped = simple_clump(filtered)

        # Save
        weights = clumped[['SNP', 'CHR', 'POS', 'A1', 'A2',
                           'BETA_combined', 'SE_combined', 'P']].copy()
        weights.columns = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']
        weights['WEIGHT'] = weights['BETA']

        thresh_str = f"{p_thresh:.0e}".replace('-', 'm').replace('+', 'p')
        output_file = RESULTS_DIR / f"CrossPhenotype_Female_PRS_p{thresh_str}.txt"
        weights.to_csv(output_file, sep='\t', index=False)

        results.append({
            'p_threshold': p_thresh,
            'n_snps': len(clumped),
            'output_file': output_file.name
        })

        print(f"  P<{p_thresh:.0e}: {len(clumped)} SNPs")

    # Also create BPH-specific PRS
    print("\nCreating BPH-specific PRS...")
    bph_df = load_sumstats('BPH')
    if bph_df is not None:
        for p_thresh in [5e-8, 1e-5, 1e-3, 0.01]:
            filtered = bph_df[bph_df['P'] <= p_thresh].copy()
            if len(filtered) > 0:
                clumped = simple_clump(filtered)
                weights = clumped[['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']].copy()
                weights['WEIGHT'] = weights['BETA']

                thresh_str = f"{p_thresh:.0e}".replace('-', 'm').replace('+', 'p')
                output_file = RESULTS_DIR / f"BPH_PRS_p{thresh_str}.txt"
                weights.to_csv(output_file, sep='\t', index=False)
                print(f"  BPH P<{p_thresh:.0e}: {len(clumped)} SNPs")

    return results


def create_validation_scripts():
    """Create scripts for PRS validation in external cohorts."""

    # PRSice-2 script template
    prsice_script = """#!/bin/bash
# PRSice-2 validation script for pelvic floor PRS
# Requires: PRSice-2, individual-level genotype data

# Input files (modify paths as needed)
TARGET_GENO="path/to/target_genotypes"  # PLINK format prefix
TARGET_PHENO="path/to/phenotype_file.txt"  # FID IID Phenotype
PRS_WEIGHTS="path/to/prs_weights.txt"

# Output
OUTPUT_PREFIX="pelvic_floor_prs_validation"

# Run PRSice-2
PRSice_linux \\
    --base ${PRS_WEIGHTS} \\
    --target ${TARGET_GENO} \\
    --pheno ${TARGET_PHENO} \\
    --snp SNP \\
    --chr CHR \\
    --bp POS \\
    --A1 A1 \\
    --A2 A2 \\
    --stat WEIGHT \\
    --pvalue P \\
    --beta \\
    --binary-target T \\
    --out ${OUTPUT_PREFIX} \\
    --thread 4

echo "PRS validation complete!"
echo "Results in ${OUTPUT_PREFIX}.summary"
"""

    with open(RESULTS_DIR / "validate_prs_prsice2.sh", 'w') as f:
        f.write(prsice_script)

    # PLINK score script
    plink_script = """#!/bin/bash
# PLINK PRS scoring script
# Requires: PLINK 1.9/2.0, individual-level genotype data

# Input files
TARGET_GENO="path/to/target_genotypes"  # PLINK format prefix
PRS_WEIGHTS="path/to/prs_weights.txt"

# Output
OUTPUT_PREFIX="pelvic_floor_prs_scores"

# Run PLINK score
plink \\
    --bfile ${TARGET_GENO} \\
    --score ${PRS_WEIGHTS} 1 4 8 header sum \\
    --out ${OUTPUT_PREFIX}

# Columns: 1=SNP, 4=A1, 8=WEIGHT

echo "PRS scoring complete!"
echo "Scores in ${OUTPUT_PREFIX}.profile"
"""

    with open(RESULTS_DIR / "calculate_prs_plink.sh", 'w') as f:
        f.write(plink_script)

    print("\nValidation scripts created:")
    print(f"  - {RESULTS_DIR / 'validate_prs_prsice2.sh'}")
    print(f"  - {RESULTS_DIR / 'calculate_prs_plink.sh'}")


def generate_summary_report(all_results):
    """Generate summary report for PRS development."""

    lines = []
    lines.append("# Polygenic Risk Score Development Log")
    lines.append(f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append("**Phase**: 8 - PRS Development")

    lines.append("\n## Overview")
    lines.append("""
Developed polygenic risk scores (PRS) for pelvic floor disorders using:
- P-value thresholding + LD clumping (P+T) approach
- Cross-phenotype meta-analysis for combined scores
- Simplified LD clumping (500kb windows, no reference panel)
""")

    lines.append("\n## Methods")
    lines.append("""
### Single-Phenotype PRS
1. Filter SNPs by P-value threshold
2. Apply window-based LD clumping (500kb)
3. Use effect sizes (BETA) as weights

### Cross-Phenotype PRS
1. Meta-analyze effects across phenotypes (IVW method)
2. Filter by combined P-value
3. LD clump and generate weights

### P-value Thresholds Tested
- Genome-wide significant: P < 5×10⁻⁸
- Suggestive: P < 1×10⁻⁵
- Moderate: P < 0.001, 0.01
- Liberal: P < 0.05, 0.1, 0.5, 1.0
""")

    lines.append("\n## Results Summary")

    # Summary table
    lines.append("\n### SNPs per PRS by Threshold")
    lines.append("\n| Phenotype | P<5e-8 | P<1e-5 | P<1e-3 | P<0.01 | P<0.05 |")
    lines.append("|-----------|--------|--------|--------|--------|--------|")

    for pheno in PHENOTYPES.keys():
        pheno_results = [r for r in all_results if r.get('phenotype') == pheno]
        if pheno_results:
            row = f"| {pheno} |"
            for thresh in [5e-8, 1e-5, 1e-3, 0.01, 0.05]:
                match = [r for r in pheno_results if r.get('p_threshold') == thresh]
                if match:
                    row += f" {match[0]['n_snps_clumped']} |"
                else:
                    row += " - |"
            lines.append(row)

    lines.append("\n### Recommended PRS")
    lines.append("""
Based on previous literature, the following thresholds are recommended:

| Use Case | Threshold | Expected Performance |
|----------|-----------|---------------------|
| Clinical screening | P < 5×10⁻⁸ | High specificity, low sensitivity |
| Research/discovery | P < 1×10⁻⁵ | Balanced |
| Maximum prediction | P < 0.01-0.05 | Requires validation |
""")

    lines.append("\n## Cross-Phenotype PRS")
    lines.append("""
### Female Pelvic Floor Combined PRS
Combined: POP + FemaleProlapse + Incontinence

This cross-phenotype PRS captures shared genetic liability for:
- Pelvic organ prolapse
- Urinary incontinence
- Related pelvic floor dysfunction

### BPH-Specific PRS
Optimized for male benign prostatic hyperplasia.
""")

    lines.append("\n## Validation Requirements")
    lines.append("""
### External Validation Needed
PRS validation requires individual-level genotype + phenotype data:

1. **UK Biobank** (application required)
   - Large sample size (N > 400,000)
   - Pelvic floor phenotypes available via ICD codes

2. **FinnGen** (if validation set available)
   - Same source as discovery (potential overfitting)
   - Use independent samples if possible

3. **Other Biobanks**
   - Estonian Biobank
   - Biobank Japan (ancestry-specific)
   - MVP (Veterans Affairs)

### Validation Metrics
- Nagelkerke R²
- AUC (C-statistic)
- Odds ratio per SD of PRS
- Calibration plots
""")

    lines.append("\n## Output Files")
    lines.append(f"\nPRS weights files in `{RESULTS_DIR}/`:")
    lines.append("- `{Phenotype}_PRS_p{threshold}.txt` - Single-phenotype weights")
    lines.append("- `CrossPhenotype_Female_PRS_p{threshold}.txt` - Combined female PRS")
    lines.append("- `BPH_PRS_p{threshold}.txt` - BPH-specific PRS")
    lines.append("- `validate_prs_prsice2.sh` - PRSice-2 validation script")
    lines.append("- `calculate_prs_plink.sh` - PLINK scoring script")

    lines.append("\n## Limitations")
    lines.append("""
1. **Simplified LD clumping**: Used window-based approach instead of proper LD reference
2. **No validation**: Individual-level data not available for validation
3. **European ancestry**: Weights derived from EUR populations only
4. **Winner's curse**: Effect sizes may be inflated in discovery sample
""")

    lines.append("\n## Next Steps")
    lines.append("""
1. Apply for UK Biobank access for validation
2. Consider LDpred2/PRS-CS for improved weights
3. Test transferability to other ancestries
4. Develop risk stratification thresholds for clinical use
""")

    return "\n".join(lines)


def main():
    print("="*60)
    print("Phase 8: PRS Development")
    print("="*60)

    all_results = []

    # Generate single-phenotype PRS
    print("\n[Step 1] Generating single-phenotype PRS weights...")

    for pheno in PHENOTYPES.keys():
        print(f"\n{pheno}:")
        df = load_sumstats(pheno)
        if df is not None:
            print(f"  Loaded {len(df)} SNPs")
            results = generate_prs_weights(pheno, df)
            all_results.extend(results)

    # Generate cross-phenotype PRS
    print("\n[Step 2] Generating cross-phenotype PRS...")
    cross_results = create_cross_phenotype_prs()

    # Create validation scripts
    print("\n[Step 3] Creating validation scripts...")
    create_validation_scripts()

    # Generate report
    print("\n[Step 4] Generating summary report...")
    report = generate_summary_report(all_results)

    report_path = BASE_DIR / "logs/11_prs_development.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("PRS DEVELOPMENT SUMMARY")
    print("="*60)

    # Count total files
    prs_files = list(RESULTS_DIR.glob("*_PRS_*.txt"))
    print(f"\nTotal PRS weight files generated: {len(prs_files)}")

    print("\n[Key PRS Files]")
    for f in sorted(prs_files)[:10]:
        df = pd.read_csv(f, sep='\t')
        print(f"  {f.name}: {len(df)} SNPs")

    print("\n" + "="*60)
    print("PRS Development Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
