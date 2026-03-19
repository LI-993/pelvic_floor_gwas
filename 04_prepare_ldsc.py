#!/usr/bin/env python3
"""
Prepare LDSC sumstats files directly
Bypasses munge_sumstats.py which has Python 3 compatibility issues

Output format: SNP, A1, A2, Z, N, P
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
PROCESSED_DIR = BASE_DIR / "data/processed"
LDSC_DIR = BASE_DIR / "data/ldsc"
REF_DIR = Path("D:/Nproject/gwas/gwas_stroke_incontinence/reference/eur_w_ld_chr")

LDSC_DIR.mkdir(parents=True, exist_ok=True)

# Sample sizes
SAMPLE_SIZES = {
    'POP': 574377,
    'BPH': 501137,
    'Bladder': 503550,
    'Constipation': 501956,
    'FemaleProlapse': 503074,
    'Incontinence': 430019,
}

def load_hm3_snps():
    """Load HapMap3 SNP list for filtering."""
    snplist_file = REF_DIR / "w_hm3.snplist"
    print(f"Loading HapMap3 SNP list from {snplist_file}...")
    snps = pd.read_csv(snplist_file, sep='\t')
    print(f"  Loaded {len(snps):,} HapMap3 SNPs")
    return set(snps['SNP'].values)

def process_phenotype(name, input_file, output_file, n_samples, hm3_snps):
    """Convert preprocessed sumstats to LDSC format."""
    print(f"\nProcessing {name}...")
    print(f"  Input: {input_file}")

    # Read data
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    print(f"  Input rows: {len(df):,}")

    # Calculate Z score: Z = BETA / SE
    df['Z'] = df['BETA'] / df['SE']

    # Filter to HapMap3 SNPs
    df = df[df['SNP'].isin(hm3_snps)]
    print(f"  After HM3 filter: {len(df):,}")

    # Filter to bi-allelic SNPs only (A, T, C, G)
    valid_alleles = {'A', 'T', 'C', 'G'}
    df = df[df['A1'].isin(valid_alleles) & df['A2'].isin(valid_alleles)]
    print(f"  After bi-allelic filter: {len(df):,}")

    # Remove invalid Z scores
    df = df[np.isfinite(df['Z'])]
    print(f"  After Z QC: {len(df):,}")

    # Create output dataframe
    df_out = pd.DataFrame({
        'SNP': df['SNP'],
        'A1': df['A1'],
        'A2': df['A2'],
        'Z': df['Z'],
        'N': n_samples,
        'P': df['P'],
    })

    # Save
    df_out.to_csv(output_file, sep='\t', index=False, compression='gzip')
    print(f"  Output: {output_file}")
    print(f"  Final SNPs: {len(df_out):,}")

    return len(df_out)

def main():
    print("=" * 60)
    print("Prepare LDSC Summary Statistics")
    print("=" * 60)

    # Load HapMap3 SNPs
    hm3_snps = load_hm3_snps()

    # Process each phenotype
    results = {}

    phenotypes = [
        ('POP', 'POP_GRCh38.tsv.gz'),
        ('BPH', 'BPH_GRCh38.tsv.gz'),
        ('Bladder', 'Bladder_GRCh38.tsv.gz'),
        ('Constipation', 'Constipation_GRCh38.tsv.gz'),
        ('FemaleProlapse', 'FemaleProlapse_GRCh38.tsv.gz'),
        ('Incontinence', 'Incontinence_GRCh38.tsv.gz'),
    ]

    for name, filename in phenotypes:
        input_file = PROCESSED_DIR / filename
        output_file = LDSC_DIR / f"{name}.sumstats.gz"
        n_samples = SAMPLE_SIZES[name]

        n_snps = process_phenotype(name, input_file, output_file, n_samples, hm3_snps)
        results[name] = n_snps

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, n_snps in results.items():
        print(f"  {name}: {n_snps:,} SNPs")

    print(f"\nOutput directory: {LDSC_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
