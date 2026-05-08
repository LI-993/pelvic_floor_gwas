#!/usr/bin/env python3
"""
GWAS Summary Statistics Preprocessing
=====================================
Standardize formats for all phenotypes and prepare for LDSC analysis.

Output format (standardized):
- CHR: chromosome
- POS: position (GRCh38)
- SNP: rsID
- A1: effect allele
- A2: other allele
- BETA: effect size (log(OR) for binary traits)
- SE: standard error
- P: p-value
- EAF: effect allele frequency
- N: sample size (if available)
"""

import pandas as pd
import numpy as np
import gzip
import os
from pathlib import Path

# Paths
BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_finngen(input_file, output_file, phenotype_name):
    """Process FinnGen format summary statistics."""
    print(f"\nProcessing FinnGen: {phenotype_name}")
    print(f"  Input: {input_file}")

    # Read data
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    print(f"  Rows: {len(df):,}")

    # FinnGen columns: #chrom, pos, ref, alt, rsids, nearest_genes, pval, mlogp, beta, sebeta, af_alt, af_alt_cases, af_alt_controls
    # Note: In FinnGen, alt is the effect allele, ref is the other allele

    # Standardize
    df_out = pd.DataFrame({
        'CHR': df['#chrom'],
        'POS': df['pos'],
        'SNP': df['rsids'],
        'A1': df['alt'],      # effect allele (alt in FinnGen)
        'A2': df['ref'],      # other allele (ref in FinnGen)
        'BETA': df['beta'],
        'SE': df['sebeta'],
        'P': df['pval'],
        'EAF': df['af_alt'],
    })

    # QC: Remove missing values
    initial_count = len(df_out)
    df_out = df_out.dropna(subset=['CHR', 'POS', 'SNP', 'A1', 'A2', 'BETA', 'SE', 'P'])
    df_out = df_out[df_out['P'] > 0]  # Remove P=0
    df_out = df_out[df_out['SE'] > 0]  # Remove SE=0
    print(f"  After QC: {len(df_out):,} ({len(df_out)/initial_count*100:.1f}%)")

    # Save
    df_out.to_csv(output_file, sep='\t', index=False, compression='gzip')
    print(f"  Output: {output_file}")

    return df_out

def process_gwas_catalog_beta(input_file, output_file, phenotype_name):
    """Process GWAS Catalog format with beta (POP)."""
    print(f"\nProcessing GWAS Catalog (beta): {phenotype_name}")
    print(f"  Input: {input_file}")
    print(f"  NOTE: This is GRCh37, will need LiftOver!")

    # Read data
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    print(f"  Rows: {len(df):,}")

    # Columns: chromosome, base_pair_location, variant_id, effect_allele, other_allele,
    #          effect_allele_frequency, beta, standard_error, p_value

    # Standardize
    df_out = pd.DataFrame({
        'CHR': df['chromosome'],
        'POS': df['base_pair_location'],
        'SNP': df['variant_id'],
        'A1': df['effect_allele'],
        'A2': df['other_allele'],
        'BETA': df['beta'],
        'SE': df['standard_error'],
        'P': df['p_value'],
        'EAF': df['effect_allele_frequency'],
    })

    # QC
    initial_count = len(df_out)
    df_out = df_out.dropna(subset=['CHR', 'POS', 'SNP', 'A1', 'A2', 'BETA', 'SE', 'P'])
    df_out = df_out[df_out['P'] > 0]
    df_out = df_out[df_out['SE'] > 0]
    print(f"  After QC: {len(df_out):,} ({len(df_out)/initial_count*100:.1f}%)")

    # Save
    df_out.to_csv(output_file, sep='\t', index=False, compression='gzip')
    print(f"  Output: {output_file}")

    return df_out

def process_gwas_catalog_or(input_file, output_file, phenotype_name):
    """Process GWAS Catalog format with odds ratio (Incontinence)."""
    print(f"\nProcessing GWAS Catalog (OR): {phenotype_name}")
    print(f"  Input: {input_file}")

    # Read data
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    print(f"  Rows: {len(df):,}")

    # Columns include: chromosome, base_pair_location, effect_allele, other_allele,
    #                  odds_ratio, standard_error, effect_allele_frequency, p_value, rsid, n, num_cases, num_controls

    # Convert OR to beta: beta = log(OR)
    # SE for log(OR) is approximately SE_OR / OR
    df['beta_calc'] = np.log(df['odds_ratio'])

    # Standardize
    df_out = pd.DataFrame({
        'CHR': df['chromosome'],
        'POS': df['base_pair_location'],
        'SNP': df['rsid'],
        'A1': df['effect_allele'],
        'A2': df['other_allele'],
        'BETA': df['beta_calc'],
        'SE': df['standard_error'],
        'P': df['p_value'],
        'EAF': df['effect_allele_frequency'],
        'N': df['n'] if 'n' in df.columns else None,
    })

    # QC
    initial_count = len(df_out)
    df_out = df_out.dropna(subset=['CHR', 'POS', 'SNP', 'A1', 'A2', 'BETA', 'SE', 'P'])
    df_out = df_out[df_out['P'] > 0]
    df_out = df_out[df_out['SE'] > 0]
    df_out = df_out[np.isfinite(df_out['BETA'])]  # Remove infinite beta (from OR=0 or OR=inf)
    print(f"  After QC: {len(df_out):,} ({len(df_out)/initial_count*100:.1f}%)")

    # Save
    df_out.to_csv(output_file, sep='\t', index=False, compression='gzip')
    print(f"  Output: {output_file}")

    return df_out

def main():
    print("=" * 60)
    print("GWAS Summary Statistics Preprocessing")
    print("=" * 60)

    # 1. FinnGen datasets (GRCh38)
    finngen_files = {
        'BPH': 'finngen_R12_N14_PROSTHYPERPLA.gz',
        'Bladder': 'finngen_R12_N14_NEUROMUSCDYSBLADD.gz',
        'Constipation': 'finngen_R12_K11_CONSTIPATION.gz',
        'FemaleProlapse': 'finngen_R12_N14_FEMGENPROL.gz',
    }

    for name, filename in finngen_files.items():
        input_path = RAW_DIR / "FinnGen" / filename
        output_path = PROCESSED_DIR / f"{name}_GRCh38.tsv.gz"
        process_finngen(input_path, output_path, name)

    # 2. POP - GWAS Catalog with beta (GRCh37 - needs liftover later)
    process_gwas_catalog_beta(
        RAW_DIR / "POP" / "GCST90102470_buildGRCh37.tsv.gz",
        PROCESSED_DIR / "POP_GRCh37.tsv.gz",
        "POP"
    )

    # 3. Incontinence - GWAS Catalog with OR (GRCh38)
    process_gwas_catalog_or(
        RAW_DIR / "Incontinence" / "Incontinence.h.tsv.gz",
        PROCESSED_DIR / "Incontinence_GRCh38.tsv.gz",
        "Incontinence"
    )

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("\nOutput files in:", PROCESSED_DIR)
    print("\nNOTE: POP is still in GRCh37, needs LiftOver to GRCh38")

if __name__ == "__main__":
    main()
