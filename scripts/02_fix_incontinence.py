#!/usr/bin/env python3
"""
Fix Incontinence data - calculate SE from confidence intervals
SE for log(OR) = (log(CI_upper) - log(CI_lower)) / (2 * 1.96)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

print("Processing Incontinence data...")
print("Reading file...")

# Read data
df = pd.read_csv(
    RAW_DIR / "Incontinence" / "Incontinence.h.tsv.gz",
    sep='\t',
    compression='gzip',
    low_memory=False
)

print(f"Rows: {len(df):,}")

# Check columns
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample of OR, CI_upper, CI_lower:")
print(df[['odds_ratio', 'ci_upper', 'ci_lower']].head())

# Calculate beta = log(OR)
df['beta'] = np.log(df['odds_ratio'])

# Calculate SE from confidence intervals
# SE = (log(CI_upper) - log(CI_lower)) / (2 * 1.96)
df['se_calc'] = (np.log(df['ci_upper']) - np.log(df['ci_lower'])) / (2 * 1.96)

print(f"\nCalculated beta and SE:")
print(df[['odds_ratio', 'beta', 'ci_upper', 'ci_lower', 'se_calc']].head())

# Create standardized output
df_out = pd.DataFrame({
    'CHR': df['chromosome'],
    'POS': df['base_pair_location'],
    'SNP': df['rsid'],
    'A1': df['effect_allele'],
    'A2': df['other_allele'],
    'BETA': df['beta'],
    'SE': df['se_calc'],
    'P': df['p_value'],
    'EAF': df['effect_allele_frequency'],
    'N': df['n'],
})

# QC
print(f"\nBefore QC: {len(df_out):,}")
initial_count = len(df_out)

# Remove rows with missing values
df_out = df_out.dropna(subset=['CHR', 'POS', 'SNP', 'A1', 'A2', 'BETA', 'SE', 'P'])

# Remove invalid values
df_out = df_out[df_out['P'] > 0]
df_out = df_out[df_out['SE'] > 0]
df_out = df_out[np.isfinite(df_out['BETA'])]
df_out = df_out[np.isfinite(df_out['SE'])]

print(f"After QC: {len(df_out):,} ({len(df_out)/initial_count*100:.1f}%)")

# Save
output_file = PROCESSED_DIR / "Incontinence_GRCh38.tsv.gz"
df_out.to_csv(output_file, sep='\t', index=False, compression='gzip')
print(f"\nSaved to: {output_file}")

# Verify
print("\nVerification - first few rows:")
print(df_out.head())
