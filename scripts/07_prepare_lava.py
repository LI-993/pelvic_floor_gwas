#!/usr/bin/env python3
"""
Prepare input files for LAVA local genetic correlation analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gzip

# Directories
DATA_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\data\processed")
LDSC_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\data\ldsc")
OUT_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\data\lava")
RESULTS_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\results\ldsc")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phenotype info (from LDSC analysis log)
PHENOTYPES = {
    'POP': {'cases': 28086, 'controls': 546291},
    'BPH': {'cases': 41137, 'controls': 460000},
    'Bladder': {'cases': 3550, 'controls': 500000},
    'Constipation': {'cases': 51956, 'controls': 450000},
    'FemaleProlapse': {'cases': 23074, 'controls': 480000},
    'Incontinence': {'cases': 27714, 'controls': 402305}
}

def prepare_sumstats(phenotype):
    """Convert processed GWAS to LAVA format."""
    print(f"\nProcessing {phenotype}...")

    # Read processed data
    input_file = DATA_DIR / f"{phenotype}_GRCh38.tsv.gz"
    if phenotype == 'POP':
        input_file = DATA_DIR / f"{phenotype}_GRCh37.tsv.gz"

    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    print(f"  Loaded {len(df):,} SNPs")

    # Calculate N and Z
    info = PHENOTYPES[phenotype]
    N = info['cases'] + info['controls']

    # Z = BETA / SE
    df['Z'] = df['BETA'] / df['SE']
    df['N'] = N

    # Remove invalid values
    df = df[~df['Z'].isna() & ~np.isinf(df['Z'])]
    print(f"  After removing invalid Z: {len(df):,} SNPs")

    # LAVA format: SNP A1 A2 N Z
    out_df = df[['SNP', 'A1', 'A2', 'N', 'Z']].copy()

    # Save
    out_file = OUT_DIR / f"{phenotype}.sumstats.txt"
    out_df.to_csv(out_file, sep='\t', index=False)
    print(f"  Saved to: {out_file}")

    return len(out_df)

def create_input_info():
    """Create input.info.txt file for LAVA."""
    print("\nCreating input.info.txt...")

    rows = []
    for pheno, info in PHENOTYPES.items():
        rows.append({
            'phenotype': pheno,
            'cases': info['cases'],
            'controls': info['controls'],
            'filename': f"data/lava/{pheno}.sumstats.txt"
        })

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / "input.info.txt"
    df.to_csv(out_file, sep='\t', index=False)
    print(f"Saved to: {out_file}")

def create_sample_overlap():
    """Create sample.overlap.txt from LDSC genetic covariance intercept.

    The sample overlap is computed from the gcov_int from LDSC,
    which represents the phenotypic correlation due to sample overlap.
    """
    print("\nCreating sample.overlap.txt...")

    # Parse LDSC log files for gcov_int
    phenotypes = list(PHENOTYPES.keys())
    n_pheno = len(phenotypes)

    # Initialize matrix with 1s on diagonal
    overlap = pd.DataFrame(
        np.eye(n_pheno),
        index=phenotypes,
        columns=phenotypes
    )

    # Read gcov_int from LDSC logs
    # Format: Third "Intercept:" value is gcov_int
    import re
    for log_file in RESULTS_DIR.glob("*_vs_*.log"):
        name = log_file.stem
        parts = name.split("_vs_")
        p1, p2 = parts[0], parts[1]

        with open(log_file) as f:
            content = f.read()

        # Find all intercepts - the third one is gcov_int
        matches = re.findall(r"Intercept:\s+([\d.-]+)", content)
        if len(matches) >= 3:
            gcov_int = float(matches[2])
            overlap.loc[p1, p2] = gcov_int
            overlap.loc[p2, p1] = gcov_int
            print(f"  {p1} vs {p2}: gcov_int = {gcov_int}")

    # Save
    out_file = OUT_DIR / "sample.overlap.txt"
    overlap.to_csv(out_file, sep=' ')
    print(f"Saved to: {out_file}")
    print("\nSample overlap matrix:")
    print(overlap.round(4))

def main():
    print("=" * 60)
    print("LAVA Input Preparation")
    print("=" * 60)

    # Prepare summary statistics
    for pheno in PHENOTYPES:
        prepare_sumstats(pheno)

    # Create input info file
    create_input_info()

    # Create sample overlap file
    create_sample_overlap()

    print("\n" + "=" * 60)
    print("LAVA input preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
