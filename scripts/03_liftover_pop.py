#!/usr/bin/env python3
"""
LiftOver POP GWAS data from GRCh37 to GRCh38
Uses pyliftover package
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyliftover import LiftOver
import sys

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
PROCESSED_DIR = BASE_DIR / "data/processed"

def main():
    print("=" * 60)
    print("LiftOver POP from GRCh37 to GRCh38")
    print("=" * 60)

    # Initialize liftover (will download chain file automatically)
    print("\nInitializing LiftOver (downloading chain file if needed)...")
    lo = LiftOver('hg19', 'hg38')

    # Read POP data
    input_file = PROCESSED_DIR / "POP_GRCh37.tsv.gz"
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    print(f"Input rows: {len(df):,}")

    # LiftOver coordinates
    print("\nPerforming LiftOver...")
    new_positions = []
    failed_count = 0

    for idx, row in df.iterrows():
        if idx % 1000000 == 0:
            print(f"  Processed {idx:,} / {len(df):,} ({idx/len(df)*100:.1f}%)")

        chrom = f"chr{row['CHR']}"
        pos = int(row['POS'])

        # LiftOver returns list of possible mappings
        result = lo.convert_coordinate(chrom, pos)

        if result and len(result) > 0:
            # Take the first (best) result
            new_chrom, new_pos, strand, score = result[0]
            new_positions.append(new_pos)
        else:
            new_positions.append(np.nan)
            failed_count += 1

    print(f"\nLiftOver complete!")
    print(f"  Successful: {len(df) - failed_count:,}")
    print(f"  Failed: {failed_count:,} ({failed_count/len(df)*100:.2f}%)")

    # Update dataframe
    df['POS'] = new_positions

    # Remove failed mappings
    df_clean = df.dropna(subset=['POS'])
    df_clean['POS'] = df_clean['POS'].astype(int)
    print(f"  After removing failed: {len(df_clean):,}")

    # Save
    output_file = PROCESSED_DIR / "POP_GRCh38.tsv.gz"
    df_clean.to_csv(output_file, sep='\t', index=False, compression='gzip')
    print(f"\nSaved to: {output_file}")

    # Remove old GRCh37 file
    # input_file.unlink()  # Uncomment to delete

    print("\nDone!")

if __name__ == "__main__":
    main()
