#!/usr/bin/env python3
"""Convert standardized sumstats into LDSC `.sumstats.gz` format.

This is a Python re-implementation of `munge_sumstats.py` that bypasses the
Python-2 quirks of the upstream script: filters to HapMap3 SNPs, drops
non-biallelic alleles, and computes Z = BETA / SE. Output schema is
SNP A1 A2 Z N P, gzip-compressed, written to `data/ldsc/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HM3_SNPLIST, LDSC_DATA_DIR, PHENOTYPES, SAMPLE_SIZES
from utils.io import load_hm3_snps, load_sumstats, prepare_ldsc_record, write_tsv_gz


def main() -> None:
    print("=" * 60)
    print("Prepare LDSC sumstats (HM3 filter, Z = BETA/SE)")
    print("=" * 60)

    LDSC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading HapMap3 SNP list: {HM3_SNPLIST}")
    hm3 = load_hm3_snps(HM3_SNPLIST)
    print(f"  {len(hm3):,} HapMap3 SNPs loaded")

    counts: dict[str, int] = {}
    for pheno in PHENOTYPES:
        print(f"\nProcessing {pheno}...")
        df = load_sumstats(pheno)
        n_in = len(df)
        out = prepare_ldsc_record(df, n_samples=SAMPLE_SIZES[pheno], hm3_snps=hm3)
        out_path = LDSC_DATA_DIR / f"{pheno}.sumstats.gz"
        write_tsv_gz(out, out_path)
        counts[pheno] = len(out)
        print(f"  {n_in:,} -> {len(out):,} SNPs -> {out_path}")

    print("\n" + "=" * 60)
    print("Summary")
    for name, n in counts.items():
        print(f"  {name}: {n:,} SNPs")
    print(f"Output directory: {LDSC_DATA_DIR}")


if __name__ == "__main__":
    main()
