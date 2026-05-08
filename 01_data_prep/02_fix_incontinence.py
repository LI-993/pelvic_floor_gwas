#!/usr/bin/env python3
"""Re-derive Incontinence SE from the 95% CI of the odds ratio.

The GWAS Catalog file has rows where `standard_error` is missing but `ci_upper`
and `ci_lower` are present. For log(OR), SE = (log(CI_upper) - log(CI_lower)) / (2 * 1.96).
This script regenerates `Incontinence_GRCh38.tsv.gz` using that fallback so
all rows have a usable SE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RAW_DIR
from utils.io import write_tsv_gz


def main() -> None:
    print("Re-deriving SE from 95% CI for Incontinence...")
    df = pd.read_csv(
        RAW_DIR / "Incontinence" / "Incontinence.h.tsv.gz",
        sep="\t",
        compression="gzip",
        low_memory=False,
    )
    print(f"  Rows: {len(df):,}")

    df["beta"] = np.log(df["odds_ratio"])
    df["se_calc"] = (np.log(df["ci_upper"]) - np.log(df["ci_lower"])) / (2 * 1.96)

    out = pd.DataFrame({
        "CHR": df["chromosome"],
        "POS": df["base_pair_location"],
        "SNP": df["rsid"],
        "A1": df["effect_allele"],
        "A2": df["other_allele"],
        "BETA": df["beta"],
        "SE": df["se_calc"],
        "P": df["p_value"],
        "EAF": df["effect_allele_frequency"],
        "N": df["n"],
    })

    n0 = len(out)
    out = out.dropna(subset=["CHR", "POS", "SNP", "A1", "A2", "BETA", "SE", "P"])
    out = out[(out["P"] > 0) & (out["SE"] > 0)]
    out = out[np.isfinite(out["BETA"]) & np.isfinite(out["SE"])]
    print(f"  After QC: {len(out):,} ({len(out) / n0 * 100:.1f}%)")

    output_file = PROCESSED_DIR / "Incontinence_GRCh38.tsv.gz"
    write_tsv_gz(out, output_file)
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
