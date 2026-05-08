#!/usr/bin/env python3
"""Standardize raw GWAS sumstats to a common schema.

Reads FinnGen R12 and GWAS-Catalog source files and writes
`{phenotype}_{build}.tsv.gz` with columns:
CHR POS SNP A1 A2 BETA SE P EAF (and N when available).

POP arrives in GRCh37 from GWAS Catalog and needs a separate liftover step
(03_liftover_pop). Incontinence is GWAS Catalog with OR + 95% CI; this script
handles the OR -> log(OR) conversion using the standard error column. The CI
based recovery for Incontinence rows missing SE is handled in
02_fix_incontinence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RAW_DIR
from utils.io import write_tsv_gz


REQUIRED_COLS = ["CHR", "POS", "SNP", "A1", "A2", "BETA", "SE", "P"]


def _qc(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df["P"] > 0) & (df["SE"] > 0)]
    df = df[np.isfinite(df["BETA"])]
    print(f"  After QC: {len(df):,} ({len(df) / n0 * 100:.1f}%) [{label}]")
    return df


def process_finngen(input_file: Path, output_file: Path, label: str) -> pd.DataFrame:
    """FinnGen columns: #chrom pos ref alt rsids ... pval beta sebeta af_alt.

    In FinnGen, `alt` is the effect allele and `ref` the other allele.
    """
    print(f"\nProcessing FinnGen: {label}")
    print(f"  Input: {input_file}")
    df = pd.read_csv(input_file, sep="\t", compression="gzip")
    print(f"  Rows: {len(df):,}")

    out = pd.DataFrame({
        "CHR": df["#chrom"],
        "POS": df["pos"],
        "SNP": df["rsids"],
        "A1": df["alt"],
        "A2": df["ref"],
        "BETA": df["beta"],
        "SE": df["sebeta"],
        "P": df["pval"],
        "EAF": df["af_alt"],
    })
    out = _qc(out, label)
    write_tsv_gz(out, output_file)
    print(f"  Output: {output_file}")
    return out


def process_gwas_catalog(
    input_file: Path,
    output_file: Path,
    label: str,
    *,
    effect: str = "beta",
) -> pd.DataFrame:
    """GWAS Catalog harmonized format. Set `effect="or"` for odds-ratio sources.

    Beta sources expose `beta` directly; OR sources expose `odds_ratio` and
    require log conversion (SE refers to log(OR) under the standard reporting
    convention used by GWAS Catalog).
    """
    print(f"\nProcessing GWAS Catalog ({effect}): {label}")
    print(f"  Input: {input_file}")
    df = pd.read_csv(input_file, sep="\t", compression="gzip")
    print(f"  Rows: {len(df):,}")

    snp_col = "rsid" if "rsid" in df.columns else "variant_id"

    if effect == "or":
        beta = np.log(df["odds_ratio"])
    elif effect == "beta":
        beta = df["beta"]
    else:
        raise ValueError(f"Unknown effect type: {effect}")

    out = pd.DataFrame({
        "CHR": df["chromosome"],
        "POS": df["base_pair_location"],
        "SNP": df[snp_col],
        "A1": df["effect_allele"],
        "A2": df["other_allele"],
        "BETA": beta,
        "SE": df["standard_error"],
        "P": df["p_value"],
        "EAF": df["effect_allele_frequency"],
    })
    if "n" in df.columns:
        out["N"] = df["n"]

    out = _qc(out, label)
    write_tsv_gz(out, output_file)
    print(f"  Output: {output_file}")
    return out


def main() -> None:
    print("=" * 60)
    print("GWAS Summary Statistics Preprocessing")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    finngen = {
        "BPH": "finngen_R12_N14_PROSTHYPERPLA.gz",
        "Bladder": "finngen_R12_N14_NEUROMUSCDYSBLADD.gz",
        "Constipation": "finngen_R12_K11_CONSTIPATION.gz",
        "FemaleProlapse": "finngen_R12_N14_FEMGENPROL.gz",
    }
    for name, filename in finngen.items():
        process_finngen(RAW_DIR / "FinnGen" / filename, PROCESSED_DIR / f"{name}_GRCh38.tsv.gz", name)

    process_gwas_catalog(
        RAW_DIR / "POP" / "GCST90102470_buildGRCh37.tsv.gz",
        PROCESSED_DIR / "POP_GRCh37.tsv.gz",
        "POP",
        effect="beta",
    )
    print("  POP is GRCh37; run 03_liftover_pop next.")

    process_gwas_catalog(
        RAW_DIR / "Incontinence" / "Incontinence.h.tsv.gz",
        PROCESSED_DIR / "Incontinence_GRCh38.tsv.gz",
        "Incontinence",
        effect="or",
    )

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
