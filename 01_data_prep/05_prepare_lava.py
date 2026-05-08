#!/usr/bin/env python3
"""Prepare LAVA inputs: per-phenotype sumstats, input.info, sample.overlap.

LAVA expects sumstats in `SNP A1 A2 N Z` form, an info table listing each
phenotype's case/control counts and file path, and a sample-overlap matrix.
The overlap matrix is built from LDSC `gcov_int` (third Intercept entry per rg
log) so the local correlations correct for the same shared-sample bias the
genome-wide rg estimates already account for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CASES_CONTROLS,
    LAVA_DATA_DIR,
    LDSC_DATA_DIR,
    PHENOTYPES,
    PROCESSED_DIR,
    RESULTS_DIR,
    sumstats_path,
)
from utils.ldsc import parse_rg_log


def prepare_sumstats(pheno: str) -> int:
    """Write the per-phenotype LAVA sumstats file."""
    print(f"\nProcessing {pheno}...")

    # POP is in build 37 in raw; LAVA loci file is GRCh37/hg19 so we keep that.
    src = sumstats_path(pheno, "GRCh37" if pheno == "POP" else "GRCh38")
    df = pd.read_csv(src, sep="\t", compression="gzip")
    print(f"  Loaded {len(df):,} SNPs from {src.name}")

    cases, controls = CASES_CONTROLS[pheno]
    df["Z"] = df["BETA"] / df["SE"]
    df["N"] = cases + controls
    df = df[~df["Z"].isna() & ~np.isinf(df["Z"])]
    print(f"  After Z QC: {len(df):,}")

    out = df[["SNP", "A1", "A2", "N", "Z"]]
    out_path = LAVA_DATA_DIR / f"{pheno}.sumstats.txt"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path}")
    return len(out)


def write_input_info() -> None:
    """LAVA input.info: phenotype, cases, controls, filename."""
    rows = [
        {
            "phenotype": p,
            "cases": CASES_CONTROLS[p][0],
            "controls": CASES_CONTROLS[p][1],
            "filename": f"data/lava/{p}.sumstats.txt",
        }
        for p in PHENOTYPES
    ]
    out_path = LAVA_DATA_DIR / "input.info.txt"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"\ninput.info: {out_path}")


def write_sample_overlap() -> None:
    """Build the sample-overlap matrix from LDSC gcov_int values."""
    print("\nBuilding sample overlap matrix from LDSC rg logs...")
    overlap = pd.DataFrame(np.eye(len(PHENOTYPES)), index=PHENOTYPES, columns=PHENOTYPES)

    for log_file in (RESULTS_DIR / "ldsc").glob("*_vs_*.log"):
        p1, p2 = log_file.stem.split("_vs_")
        parsed = parse_rg_log(log_file)
        if "gcov_int" in parsed:
            overlap.loc[p1, p2] = parsed["gcov_int"]
            overlap.loc[p2, p1] = parsed["gcov_int"]
            print(f"  {p1} vs {p2}: gcov_int = {parsed['gcov_int']:.4f}")

    out_path = LAVA_DATA_DIR / "sample.overlap.txt"
    overlap.to_csv(out_path, sep=" ")
    print(f"\nsample.overlap: {out_path}")
    print(overlap.round(4))


def main() -> None:
    print("=" * 60)
    print("LAVA Input Preparation")
    print("=" * 60)
    LAVA_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pheno in PHENOTYPES:
        prepare_sumstats(pheno)

    write_input_info()
    write_sample_overlap()
    print(f"\nOutputs in: {LAVA_DATA_DIR}")


if __name__ == "__main__":
    main()
