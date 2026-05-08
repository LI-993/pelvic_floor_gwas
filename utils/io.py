"""Sumstats I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import sumstats_path

VALID_ALLELES = frozenset({"A", "T", "C", "G"})


def load_sumstats(phenotype: str, build: str = "GRCh38") -> pd.DataFrame:
    """Load standardized sumstats for a phenotype.

    The file produced by 01_preprocess_sumstats has columns
    CHR POS SNP A1 A2 BETA SE P EAF (and N for some phenotypes).
    """
    return pd.read_csv(sumstats_path(phenotype, build), sep="\t", compression="gzip")


def write_tsv_gz(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as gzipped tab-separated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def prepare_ldsc_record(
    df: pd.DataFrame,
    n_samples: int,
    hm3_snps: set[str] | None = None,
) -> pd.DataFrame:
    """Convert standardized sumstats to LDSC sumstats format (SNP A1 A2 Z N P).

    Drops rows with missing/invalid Z, non-biallelic alleles, and (optionally)
    SNPs not in the HapMap3 list used by the LD-score regression.
    """
    out = df.copy()
    out["Z"] = out["BETA"] / out["SE"]

    if hm3_snps is not None:
        out = out[out["SNP"].isin(hm3_snps)]

    out = out[out["A1"].isin(VALID_ALLELES) & out["A2"].isin(VALID_ALLELES)]
    out = out[np.isfinite(out["Z"])]

    return pd.DataFrame({
        "SNP": out["SNP"],
        "A1": out["A1"],
        "A2": out["A2"],
        "Z": out["Z"],
        "N": n_samples,
        "P": out["P"],
    })


def load_hm3_snps(snplist_path: Path) -> set[str]:
    """Load HapMap3 SNP list used by LDSC."""
    snps = pd.read_csv(snplist_path, sep="\t")
    return set(snps["SNP"].values)
