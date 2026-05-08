"""Plotting helpers shared across visualization scripts."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from config import CHR_LENGTHS_GRCh38


def setup_publication_style() -> None:
    """Apply consistent matplotlib defaults for figures in this project."""
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10


def chr_offsets(
    df: pd.DataFrame | None = None,
    *,
    pos_col: str = "stop",
    gap: int = 10_000_000,
) -> dict[int, int]:
    """Cumulative chromosome offsets for a Manhattan-style plot.

    If `df` is given, offsets are computed from the data; otherwise GRCh38
    chromosome lengths from config are used.
    """
    offsets: dict[int, int] = {}
    cumulative = 0
    if df is None:
        for chrom in range(1, 23):
            offsets[chrom] = cumulative
            cumulative += CHR_LENGTHS_GRCh38[chrom] + gap
    else:
        chr_max = df.groupby("chr")[pos_col].max()
        for chrom in range(1, 23):
            offsets[chrom] = cumulative
            if chrom in chr_max.index:
                cumulative += int(chr_max[chrom]) + gap
    return offsets


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected significance threshold."""
    return alpha / max(n_tests, 1)


def significance_marker(p: float) -> str:
    """Star notation: *** <0.001, ** <0.01, * <0.05, blank otherwise."""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""
