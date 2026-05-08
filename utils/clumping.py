"""Distance-based LD clumping.

For proper LD-aware clumping use PLINK with a reference panel; this lightweight
function is good enough for PRS prototyping where no individual-level reference
is available. SNPs are sorted by p-value, then a SNP is kept iff no already-kept
SNP on the same chromosome lies within `window` bp.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WINDOW = 500_000


def simple_clump(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    p_col: str = "P",
    chr_col: str = "CHR",
    pos_col: str = "POS",
) -> pd.DataFrame:
    """Window-based clumping: keep the most-significant SNP per `window` bp."""
    df = df.sort_values(p_col)
    kept_rows = []
    used: dict[int, list[int]] = {}

    for row in df.itertuples(index=False):
        chrom = getattr(row, chr_col)
        pos = getattr(row, pos_col)
        positions = used.setdefault(chrom, [])
        if all(abs(pos - p) >= window for p in positions):
            kept_rows.append(row)
            positions.append(pos)

    return pd.DataFrame(kept_rows, columns=df.columns)
