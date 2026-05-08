#!/usr/bin/env python3
"""LiftOver POP sumstats from GRCh37 to GRCh38 using pyliftover.

POP is the only phenotype shipped in build 37; all others are already 38. The
chain file is downloaded automatically by pyliftover on first use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyliftover import LiftOver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR
from utils.io import write_tsv_gz


def main() -> None:
    print("LiftOver: POP GRCh37 -> GRCh38")
    lo = LiftOver("hg19", "hg38")

    input_file = PROCESSED_DIR / "POP_GRCh37.tsv.gz"
    df = pd.read_csv(input_file, sep="\t", compression="gzip")
    print(f"  Input rows: {len(df):,}")

    new_positions: list[float] = []
    failed = 0
    for idx, row in df.iterrows():
        if idx % 1_000_000 == 0:
            print(f"  Processed {idx:,} / {len(df):,} ({idx / len(df) * 100:.1f}%)")
        result = lo.convert_coordinate(f"chr{row['CHR']}", int(row["POS"]))
        if result:
            new_positions.append(result[0][1])
        else:
            new_positions.append(np.nan)
            failed += 1

    print(f"  Successful: {len(df) - failed:,}")
    print(f"  Failed: {failed:,} ({failed / len(df) * 100:.2f}%)")

    df["POS"] = new_positions
    df = df.dropna(subset=["POS"])
    df["POS"] = df["POS"].astype(int)

    output_file = PROCESSED_DIR / "POP_GRCh38.tsv.gz"
    write_tsv_gz(df, output_file)
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
