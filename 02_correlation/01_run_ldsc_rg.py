#!/usr/bin/env python3
"""Run LDSC pairwise genetic correlation across all phenotype pairs.

Wraps the upstream LDSC `--rg` runner. Output goes to `results/ldsc/` as one
log per pair plus the standard summary table once 02_parse_ldsc finishes.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LDSC_DATA_DIR,
    LDSC_PYTHON,
    LDSC_REF_DIR,
    LDSC_SCRIPT,
    PHENOTYPES,
    RESULTS_DIR,
)


def main() -> None:
    out_dir = RESULTS_DIR / "ldsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(combinations(PHENOTYPES, 2))
    print(f"Running LDSC rg for {len(pairs)} pairs")
    print("=" * 60)

    ref_prefix = f"{LDSC_REF_DIR}/eur_w_ld_chr/@"

    for i, (p1, p2) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {p1} vs {p2}")
        out_prefix = out_dir / f"{p1}_vs_{p2}"

        cmd = [
            str(LDSC_PYTHON), str(LDSC_SCRIPT),
            "--rg", f"{LDSC_DATA_DIR / f'{p1}.sumstats.gz'},{LDSC_DATA_DIR / f'{p2}.sumstats.gz'}",
            "--ref-ld-chr", ref_prefix,
            "--w-ld-chr", ref_prefix,
            "--out", str(out_prefix),
            "--no-check-alleles",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            continue

        log = (out_prefix.with_suffix(".log")).read_text()
        for line in log.splitlines():
            if "Genetic Correlation:" in line or (line.startswith("P:") and "P-value" not in line):
                print(f"  {line.strip()}")

    print("\n" + "=" * 60)
    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
