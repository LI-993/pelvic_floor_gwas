#!/usr/bin/env python3
"""Run MTAG cross-phenotype meta-analysis.

Default mode runs MTAG with the standard settings used in the manuscript.
Pass `--fdr` to additionally compute the maxFDR diagnostic (skips the main
MTAG step and only runs the FDR routine, writing to a separate output prefix).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LDSC_DATA_DIR, MTAG_DIR, MTAG_LD_REF, PHENOTYPES, RESULTS_DIR


def build_command(fdr: bool, out_dir: Path) -> list[str]:
    sumstats = ",".join(str(LDSC_DATA_DIR / f"{p}.sumstats.gz") for p in PHENOTYPES)
    out_prefix = out_dir / ("pelvic_floor_fdr" if fdr else "pelvic_floor")

    cmd = [
        "python", str(MTAG_DIR / "mtag.py"),
        "--sumstats", sumstats,
        "--out", str(out_prefix),
        "--snp_name", "SNP",
        "--z_name", "Z",
        "--n_name", "N",
        "--a1_name", "A1",
        "--a2_name", "A2",
        "--p_name", "P",
        "--ld_ref_panel", str(MTAG_LD_REF),
        "--maf_min", "0",      # no MAF column in sumstats
        "--no_chr_data",       # no CHR/BP columns in sumstats
        "--stream_stdout",
    ]
    if fdr:
        # --fit_ss + --intervals=3 keeps the spike-slab grid tractable for 6 traits.
        cmd += ["--fdr", "--intervals", "3", "--skip_mtag"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr", action="store_true", help="Compute maxFDR diagnostic instead of the main MTAG run.")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "mtag"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args.fdr, out_dir)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 60)

    result = subprocess.run(cmd, cwd=MTAG_DIR)
    print(f"\nMTAG{' (FDR)' if args.fdr else ''} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
