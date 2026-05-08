#!/usr/bin/env python3
"""Run S-LDSC partitioned heritability with the baselineLD v2.2 model (97 annotations)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LDSC_SCRIPT,
    PHENOTYPES,
    RESULTS_DIR,
    SLDSC_ANNOT_DIR,
    SLDSC_BASELINE_PREFIX,
    SLDSC_FRQ_PREFIX,
    SLDSC_WEIGHTS_PREFIX,
    ldsc_sumstats_path,
)


def check_inputs() -> bool:
    """Verify the annotation files and per-phenotype sumstats exist."""
    missing: list[str] = []
    if not (SLDSC_ANNOT_DIR / f"{SLDSC_BASELINE_PREFIX.name}1.l2.ldscore.gz").exists():
        missing.append(f"Baseline LD: {SLDSC_BASELINE_PREFIX}*1.l2.ldscore.gz")
    if not (SLDSC_WEIGHTS_PREFIX.parent / f"{SLDSC_WEIGHTS_PREFIX.name}1.l2.ldscore.gz").exists():
        missing.append(f"Weights: {SLDSC_WEIGHTS_PREFIX}*1.l2.ldscore.gz")
    if not (SLDSC_FRQ_PREFIX.parent / f"{SLDSC_FRQ_PREFIX.name}1.frq").exists():
        missing.append(f"Frequencies: {SLDSC_FRQ_PREFIX}*1.frq")
    for p in PHENOTYPES:
        if not ldsc_sumstats_path(p).exists():
            missing.append(f"{p} sumstats: {ldsc_sumstats_path(p)}")
    if missing:
        print("\nMissing files:")
        for line in missing:
            print(f"  - {line}")
        return False
    return True


def run_one(pheno: str, out_dir: Path) -> bool:
    print(f"\n{'=' * 60}\nS-LDSC: {pheno}\n{'=' * 60}")
    out_prefix = out_dir / f"{pheno}_baselineLD"
    cmd = [
        "python", str(LDSC_SCRIPT),
        "--h2", str(ldsc_sumstats_path(pheno)),
        "--ref-ld-chr", str(SLDSC_BASELINE_PREFIX),
        "--w-ld-chr", str(SLDSC_WEIGHTS_PREFIX),
        "--overlap-annot",
        "--frqfile-chr", str(SLDSC_FRQ_PREFIX),
        "--out", str(out_prefix),
        "--print-coefficients",
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode == 0:
        print(f"-> {out_prefix}.results")
        return True
    print(f"FAILED (exit code {result.returncode})")
    return False


def main() -> None:
    out_dir = RESULTS_DIR / "sldsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("S-LDSC Partitioned Heritability (baselineLD v2.2, 97 annotations)")
    if not check_inputs():
        return

    results = {p: run_one(p, out_dir) for p in PHENOTYPES}
    print("\n" + "=" * 60)
    print("Summary")
    for p, ok in results.items():
        print(f"  {p}: {'OK' if ok else 'FAILED'}")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
