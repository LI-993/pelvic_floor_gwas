#!/usr/bin/env python3
"""
Run S-LDSC partitioned heritability analysis for pelvic floor phenotypes.

Uses Python 3 version of LDSC with baselineLD v2.2 model.
"""

import os
import subprocess
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
LDSC_DIR = Path("D:/Nproject/gwas/ldsc-python3")

# Annotation paths
ANNOT_DIR = BASE_DIR / "reference/ldsc_annotations"
BASELINE_PREFIX = ANNOT_DIR / "baselineLD."
WEIGHTS_PREFIX = ANNOT_DIR / "1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC."
FRQ_PREFIX = ANNOT_DIR / "1000G_Phase3_frq/1000G.EUR.QC."

# Input/Output
SUMSTATS_DIR = BASE_DIR / "data/ldsc"  # Munged sumstats from Phase 2
RESULTS_DIR = BASE_DIR / "results/sldsc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Phenotypes
PHENOTYPES = {
    "POP": "POP.sumstats.gz",
    "BPH": "BPH.sumstats.gz",
    "Bladder": "Bladder.sumstats.gz",
    "Constipation": "Constipation.sumstats.gz",
    "FemaleProlapse": "FemaleProlapse.sumstats.gz",
    "Incontinence": "Incontinence.sumstats.gz",
}


def check_files():
    """Verify all required files exist."""
    print("Checking required files...")
    missing = []

    # Check baseline LD scores (chromosome 1 as test)
    test_file = Path(str(BASELINE_PREFIX) + "1.l2.ldscore.gz")
    if not test_file.exists():
        missing.append(f"Baseline LD: {test_file}")

    # Check weights
    test_file = Path(str(WEIGHTS_PREFIX) + "1.l2.ldscore.gz")
    if not test_file.exists():
        missing.append(f"Weights: {test_file}")

    # Check frequency files
    test_file = Path(str(FRQ_PREFIX) + "1.frq")
    if not test_file.exists():
        missing.append(f"Frequency: {test_file}")

    # Check sumstats
    for pheno, fname in PHENOTYPES.items():
        fpath = SUMSTATS_DIR / fname
        if not fpath.exists():
            missing.append(f"{pheno} sumstats: {fpath}")

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"  - {f}")
        return False

    print("All required files found!")
    return True


def run_sldsc_h2(pheno_name, sumstats_file):
    """
    Run S-LDSC partitioned heritability with baselineLD v2.2 model.

    This estimates heritability enrichment across 97 functional annotations.
    """
    print(f"\n{'='*60}")
    print(f"S-LDSC Partitioned Heritability: {pheno_name}")
    print(f"{'='*60}")

    sumstats_path = SUMSTATS_DIR / sumstats_file
    output_prefix = RESULTS_DIR / f"{pheno_name}_baselineLD"

    # Build command
    cmd = [
        "python", str(LDSC_DIR / "ldsc.py"),
        "--h2", str(sumstats_path),
        "--ref-ld-chr", str(BASELINE_PREFIX),
        "--w-ld-chr", str(WEIGHTS_PREFIX),
        "--overlap-annot",
        "--frqfile-chr", str(FRQ_PREFIX),
        "--out", str(output_prefix),
        "--print-coefficients"
    ]

    print(f"\nCommand:\n{' '.join(cmd)}\n")

    # Run
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        print(f"\nSuccess! Results: {output_prefix}.results")
        return True
    else:
        print(f"\nFailed with return code: {result.returncode}")
        return False


def main():
    print("="*60)
    print("S-LDSC Partitioned Heritability Analysis")
    print("Pelvic Floor GWAS - Phase 4")
    print("="*60)
    print(f"\nUsing baselineLD v2.2 model (97 annotations)")
    print(f"LDSC: {LDSC_DIR}")
    print(f"Results: {RESULTS_DIR}")

    # Check files
    if not check_files():
        print("\nPlease ensure all required files are present.")
        return

    # Run S-LDSC for each phenotype
    results = {}
    for pheno, sumstats in PHENOTYPES.items():
        success = run_sldsc_h2(pheno, sumstats)
        results[pheno] = success

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for pheno, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {pheno}: {status}")

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
