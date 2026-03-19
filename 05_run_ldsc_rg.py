#!/usr/bin/env python3
"""
Run LDSC genetic correlation analysis for all phenotype pairs.
"""

import subprocess
import sys
from pathlib import Path
from itertools import combinations

PYTHON = r"D:\miniconda3\envs\ldsc_py311\python.exe"
LDSC = r"D:\Nproject\gwas\ldsc-python3\ldsc.py"
REF = r"D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr\@"
DATA_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\data\ldsc")
OUT_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\results\ldsc")

OUT_DIR.mkdir(parents=True, exist_ok=True)

phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
pairs = list(combinations(phenotypes, 2))

print(f"Running LDSC rg for {len(pairs)} phenotype pairs")
print("=" * 60)

for i, (p1, p2) in enumerate(pairs, 1):
    print(f"\n[{i}/{len(pairs)}] {p1} vs {p2}")

    file1 = DATA_DIR / f"{p1}.sumstats.gz"
    file2 = DATA_DIR / f"{p2}.sumstats.gz"
    out = OUT_DIR / f"{p1}_vs_{p2}"

    cmd = [
        PYTHON, LDSC,
        "--rg", f"{file1},{file2}",
        "--ref-ld-chr", REF,
        "--w-ld-chr", REF,
        "--out", str(out),
        "--no-check-alleles"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    else:
        # Parse result from log
        log_file = f"{out}.log"
        with open(log_file) as f:
            content = f.read()
            if "Genetic Correlation:" in content:
                for line in content.split('\n'):
                    if "Genetic Correlation:" in line:
                        print(f"  {line.strip()}")
                    elif "P:" in line and "P-value" not in line:
                        print(f"  {line.strip()}")
            else:
                print("  Analysis completed")

print("\n" + "=" * 60)
print("All analyses complete!")
print(f"Results saved to: {OUT_DIR}")
