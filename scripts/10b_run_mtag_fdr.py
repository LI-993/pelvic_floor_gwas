#!/usr/bin/env python3
"""Run MTAG cross-phenotype meta-analysis with --fdr to compute maxFDR."""

import subprocess
import os

BASE_DIR = "D:/Nproject/gwas/pelvic_floor_gwas"
MTAG_DIR = f"{BASE_DIR}/mtag"
DATA_DIR = f"{BASE_DIR}/data/ldsc"
OUT_DIR = f"{BASE_DIR}/results/mtag"

os.makedirs(OUT_DIR, exist_ok=True)

# Build sumstats list
phenotypes = ["POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence"]
sumstats_files = [f"{DATA_DIR}/{p}.sumstats.gz" for p in phenotypes]
sumstats_arg = ",".join(sumstats_files)

# Run MTAG with FDR calculation
# --fdr: enable maxFDR calculation
# --intervals 5: partition [0,1] into 5 intervals (fewer grid points for 6 traits)
# --fit_ss: restrict grid search using spike-slab fitted causal probabilities
# --out: use separate output name to avoid overwriting existing results
cmd = [
    "python", f"{MTAG_DIR}/mtag.py",
    "--sumstats", sumstats_arg,
    "--out", f"{OUT_DIR}/pelvic_floor_fdr",
    "--snp_name", "SNP",
    "--z_name", "Z",
    "--n_name", "N",
    "--a1_name", "A1",
    "--a2_name", "A2",
    "--p_name", "P",
    "--ld_ref_panel", f"{MTAG_DIR}/ld_ref_panel/eur_w_ld_chr/",
    "--maf_min", "0",
    "--no_chr_data",
    "--stream_stdout",
    "--fdr",
    "--intervals", "3",
    "--skip_mtag",
]

print("Running MTAG with FDR calculation:")
print(" ".join(cmd))
print("\n" + "="*60 + "\n")

result = subprocess.run(cmd, cwd=MTAG_DIR)
print("\nMTAG FDR completed with exit code:", result.returncode)
