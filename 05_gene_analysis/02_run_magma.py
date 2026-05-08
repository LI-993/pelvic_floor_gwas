#!/usr/bin/env python3
"""Run MAGMA gene-based association analysis.

Two-step workflow:
  1. SNP-to-gene annotation against the 1000G EUR LD reference (one-shot).
  2. Per-phenotype gene-level p-values from the standardized sumstats.
Outputs `{pheno}_genes.genes.out` and a small summary CSV.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MAGMA_BIN,
    MAGMA_G1000_EUR,
    MAGMA_GENE_LOC,
    PHENOTYPES,
    RESULTS_DIR,
    SAMPLE_SIZES,
    sumstats_path,
)


def prepare_magma_input(pheno: str, out_dir: Path) -> Path:
    """Write the MAGMA-format SNP/P/N table for a phenotype."""
    print(f"  Preparing input for {pheno}...")
    df = pd.read_csv(sumstats_path(pheno), sep="\t", compression="gzip")
    magma_df = pd.DataFrame({"SNP": df["SNP"], "P": df["P"], "N": SAMPLE_SIZES[pheno]}).dropna()

    out_path = out_dir / f"{pheno}_magma_input.txt"
    magma_df.to_csv(out_path, sep="\t", index=False)
    print(f"    {len(magma_df)} SNPs -> {out_path.name}")
    return out_path


def run_annotation(out_dir: Path) -> Path:
    """SNP-to-gene mapping. Idempotent — skips if `gene_annotation.genes.annot` exists."""
    print(f"\n{'=' * 60}\nStep 1: SNP-to-gene annotation\n{'=' * 60}")
    annot_prefix = out_dir / "gene_annotation"
    if (out_dir / "gene_annotation.genes.annot").exists():
        print("Annotation exists; skipping.")
        return annot_prefix

    cmd = [
        str(MAGMA_BIN),
        "--annotate",
        "--snp-loc", f"{MAGMA_G1000_EUR}.bim",
        "--gene-loc", str(MAGMA_GENE_LOC),
        "--out", str(annot_prefix),
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print("Annotation complete.")
    return annot_prefix


def run_gene_analysis(pheno: str, input_file: Path, annot_prefix: Path, out_dir: Path) -> bool:
    print(f"\n  Running gene analysis for {pheno}...")
    out_prefix = out_dir / f"{pheno}_genes"
    cmd = [
        str(MAGMA_BIN),
        "--bfile", str(MAGMA_G1000_EUR),
        "--pval", str(input_file), "use=SNP,P", "ncol=N",
        "--gene-annot", f"{annot_prefix}.genes.annot",
        "--out", str(out_prefix),
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  -> {out_prefix}.genes.out")
        return True
    print(f"  Error: {result.stderr}")
    return False


def summarize(out_dir: Path) -> None:
    """Per-phenotype gene counts at p<0.05 and Bonferroni; write magma_summary.csv."""
    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    rows = []
    for pheno in PHENOTYPES:
        path = out_dir / f"{pheno}_genes.genes.out"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=r"\s+")
        n_genes = len(df)
        n_05 = (df["P"] < 0.05).sum()
        n_bonf = (df["P"] < 0.05 / n_genes).sum()
        print(f"\n{pheno}: {n_genes} genes | nominal {n_05} | Bonferroni {n_bonf}")
        for _, row in df.nsmallest(5, "P")[["GENE", "NSNPS", "ZSTAT", "P"]].iterrows():
            print(f"  {row['GENE']}: P={row['P']:.2e}, Z={row['ZSTAT']:.2f}")
        rows.append({"Phenotype": pheno, "N_genes": n_genes, "N_sig_005": n_05, "N_bonferroni": n_bonf})

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "magma_summary.csv", index=False)
        print(f"\nSummary -> {out_dir / 'magma_summary.csv'}")


def main() -> None:
    out_dir = RESULTS_DIR / "magma"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MAGMA Gene-based Analysis")
    print("=" * 60)

    annot_prefix = run_annotation(out_dir)
    print(f"\n{'=' * 60}\nStep 2: Gene-level p-values\n{'=' * 60}")
    for pheno in PHENOTYPES:
        print(f"\n[{pheno}]")
        input_file = prepare_magma_input(pheno, out_dir)
        run_gene_analysis(pheno, input_file, annot_prefix, out_dir)

    summarize(out_dir)


if __name__ == "__main__":
    main()
