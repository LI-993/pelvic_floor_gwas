#!/usr/bin/env python3
"""
Run MAGMA gene-based analysis for pelvic floor phenotypes.

MAGMA workflow:
1. Annotation: Map SNPs to genes based on location
2. Gene analysis: Compute gene-level p-values from GWAS summary stats
"""

import subprocess
import gzip
from pathlib import Path
import pandas as pd

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
MAGMA = BASE_DIR / "tools/magma/magma.exe"

# Reference files
GENE_LOC = BASE_DIR / "reference/magma/NCBI37.3.gene.loc"
G1000_EUR = BASE_DIR / "reference/magma/g1000_eur"

# Input/Output
SUMSTATS_DIR = BASE_DIR / "data/processed"
RESULTS_DIR = BASE_DIR / "results/magma"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Phenotypes and their summary stats files
PHENOTYPES = {
    "POP": "POP_GRCh38.tsv.gz",
    "BPH": "BPH_GRCh38.tsv.gz",
    "Bladder": "Bladder_GRCh38.tsv.gz",
    "Constipation": "Constipation_GRCh38.tsv.gz",
    "FemaleProlapse": "FemaleProlapse_GRCh38.tsv.gz",
    "Incontinence": "Incontinence_GRCh38.tsv.gz",
}

# Sample sizes (for MAGMA)
SAMPLE_SIZES = {
    "POP": 574377,
    "BPH": 501137,
    "Bladder": 503550,
    "Constipation": 501956,
    "FemaleProlapse": 503074,
    "Incontinence": 430019,
}


def prepare_magma_input(pheno, sumstats_file):
    """
    Convert GWAS summary stats to MAGMA format.
    Required columns: SNP, P, N
    """
    print(f"  Preparing input for {pheno}...")

    input_path = SUMSTATS_DIR / sumstats_file
    output_path = RESULTS_DIR / f"{pheno}_magma_input.txt"

    # Read summary stats
    df = pd.read_csv(input_path, sep='\t', compression='gzip')

    # Select and rename columns for MAGMA
    # MAGMA needs: SNP, P (and optionally N)
    magma_df = pd.DataFrame({
        'SNP': df['SNP'],
        'P': df['P'],
        'N': SAMPLE_SIZES[pheno]
    })

    # Remove any rows with missing values
    magma_df = magma_df.dropna()

    # Save
    magma_df.to_csv(output_path, sep='\t', index=False)
    print(f"    Saved {len(magma_df)} SNPs to {output_path.name}")

    return output_path


def run_annotation():
    """
    Step 1: SNP-to-gene annotation using gene locations.
    This only needs to be done once.
    """
    print("\n" + "="*60)
    print("Step 1: SNP-to-Gene Annotation")
    print("="*60)

    annot_output = RESULTS_DIR / "gene_annotation"

    # Check if already done
    if (RESULTS_DIR / "gene_annotation.genes.annot").exists():
        print("Annotation already exists, skipping...")
        return annot_output

    cmd = [
        str(MAGMA),
        "--annotate",
        "--snp-loc", str(G1000_EUR) + ".bim",
        "--gene-loc", str(GENE_LOC),
        "--out", str(annot_output)
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Annotation complete!")
    else:
        print(f"Error: {result.stderr}")

    return annot_output


def run_gene_analysis(pheno, input_file, annot_prefix):
    """
    Step 2: Gene-based analysis using GWAS summary statistics.
    """
    print(f"\n  Running gene analysis for {pheno}...")

    output_prefix = RESULTS_DIR / f"{pheno}_genes"

    cmd = [
        str(MAGMA),
        "--bfile", str(G1000_EUR),
        "--pval", str(input_file),
        "use=SNP,P",
        "ncol=N",
        "--gene-annot", str(annot_prefix) + ".genes.annot",
        "--out", str(output_prefix)
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Success! Output: {output_prefix}.genes.out")
        return True
    else:
        print(f"  Error: {result.stderr}")
        return False


def summarize_results():
    """Parse and summarize MAGMA results."""
    print("\n" + "="*60)
    print("Summary of Gene-Based Results")
    print("="*60)

    all_results = []

    for pheno in PHENOTYPES:
        results_file = RESULTS_DIR / f"{pheno}_genes.genes.out"
        if results_file.exists():
            df = pd.read_csv(results_file, sep='\s+')

            # Count significant genes at different thresholds
            n_genes = len(df)
            n_sig_005 = len(df[df['P'] < 0.05])
            n_bonf = len(df[df['P'] < 0.05/n_genes])
            n_fdr = len(df[df['P'] < 0.05 * df['P'].rank() / n_genes])

            print(f"\n{pheno}:")
            print(f"  Total genes tested: {n_genes}")
            print(f"  Nominally significant (p<0.05): {n_sig_005}")
            print(f"  Bonferroni significant: {n_bonf}")

            # Top genes
            top_genes = df.nsmallest(5, 'P')[['GENE', 'NSNPS', 'ZSTAT', 'P']]
            print(f"  Top 5 genes:")
            for _, row in top_genes.iterrows():
                print(f"    {row['GENE']}: P={row['P']:.2e}, Z={row['ZSTAT']:.2f}")

            all_results.append({
                'Phenotype': pheno,
                'N_genes': n_genes,
                'N_sig_005': n_sig_005,
                'N_bonferroni': n_bonf
            })

    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(RESULTS_DIR / "magma_summary.csv", index=False)
        print(f"\nSummary saved to: {RESULTS_DIR / 'magma_summary.csv'}")


def main():
    print("="*60)
    print("MAGMA Gene-Based Analysis")
    print("Pelvic Floor GWAS - Phase 5")
    print("="*60)

    # Step 1: Run annotation (once)
    annot_prefix = run_annotation()

    # Step 2: Prepare input and run gene analysis for each phenotype
    print("\n" + "="*60)
    print("Step 2: Gene-Based Analysis")
    print("="*60)

    for pheno, sumstats in PHENOTYPES.items():
        print(f"\n[{pheno}]")

        # Prepare input
        input_file = prepare_magma_input(pheno, sumstats)

        # Run gene analysis
        run_gene_analysis(pheno, input_file, annot_prefix)

    # Summarize results
    summarize_results()

    print("\n" + "="*60)
    print("MAGMA Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
