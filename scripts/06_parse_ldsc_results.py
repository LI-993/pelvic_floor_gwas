#!/usr/bin/env python3
"""
Parse LDSC genetic correlation results and create summary table.
"""

import pandas as pd
import re
from pathlib import Path

RESULTS_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\results\ldsc")
OUTPUT_FILE = RESULTS_DIR / "genetic_correlation_summary.tsv"

def parse_log_file(log_file):
    """Extract rg results from LDSC log file."""
    with open(log_file) as f:
        content = f.read()

    # Find the last Summary of Genetic Correlation Results section
    results = {}

    # Parse heritability for phenotype 1
    h2_match = re.search(r"Heritability of phenotype 1.*?Total Observed scale h2: ([\d.]+) \(([\d.]+)\)", content, re.DOTALL)
    if h2_match:
        results['h2_p1'] = float(h2_match.group(1))
        results['h2_p1_se'] = float(h2_match.group(2))

    # Parse heritability for phenotype 2
    h2_2_match = re.search(r"Heritability of phenotype 2/2.*?Total Observed scale h2: ([\d.]+) \(([\d.]+)\)", content, re.DOTALL)
    if h2_2_match:
        results['h2_p2'] = float(h2_2_match.group(1))
        results['h2_p2_se'] = float(h2_2_match.group(2))

    # Parse genetic correlation
    rg_match = re.search(r"Genetic Correlation: ([\d.-]+) \(([\d.]+)\)", content)
    if rg_match:
        results['rg'] = float(rg_match.group(1))
        results['rg_se'] = float(rg_match.group(2))

    # Parse Z-score
    z_match = re.search(r"Z-score: ([\d.-]+)", content)
    if z_match:
        results['z'] = float(z_match.group(1))

    # Parse P-value
    p_match = re.search(r"P: ([\d.e-]+)", content)
    if p_match:
        results['p'] = float(p_match.group(1))

    return results

def main():
    print("Parsing LDSC results...")

    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
    all_results = []

    for log_file in RESULTS_DIR.glob("*_vs_*.log"):
        name = log_file.stem
        parts = name.split("_vs_")
        p1, p2 = parts[0], parts[1]

        results = parse_log_file(log_file)
        results['phenotype1'] = p1
        results['phenotype2'] = p2
        all_results.append(results)

    df = pd.DataFrame(all_results)
    df = df[['phenotype1', 'phenotype2', 'rg', 'rg_se', 'z', 'p', 'h2_p1', 'h2_p1_se', 'h2_p2', 'h2_p2_se']]
    df = df.sort_values(['phenotype1', 'phenotype2'])

    # Save
    df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 80)
    print("GENETIC CORRELATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Phenotype 1':<16} {'Phenotype 2':<16} {'rg':>8} {'SE':>8} {'P-value':>12} {'Sig':>5}")
    print("-" * 80)

    for _, row in df.iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        print(f"{row['phenotype1']:<16} {row['phenotype2']:<16} {row['rg']:>8.4f} {row['rg_se']:>8.4f} {row['p']:>12.2e} {sig:>5}")

    print("\n" + "=" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # Create correlation matrix
    print("\n\nGenetic Correlation Matrix:")
    print("-" * 80)

    matrix = pd.DataFrame(index=phenotypes, columns=phenotypes, dtype=float)
    for i, p in enumerate(phenotypes):
        matrix.loc[p, p] = 1.0

    for _, row in df.iterrows():
        matrix.loc[row['phenotype1'], row['phenotype2']] = row['rg']
        matrix.loc[row['phenotype2'], row['phenotype1']] = row['rg']

    matrix_file = RESULTS_DIR / "genetic_correlation_matrix.tsv"
    matrix.to_csv(matrix_file, sep='\t')
    print(matrix.round(3).to_string())
    print(f"\nMatrix saved to: {matrix_file}")

if __name__ == "__main__":
    main()
