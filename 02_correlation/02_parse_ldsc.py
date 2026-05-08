#!/usr/bin/env python3
"""Parse LDSC `--rg` logs into a tidy summary table + correlation matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PHENOTYPES, RESULTS_DIR
from utils.ldsc import parse_rg_log
from utils.plotting import significance_marker


def main() -> None:
    results_dir = RESULTS_DIR / "ldsc"

    rows: list[dict] = []
    for log_file in results_dir.glob("*_vs_*.log"):
        p1, p2 = log_file.stem.split("_vs_")
        rec = parse_rg_log(log_file)
        rec.update({"phenotype1": p1, "phenotype2": p2})
        rows.append(rec)

    df = pd.DataFrame(rows)
    cols = ["phenotype1", "phenotype2", "rg", "rg_se", "z", "p", "h2_p1", "h2_p1_se", "h2_p2", "h2_p2_se"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["phenotype1", "phenotype2"])

    summary_path = results_dir / "genetic_correlation_summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("GENETIC CORRELATION RESULTS")
    print("=" * 80)
    print(f"{'Phenotype 1':<16} {'Phenotype 2':<16} {'rg':>8} {'SE':>8} {'P-value':>12} {'Sig':>5}")
    print("-" * 80)
    for _, r in df.iterrows():
        print(
            f"{r['phenotype1']:<16} {r['phenotype2']:<16} "
            f"{r['rg']:>8.4f} {r['rg_se']:>8.4f} {r['p']:>12.2e} {significance_marker(r['p']):>5}"
        )
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # Symmetric correlation matrix
    matrix = pd.DataFrame(index=PHENOTYPES, columns=PHENOTYPES, dtype=float)
    for p in PHENOTYPES:
        matrix.loc[p, p] = 1.0
    for _, r in df.iterrows():
        matrix.loc[r["phenotype1"], r["phenotype2"]] = r["rg"]
        matrix.loc[r["phenotype2"], r["phenotype1"]] = r["rg"]

    matrix_path = results_dir / "genetic_correlation_matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t")
    print("\nGenetic Correlation Matrix:")
    print(matrix.round(3).to_string())
    print(f"\nSaved: {matrix_path}")


if __name__ == "__main__":
    main()
