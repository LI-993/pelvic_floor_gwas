#!/usr/bin/env python3
"""Detailed drug-repurposing summary.

Loads the DGIdb interactions written by 01_drug_repurposing, identifies
known pelvic-floor drugs (validation), assigns priority scores combining
multi-phenotype evidence + genetic signal + FDA approval + interaction type,
and emits a markdown report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR, RESULTS_DIR

DRUG_DIR = RESULTS_DIR / "drug_repurposing"
MAGMA_DIR = RESULTS_DIR / "magma"

# Known approved drugs by indication.
KNOWN_DRUGS: dict[str, dict[str, list[str]]] = {
    "BPH": {
        "alpha_blockers": ["TAMSULOSIN", "ALFUZOSIN", "DOXAZOSIN", "TERAZOSIN", "SILODOSIN"],
        "5ari": ["FINASTERIDE", "DUTASTERIDE"],
        "pde5i": ["TADALAFIL", "SILDENAFIL"],
    },
    "Incontinence/OAB": {
        "anticholinergics": ["OXYBUTYNIN", "TOLTERODINE", "SOLIFENACIN", "DARIFENACIN", "FESOTERODINE", "TROSPIUM"],
        "beta3_agonists": ["MIRABEGRON", "VIBEGRON"],
    },
    "POP/Prolapse": {
        "hormones": ["ESTRADIOL", "ESTROGEN", "PROGESTERONE"],
    },
}
RELEVANT_INTERACTION_TYPES = ("inhibitor", "agonist", "antagonist", "modulator", "activator", "blocker")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(DRUG_DIR / "dgidb_interactions.csv")
    candidates = pd.read_csv(DRUG_DIR / "repurposing_candidates.csv")
    top_genes = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
    return interactions, candidates, top_genes


def find_known_drugs(interactions: pd.DataFrame) -> pd.DataFrame:
    found = []
    for condition, classes in KNOWN_DRUGS.items():
        for drug_class, drugs in classes.items():
            for drug in drugs:
                matches = interactions[interactions["drug"].str.upper().str.contains(drug, na=False)]
                for _, row in matches.iterrows():
                    found.append({
                        "gene": row["gene"],
                        "drug": row["drug"],
                        "condition": condition,
                        "drug_class": drug_class,
                        "interaction_type": row["interaction_type"],
                    })
    return pd.DataFrame(found)


def prioritize(candidates: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cand in candidates.iterrows():
        gene_int = interactions[interactions["gene"] == cand["gene_symbol"]]
        score = cand["n_phenotypes"] * 3
        if cand["min_p"] < 1e-10:
            score += 5
        elif cand["min_p"] < 1e-6:
            score += 3
        if gene_int[gene_int["drug"] == cand["drug"]]["approved"].any():
            score += 4
        if any(t in str(cand.get("interaction_type", "")).lower() for t in RELEVANT_INTERACTION_TYPES):
            score += 2
        rows.append({**cand.to_dict(), "priority_score": score})
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False)


def write_report(interactions: pd.DataFrame, candidates: pd.DataFrame, known_drugs: pd.DataFrame, prioritized: pd.DataFrame) -> str:
    lines = [
        "# Drug Repurposing Analysis - Detailed Summary",
        "",
        f"**Date**: {pd.Timestamp.now():%Y-%m-%d}",
        "",
        "## Executive Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total drug-gene interactions | {len(interactions)} |",
        f"| Unique drugs | {interactions['drug'].nunique()} |",
        f"| Unique genes | {interactions['gene'].nunique()} |",
        f"| FDA-approved drugs | {interactions[interactions['approved'] == True]['drug'].nunique()} |",
        f"| Known pelvic-floor drugs in results | {len(known_drugs)} |",
        "",
        "## Validation: known drugs in results",
    ]
    if known_drugs.empty:
        lines.append("\n_No currently-used pelvic-floor drugs found._")
    else:
        for _, row in known_drugs.drop_duplicates(["drug", "gene"]).iterrows():
            lines.append(f"- **{row['drug']}** ({row['drug_class']}) → {row['gene']} ({row['condition']})")

    lines += [
        "",
        "## Top priority candidates",
        "",
        "| Rank | Gene | Drug | Phenotypes | P | Score | Mechanism |",
        "|------|------|------|------------|---|-------|-----------|",
    ]
    for i, (_, row) in enumerate(prioritized.head(30).iterrows(), 1):
        gene = row["gene_symbol"]
        drug = (row["drug"] or "")[:30]
        phenos = (str(row["phenotypes"]) or "")[:25]
        lines.append(
            f"| {i} | {gene} | {drug} | {phenos} | {row['min_p']:.2e} | "
            f"{row['priority_score']} | {(str(row.get('interaction_type', '')) or '')[:20]} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    print("=" * 60)
    print("Drug Repurposing - Detailed Summary")
    print("=" * 60)

    interactions, candidates, _top_genes = load_data()

    print(f"\nApproved drugs: {interactions[interactions['approved'] == True]['drug'].nunique()}")

    known_drugs = find_known_drugs(interactions)
    print(f"Known pelvic-floor drugs in results: {len(known_drugs)}")

    prioritized = prioritize(candidates, interactions)
    prioritized.to_csv(DRUG_DIR / "prioritized_candidates.csv", index=False)
    print(f"Prioritized candidates: {len(prioritized)} -> {DRUG_DIR / 'prioritized_candidates.csv'}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "10_drug_repurposing.md"
    report_path.write_text(write_report(interactions, candidates, known_drugs, prioritized), encoding="utf-8")
    print(f"Report: {report_path}")

    print("\nTop 15 by priority:")
    for i, (_, row) in enumerate(prioritized.head(15).iterrows(), 1):
        print(f"  {i}. {row['gene_symbol']} -> {row['drug'][:40]} (score={row['priority_score']}, P={row['min_p']:.2e})")


if __name__ == "__main__":
    main()
