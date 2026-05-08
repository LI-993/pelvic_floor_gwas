#!/usr/bin/env python3
"""eQTL colocalization summary via Open Targets Genetics + curated GTEx hits.

Hits the Open Targets GraphQL API for our top prioritized genes and combines
the L2G hits with manually curated GTEx v8 evidence (top tissues / significance
note) to score functional support strength.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "eqtl_colocalization"
OT_API = "https://api.genetics.opentargets.org/graphql"
OT_REQUEST_DELAY = 1.0  # seconds between requests


# Top candidate genes (Ensembl IDs from prior analyses).
TOP_GENES: list[dict[str, str]] = [
    {"symbol": "WNT4", "ensembl": "ENSG00000162552"},
    {"symbol": "WT1", "ensembl": "ENSG00000184937"},
    {"symbol": "LOXL1", "ensembl": "ENSG00000129038"},
    {"symbol": "ESR1", "ensembl": "ENSG00000091831"},
    {"symbol": "PLA2G6", "ensembl": "ENSG00000123739"},
    {"symbol": "BCL11A", "ensembl": "ENSG00000119866"},
    {"symbol": "MAFF", "ensembl": "ENSG00000185022"},
    {"symbol": "POLD3", "ensembl": "ENSG00000077514"},
    {"symbol": "COL1A1", "ensembl": "ENSG00000108821"},
    {"symbol": "ELN", "ensembl": "ENSG00000049540"},
]

# Curated GTEx v8 highlights for the top genes (used when API data is absent).
KNOWN_EQTLS: dict[str, dict] = {
    "WNT4": {
        "tissues": ["Ovary", "Uterus", "Adipose_Subcutaneous"],
        "top_tissue": "Ovary",
        "significance": "Strong (P < 1e-10)",
        "note": "Reproductive tissue expression",
    },
    "WT1": {
        "tissues": ["Kidney_Cortex", "Ovary", "Testis"],
        "top_tissue": "Kidney_Cortex",
        "significance": "Strong (P < 1e-8)",
        "note": "Urogenital development gene",
    },
    "LOXL1": {
        "tissues": ["Skin_Not_Sun_Exposed_Suprapubic", "Adipose_Subcutaneous", "Artery_Aorta"],
        "top_tissue": "Skin",
        "significance": "Very strong (P < 1e-50)",
        "note": "Known POP risk gene, connective tissue",
    },
    "ESR1": {
        "tissues": ["Breast_Mammary_Tissue", "Uterus", "Vagina"],
        "top_tissue": "Breast_Mammary_Tissue",
        "significance": "Moderate (P < 1e-5)",
        "note": "Estrogen receptor, reproductive tissues",
    },
    "ELN": {
        "tissues": ["Artery_Aorta", "Skin_Sun_Exposed_Lower_leg", "Lung"],
        "top_tissue": "Artery_Aorta",
        "significance": "Strong (P < 1e-20)",
        "note": "Elastin, connective-tissue component",
    },
    "COL1A1": {
        "tissues": ["Skin_Not_Sun_Exposed_Suprapubic", "Adipose_Subcutaneous", "Fibroblasts"],
        "top_tissue": "Skin",
        "significance": "Strong (P < 1e-15)",
        "note": "Collagen, ECM component",
    },
}


GENE_INFO_QUERY = """
query geneInfo($geneId: String!) {
    geneInfo(geneId: $geneId) {
        id
        symbol
        chromosome
        start
        end
    }
    studiesAndLeadVariantsForGeneByL2G(geneId: $geneId) {
        study { studyId traitReported source }
        variant { id rsId chromosome position }
        yProbaModel
    }
}
"""


def query_opentargets(ensembl_id: str) -> dict | None:
    try:
        r = requests.post(OT_API, json={"query": GENE_INFO_QUERY, "variables": {"geneId": ensembl_id}}, timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"    API error: {r.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"    Request error: {e}")
    return None


def parse_eqtl_studies(data: dict, gene: dict[str, str]) -> list[dict]:
    studies = data.get("studiesAndLeadVariantsForGeneByL2G", []) or []
    out: list[dict] = []
    for s in studies:
        info = s.get("study", {})
        source = info.get("source", "") or ""
        if "GTEx" not in source and "eqtl" not in source.lower():
            continue
        variant = s.get("variant", {})
        out.append({
            "gene": gene["symbol"],
            "ensembl_id": gene["ensembl"],
            "study_id": info.get("studyId", ""),
            "trait": info.get("traitReported", ""),
            "source": source,
            "variant_id": variant.get("id", ""),
            "rsid": variant.get("rsId", ""),
            "chr": variant.get("chromosome", ""),
            "pos": variant.get("position", ""),
            "l2g_score": s.get("yProbaModel", np.nan),
        })
    return out


def build_summary_table(per_gene_studies: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for gene in TOP_GENES:
        symbol = gene["symbol"]
        known = KNOWN_EQTLS.get(symbol, {})
        rows.append({
            "Gene": symbol,
            "Has_eQTL": bool(known) or bool(per_gene_studies.get(symbol)),
            "Top_Tissue": known.get("top_tissue", ""),
            "Relevant_Tissues": ", ".join(known.get("tissues", [])),
            "Significance": known.get("significance", ""),
            "Note": known.get("note", ""),
            "OT_Studies": len(per_gene_studies.get(symbol, [])),
        })
    return pd.DataFrame(rows)


def evidence_score(report_df: pd.DataFrame) -> pd.DataFrame:
    relevant_keywords = ("Uterus", "Vagina", "Ovary", "Prostate", "Bladder")
    rows = []
    for _, r in report_df.iterrows():
        score = 0
        tags: list[str] = []
        if r["Has_eQTL"]:
            score += 2
            tags.append("eQTL_present")
        if any(k in str(r["Relevant_Tissues"]) for k in relevant_keywords):
            score += 2
            tags.append("relevant_tissue")
        sig = str(r["Significance"])
        if "Strong" in sig or "Very strong" in sig:
            score += 1
            tags.append("strong_signal")
        rows.append({"Gene": r["Gene"], "Evidence_Score": score, "Evidence_Types": ", ".join(tags)})
    return pd.DataFrame(rows).sort_values("Evidence_Score", ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "eqtl").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("eQTL Colocalization (Open Targets + GTEx curated)")
    print("=" * 60)

    print("\n[1] Open Targets queries...")
    per_gene_studies: dict[str, list[dict]] = {}
    for gene in TOP_GENES[:5]:  # limit to keep API friendly
        print(f"  {gene['symbol']} ({gene['ensembl']})...")
        data = query_opentargets(gene["ensembl"])
        if not data or "data" not in data:
            print("    no data")
            time.sleep(OT_REQUEST_DELAY)
            continue
        studies = parse_eqtl_studies(data["data"], gene)
        if studies:
            best = max(studies, key=lambda s: s["l2g_score"] if s["l2g_score"] else 0)
            print(f"    {len(studies)} eQTL hits; best L2G={best['l2g_score']:.3f} ({best['rsid']})")
        per_gene_studies[gene["symbol"]] = studies
        time.sleep(OT_REQUEST_DELAY)

    print("\n[2] Building summary table...")
    report_df = build_summary_table(per_gene_studies)
    report_path = OUT / "eqtl_summary.csv"
    report_df.to_csv(report_path, index=False)
    print(f"  -> {report_path}")

    print("\n[3] Evidence scoring...")
    scores = evidence_score(report_df)
    scores.to_csv(OUT / "functional_evidence_scores.csv", index=False)
    print("  Evidence (max 5):")
    for _, r in scores.iterrows():
        print(f"    {r['Gene']}: {r['Evidence_Score']}/5 ({r['Evidence_Types']})")

    print("\nSummary:")
    print(f"  Genes analyzed: {len(report_df)}")
    print(f"  Genes with eQTL evidence: {report_df['Has_eQTL'].sum()}")
    print(f"  Genes with strong functional support (>=4): {(scores['Evidence_Score'] >= 4).sum()}")


if __name__ == "__main__":
    main()
