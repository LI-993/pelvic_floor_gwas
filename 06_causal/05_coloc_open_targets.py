#!/usr/bin/env python3
"""Pull pre-computed colocalization (PP.H4) and L2G hits from Open Targets.

For each top prioritized gene the script resolves the Ensembl ID, fetches the
gene's L2G associations (limited to 30 per the API), then queries the
`colocalisationsForGene` endpoint and filters/summarizes strong (PP.H4 > 0.8)
and moderate (> 0.5) colocalizations. SSL verification is disabled because the
Open Targets endpoint occasionally serves an intermediate cert that older
Windows trust stores reject.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUT = RESULTS_DIR / "coloc_formal"
API_URL = "https://api.genetics.opentargets.org/graphql"
REQUEST_DELAY = 0.3
PFD_KEYWORDS = ("prolapse", "incontinence", "prostat", "bladder", "constipation", "urinary", "pelvic")

TOP_GENES: list[str] = [
    "FGFR2", "WT1", "HNF1B", "WNT4", "SMAD3", "ESR1", "TNXB",
    "DNAH11", "LOXL1", "HOXA13", "ELN", "FBN1", "COL1A1",
    "BCL11A", "TGFBR2", "COL3A1", "FBLN5",
]


GENE_SEARCH_QUERY = """
query GeneSearch($q: String!) {
  search(queryString: $q) {
    genes { id symbol }
  }
}
"""

L2G_QUERY = """
query GeneL2G($geneId: String!) {
  studiesAndLeadVariantsForGeneByL2G(geneId: $geneId, pageSize: 30) {
    rows {
      study { studyId traitReported traitEfos source pmid }
      variant { id rsId }
      pval
      yProbaModel
    }
  }
}
"""

COLOC_QUERY = """
query GeneColoc($geneId: String!) {
  colocalisationsForGene(geneId: $geneId) {
    leftVariant { id rsId }
    leftStudy { studyId traitReported source }
    rightVariant { id rsId }
    rightStudy { studyId traitReported source }
    h3 h4 log2h4h3
  }
}
"""


def graphql(session: requests.Session, query: str, variables: dict) -> dict | None:
    try:
        r = session.post(API_URL, json={"query": query, "variables": variables}, timeout=30, verify=False)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        print(f"    API error: {e}")
        return None


def lookup_gene_id(session: requests.Session, symbol: str) -> str | None:
    res = graphql(session, GENE_SEARCH_QUERY, {"q": symbol})
    if not res:
        return None
    for g in res.get("data", {}).get("search", {}).get("genes", []) or []:
        if g["symbol"] == symbol:
            return g["id"]
    return None


def fetch_l2g(session: requests.Session, gene_id: str) -> list[dict]:
    res = graphql(session, L2G_QUERY, {"geneId": gene_id})
    if not res:
        return []
    payload = res.get("data", {}).get("studiesAndLeadVariantsForGeneByL2G") or {}
    return payload.get("rows", [])


def fetch_coloc(session: requests.Session, gene_id: str) -> list[dict]:
    res = graphql(session, COLOC_QUERY, {"geneId": gene_id})
    if not res:
        return []
    return res.get("data", {}).get("colocalisationsForGene") or []


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path} ({len(rows)} rows)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Open Targets colocalization + L2G")
    print("=" * 60)

    session = requests.Session()
    all_coloc, all_l2g, gene_summary = [], [], []

    for gene in TOP_GENES:
        print(f"\n[{gene}]", end=" ")
        gene_id = lookup_gene_id(session, gene)
        if not gene_id:
            print("not found")
            continue
        print(f"({gene_id})")

        l2g_rows = fetch_l2g(session, gene_id)
        print(f"  L2G associations: {len(l2g_rows)}")
        for row in l2g_rows:
            study = row.get("study", {})
            variant = row.get("variant") or {}
            all_l2g.append({
                "gene": gene, "gene_id": gene_id,
                "study_id": study.get("studyId", ""),
                "trait": study.get("traitReported", ""),
                "source": study.get("source", ""),
                "variant": variant.get("rsId", ""),
                "pval": row.get("pval", ""),
                "l2g_score": row.get("yProbaModel", ""),
            })

        coloc_rows = fetch_coloc(session, gene_id)
        print(f"  Colocalizations: {len(coloc_rows)}")
        n_strong = n_moderate = 0
        best_h4 = 0.0
        best_label = ""
        for cr in coloc_rows:
            h4 = cr.get("h4", 0) or 0
            ls = cr.get("leftStudy", {})
            rs = cr.get("rightStudy", {})
            lv = cr.get("leftVariant") or {}
            rv = cr.get("rightVariant") or {}
            all_coloc.append({
                "gene": gene, "gene_id": gene_id,
                "left_study": ls.get("studyId", ""), "left_trait": ls.get("traitReported", ""),
                "left_source": ls.get("source", ""),
                "right_study": rs.get("studyId", ""), "right_trait": rs.get("traitReported", ""),
                "right_source": rs.get("source", ""),
                "left_variant": lv.get("rsId", ""), "right_variant": rv.get("rsId", ""),
                "PP_H3": cr.get("h3", 0) or 0, "PP_H4": h4,
                "log2_H4_H3": cr.get("log2h4h3", ""),
            })
            if h4 > 0.8:
                n_strong += 1
            elif h4 > 0.5:
                n_moderate += 1
            if h4 > best_h4:
                best_h4 = h4
                best_label = f"{ls.get('traitReported', '?')} <-> {rs.get('traitReported', '?')} ({rs.get('source', '')})"

        if n_strong or n_moderate:
            print(f"  Strong (H4>0.8): {n_strong} | Moderate (H4>0.5): {n_moderate}")
            print(f"  Best: PP.H4={best_h4:.3f} | {best_label}")

        gene_summary.append({
            "gene": gene, "gene_id": gene_id,
            "n_l2g": len(l2g_rows),
            "n_coloc_total": len(coloc_rows),
            "n_coloc_strong": n_strong,
            "n_coloc_moderate": n_moderate,
            "best_h4": best_h4,
            "best_coloc": best_label,
        })
        time.sleep(REQUEST_DELAY)

    write_csv(OUT / "opentargets_l2g_results.csv", all_l2g)
    write_csv(OUT / "opentargets_coloc_results.csv", all_coloc)
    write_csv(OUT / "coloc_gene_summary.csv", gene_summary)

    print("\n=== Colocalization summary ===")
    print(f"{'Gene':<10} {'L2G':>4} {'Coloc':>6} {'H4>0.8':>7} {'H4>0.5':>7} {'Best H4':>8}")
    print("-" * 50)
    for gs in gene_summary:
        print(f"{gs['gene']:<10} {gs['n_l2g']:>4} {gs['n_coloc_total']:>6} "
              f"{gs['n_coloc_strong']:>7} {gs['n_coloc_moderate']:>7} {gs['best_h4']:>8.3f}")

    total_strong = sum(gs["n_coloc_strong"] for gs in gene_summary)
    total_moderate = sum(gs["n_coloc_moderate"] for gs in gene_summary)
    print(f"\nTotal strong (PP.H4 > 0.8): {total_strong}")
    print(f"Total moderate (PP.H4 > 0.5): {total_moderate}")


if __name__ == "__main__":
    main()
