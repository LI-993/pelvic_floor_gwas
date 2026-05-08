#!/usr/bin/env python3
"""Drug repurposing via DGIdb GraphQL API.

Pulls drug-gene interactions for Bonferroni-significant MAGMA genes (with a
top-genes fallback when no gene clears Bonferroni), categorizes drugs against
a curated pelvic-floor drug-class map (alpha-blockers, 5-ARI, anticholinergics,
beta-3 agonists, PDE5i, hormones, etc.), and ranks repurposing candidates by
multi-phenotype evidence.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR, MAGMA_GENE_LOC, PHENOTYPES, RESULTS_DIR

OUT = RESULTS_DIR / "drug_repurposing"
MAGMA_DIR = RESULTS_DIR / "magma"
DGIDB_GRAPHQL = "https://dgidb.org/api/graphql"
P_THRESHOLD = 0.05 / 19_000  # Bonferroni against ~19k MAGMA genes

PELVIC_DRUG_KEYWORDS: dict[str, list[str]] = {
    "alpha_blockers": ["tamsulosin", "alfuzosin", "doxazosin", "terazosin", "prazosin", "silodosin"],
    "5ari": ["finasteride", "dutasteride"],
    "anticholinergics": ["oxybutynin", "tolterodine", "solifenacin", "darifenacin", "fesoterodine", "trospium"],
    "beta3_agonists": ["mirabegron", "vibegron"],
    "pde5_inhibitors": ["sildenafil", "tadalafil", "vardenafil", "avanafil"],
    "hormones": ["estrogen", "estradiol", "testosterone", "progesterone"],
    "collagen_modulators": ["penicillamine", "collagenase"],
    "anti_inflammatories": ["ibuprofen", "naproxen", "celecoxib", "diclofenac"],
    "muscle_relaxants": ["baclofen", "diazepam", "cyclobenzaprine", "tizanidine"],
}

INTERACTION_QUERY = """
query {
  genes(names: %s) {
    nodes {
      name
      longName
      interactions {
        drug { name conceptId approved }
        interactionScore
        interactionTypes { type directionality }
        interactionAttributes { name value }
        sources { fullName }
        publications { pmid }
      }
    }
  }
}
"""


def load_significant_genes() -> dict[str, dict]:
    """Aggregate genes that clear Bonferroni in any phenotype's MAGMA output."""
    print("Loading significant genes...")
    genes: dict[str, dict] = {}
    for pheno in PHENOTYPES:
        path = MAGMA_DIR / f"{pheno}_genes.genes.out.txt"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=r"\s+")
        sig = df[df["P"] < P_THRESHOLD]
        for _, row in sig.iterrows():
            gid = row["GENE"]
            rec = genes.setdefault(gid, {"gene_id": gid, "phenotypes": [], "min_p": row["P"], "max_z": row["ZSTAT"]})
            rec["phenotypes"].append(pheno)
            rec["min_p"] = min(rec["min_p"], row["P"])
            rec["max_z"] = max(rec["max_z"], row["ZSTAT"])
    print(f"  {len(genes)} Bonferroni-significant unique genes")
    return genes


def load_gene_annotations() -> dict[str, str]:
    if not MAGMA_GENE_LOC.exists():
        return {}
    df = pd.read_csv(MAGMA_GENE_LOC, sep="\t", header=None, names=["gene_id", "chr", "start", "end", "strand", "symbol"])
    return dict(zip(df["gene_id"], df["symbol"]))


def query_dgidb(symbols: set[str], batch_size: int = 50) -> list[dict]:
    print(f"\nQuerying DGIdb for {len(symbols)} symbols...")
    interactions: list[dict] = []
    sym_list = list(symbols)
    for i in range(0, len(sym_list), batch_size):
        batch = sym_list[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} genes")
        q = INTERACTION_QUERY % json.dumps(batch)
        req = urllib.request.Request(
            DGIDB_GRAPHQL,
            data=json.dumps({"query": q}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"    Error: {e}")
            time.sleep(0.3)
            continue

        if "errors" in payload:
            print(f"    GraphQL errors: {payload['errors']}")
            time.sleep(0.3)
            continue

        for node in payload.get("data", {}).get("genes", {}).get("nodes", []):
            gene = node.get("name", "")
            for ia in node.get("interactions", []):
                drug = ia.get("drug", {}) or {}
                interactions.append({
                    "gene": gene,
                    "drug": drug.get("name", ""),
                    "interaction_type": [it.get("type", "") for it in ia.get("interactionTypes", [])],
                    "sources": [s.get("fullName", "") for s in ia.get("sources", [])],
                    "pmid": [p.get("pmid", "") for p in ia.get("publications", [])],
                    "drug_concept_id": drug.get("conceptId", ""),
                    "approved": bool(drug.get("approved", False)),
                    "score": ia.get("interactionScore", 0),
                })
        time.sleep(0.3)
    print(f"  Total interactions: {len(interactions)}")
    return interactions


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    def category(name: str) -> str:
        lower = (name or "").lower()
        for cat, kws in PELVIC_DRUG_KEYWORDS.items():
            if any(k in lower for k in kws):
                return cat
        return "other"

    df["drug_category"] = df["drug"].apply(category)
    return df


def make_candidate_table(interactions_df: pd.DataFrame, gene_info: dict[str, dict], symbols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for gene_id, info in gene_info.items():
        symbol = symbols.get(gene_id, str(gene_id))
        gene_drugs = interactions_df[interactions_df["gene"] == symbol]
        for _, drug in gene_drugs.iterrows():
            rows.append({
                "gene_id": gene_id,
                "gene_symbol": symbol,
                "phenotypes": ", ".join(info["phenotypes"]),
                "n_phenotypes": len(info["phenotypes"]),
                "min_p": info["min_p"],
                "max_z": info["max_z"],
                "drug": drug["drug"],
                "interaction_type": ", ".join(drug["interaction_type"]) if isinstance(drug["interaction_type"], list) else str(drug["interaction_type"]),
                "drug_category": drug.get("drug_category", "other"),
                "sources": ", ".join(drug["sources"][:3]) if isinstance(drug["sources"], list) else str(drug["sources"]),
                "approved": drug.get("approved", False),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Drug Repurposing Analysis")
    print("=" * 60)

    gene_info = load_significant_genes()
    if not gene_info:
        print("\nNo Bonferroni-significant genes; using top genes as fallback.")
        top = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
        gene_info = {row["GeneID"]: {"gene_id": row["GeneID"], "phenotypes": [row["Phenotype"]], "min_p": row["P"], "max_z": row["Z"]} for _, row in top.iterrows()}
        symbols = dict(zip(top["GeneID"], top["Symbol"]))
    else:
        symbols = load_gene_annotations()
        top = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
        symbols.update(dict(zip(top["GeneID"], top["Symbol"])))

    unique_symbols: set[str] = set()
    for gene_id in gene_info:
        if gene_id in symbols:
            unique_symbols.add(symbols[gene_id])
    unique_symbols.update(top["Symbol"].tolist())

    interactions = query_dgidb(unique_symbols)
    if not interactions:
        print("\nNo drug-gene interactions found.")
        return

    df = categorize(pd.DataFrame(interactions))
    df.to_csv(OUT / "dgidb_interactions.csv", index=False)
    print(f"\nSaved {len(df)} interactions -> {OUT / 'dgidb_interactions.csv'}")

    candidates = make_candidate_table(df, gene_info, symbols).sort_values(["n_phenotypes", "min_p"], ascending=[False, True])
    candidates.to_csv(OUT / "repurposing_candidates.csv", index=False)
    print(f"Saved {len(candidates)} candidates -> {OUT / 'repurposing_candidates.csv'}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "10_drug_repurposing.md"
    log_path.write_text(
        "# Drug Repurposing Analysis\n\n"
        f"- Total drug-gene interactions: {len(df)}\n"
        f"- Unique drugs: {df['drug'].nunique()}\n"
        f"- Unique genes with hits: {df['gene'].nunique()}\n"
        f"- FDA-approved drugs: {df[df['approved']]['drug'].nunique()}\n"
        f"- Repurposing candidates: {len(candidates)}\n"
    )
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
