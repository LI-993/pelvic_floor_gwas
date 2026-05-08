#!/usr/bin/env python3
"""STRING-based PPI network on MAGMA-significant genes.

Pulls interactions from the STRING REST API, builds a NetworkX graph,
computes centralities, runs Louvain community detection, and submits the
gene list back to STRING for pathway enrichment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPES, RESULTS_DIR
from utils.gene_mapping import get_symbol, load_ncbi_gene_mapping
from utils.plotting import setup_publication_style

OUT_RESULTS = RESULTS_DIR / "ppi_network"
OUT_FIGURES = FIGURES_DIR / "ppi_network"
MAGMA_DIR = RESULTS_DIR / "magma"

STRING_API = "https://string-db.org/api"
STRING_SPECIES = 9606
STRING_BATCH = 100
STRING_SCORE_MIN = 400


def load_magma_genes(p_threshold: float = 1e-3, top_n: int = 50) -> pd.DataFrame:
    """Aggregate top MAGMA genes per phenotype, keeping the strongest signal per gene."""
    frames = []
    for pheno in PHENOTYPES:
        path = MAGMA_DIR / f"{pheno}_genes.genes.out.txt"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=r"\s+", comment="#")
        df["Phenotype"] = pheno
        sig = df[df["P"] < p_threshold].head(top_n)
        frames.append(sig)
        print(f"  {pheno}: {len(sig)} genes (P<{p_threshold})")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby("GENE")
        .agg(min_P=("P", "min"), max_Z=("ZSTAT", "max"), n_phenotypes=("Phenotype", "nunique"), nSNPs=("NSNPS", "max"))
        .reset_index()
        .rename(columns={"GENE": "GeneID"})
    )
    print(f"  Unique genes: {len(summary)}")
    return summary


def gene_symbols(gene_ids: list[str], entrez_to_symbol: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    mapped = unmapped = 0
    for gid in gene_ids:
        gid = str(gid).strip()
        symbol = get_symbol(gid, entrez_to_symbol)
        if symbol == gid:
            unmapped += 1
            out[gid] = f"GENE_{gid}"
        else:
            mapped += 1
            out[gid] = symbol
    print(f"  Symbol mapping: {mapped} mapped, {unmapped} unmapped")
    return out


def fetch_string_interactions(genes: list[str]) -> list[dict]:
    if not genes:
        return []
    interactions: list[dict] = []
    for i in range(0, len(genes), STRING_BATCH):
        batch = genes[i : i + STRING_BATCH]
        params = {
            "identifiers": "%0d".join(batch),
            "species": STRING_SPECIES,
            "caller_identity": "pelvic_floor_gwas",
            "required_score": STRING_SCORE_MIN,
        }
        try:
            r = requests.get(f"{STRING_API}/json/network", params=params, timeout=30)
            if r.status_code == 200:
                got = r.json()
                interactions.extend(got)
                print(f"  Batch {i // STRING_BATCH + 1}: {len(got)} interactions")
            else:
                print(f"  STRING returned {r.status_code}")
        except Exception as e:  # noqa: BLE001
            print(f"  STRING error: {e}")
        time.sleep(0.5)
    return interactions


def build_network(interactions: list[dict], gene_data: pd.DataFrame, symbols: dict[str, str]) -> nx.Graph:
    G = nx.Graph()
    node_symbols: set[str] = set()
    for _, row in gene_data.iterrows():
        gid = str(int(row["GeneID"])) if pd.notna(row["GeneID"]) else str(row["GeneID"])
        symbol = symbols.get(gid, gid)
        G.add_node(symbol, gene_id=gid, p_value=row["min_P"], z_score=row["max_Z"], n_phenotypes=row["n_phenotypes"])
        node_symbols.add(symbol)

    edges = 0
    for ia in interactions:
        a = ia.get("preferredName_A") or ia.get("stringId_A", "")
        b = ia.get("preferredName_B") or ia.get("stringId_B", "")
        if a in node_symbols and b in node_symbols:
            G.add_edge(a, b, weight=ia.get("score", 0))
            edges += 1

    print(f"  Network: {G.number_of_nodes()} nodes, {edges} edges (from {len(interactions)} interactions)")
    return G


def network_metrics(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    deg_c = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    try:
        clo = nx.closeness_centrality(G)
    except Exception:  # noqa: BLE001
        clo = {n: 0 for n in G.nodes()}
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:  # noqa: BLE001
        eig = {n: 0 for n in G.nodes()}
    clu = nx.clustering(G)

    rows = []
    for n in G.nodes():
        rows.append({
            "Gene": n,
            "Degree": G.degree(n),
            "Degree_Centrality": deg_c.get(n, 0),
            "Betweenness": btw.get(n, 0),
            "Closeness": clo.get(n, 0),
            "Eigenvector": eig.get(n, 0),
            "Clustering": clu.get(n, 0),
            "P_value": G.nodes[n].get("p_value", 1),
            "Z_score": G.nodes[n].get("z_score", 0),
            "N_phenotypes": G.nodes[n].get("n_phenotypes", 0),
        })
    df = pd.DataFrame(rows).sort_values("Degree", ascending=False)
    print("  Top hubs:", ", ".join(df.head(5)["Gene"].tolist()))
    return df


def detect_communities(G: nx.Graph) -> dict[str, int]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, seed=42)
    except Exception:  # noqa: BLE001
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G)
    out = {n: i for i, c in enumerate(communities) for n in c}
    print(f"  Communities: {len(set(out.values()))}")
    return out


def pathway_enrichment(genes: list[str]) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame()
    params = {
        "identifiers": "%0d".join(genes[:200]),
        "species": STRING_SPECIES,
        "caller_identity": "pelvic_floor_gwas",
    }
    try:
        r = requests.get(f"{STRING_API}/json/enrichment", params=params, timeout=60)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if "fdr" in df.columns:
                df = df[df["fdr"] < 0.05]
            print(f"  Enriched pathways (FDR<0.05): {len(df)}")
            return df
    except Exception as e:  # noqa: BLE001
        print(f"  Enrichment error: {e}")
    return pd.DataFrame()


def plot_summary(G: nx.Graph, metrics: pd.DataFrame, communities: dict[str, int], enrichment: pd.DataFrame) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    if G.number_of_nodes() == 0:
        return

    fig = plt.figure(figsize=(16, 12))

    # 1. Network
    ax1 = fig.add_subplot(2, 2, 1)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_colors = [communities.get(n, 0) for n in G.nodes()] if communities else [G.nodes[n].get("n_phenotypes", 0) for n in G.nodes()]
    node_sizes = [max(G.degree(n) * 100, 50) for n in G.nodes()]
    nx.draw_networkx(G, pos, ax=ax1, node_color=node_colors, cmap="Set3", node_size=node_sizes,
                     font_size=8, font_weight="bold", edge_color="gray", alpha=0.7, with_labels=True)
    ax1.set_title("PPI Network\n(node size = degree, color = community)", fontweight="bold")
    ax1.axis("off")

    # 2. Hub bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    if not metrics.empty:
        top_hubs = metrics.head(15)
        colors = ["#E64B35" if r["N_phenotypes"] > 1 else "#4DBBD5" for _, r in top_hubs.iterrows()]
        ax2.barh(np.arange(len(top_hubs)), top_hubs["Degree"], color=colors, alpha=0.8)
        ax2.set_yticks(np.arange(len(top_hubs)))
        ax2.set_yticklabels(top_hubs["Gene"])
        ax2.invert_yaxis()
        ax2.set_xlabel("Degree")
        ax2.set_title("Hub Genes\n(red = multi-phenotype)", fontweight="bold")

    # 3. Centrality heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    if not metrics.empty:
        m = metrics.head(15)[["Gene", "Degree_Centrality", "Betweenness", "Eigenvector", "Clustering"]].set_index("Gene")
        sns.heatmap(m, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax3, cbar_kws={"label": "Score"})
        ax3.set_title("Network Centralities", fontweight="bold")
        ax3.set_xlabel("")

    # 4. Pathway enrichment
    ax4 = fig.add_subplot(2, 2, 4)
    if len(enrichment) and "fdr" in enrichment.columns:
        top = enrichment.nsmallest(10, "fdr")
        labels = (top["term"] if "term" in top.columns else top["description"]).str[:40]
        ax4.barh(np.arange(len(top)), -np.log10(top["fdr"].clip(lower=1e-16)), color="#00A087", alpha=0.8)
        ax4.set_yticks(np.arange(len(top)))
        ax4.set_yticklabels(labels)
        ax4.invert_yaxis()
        ax4.set_xlabel("-log10(FDR)")
        ax4.set_title("Top Enriched Pathways", fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No enrichment", ha="center", va="center", color="gray")
        ax4.axis("off")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"ppi_network.{ext}", bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_publication_style()
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Expanded PPI Network Analysis")
    print("=" * 60)

    print("\n[0] NCBI gene mapping...")
    entrez_to_symbol, _, _ = load_ncbi_gene_mapping()

    print("\n[1] MAGMA genes...")
    gene_data = load_magma_genes(p_threshold=1e-3, top_n=50)
    if gene_data.empty:
        print("Error: no MAGMA results")
        return

    print("\n[2] Symbol mapping...")
    symbols = gene_symbols(gene_data["GeneID"].astype(str).tolist(), entrez_to_symbol)
    known = sorted({s for s in symbols.values() if not s.startswith("GENE_")})
    print(f"  Querying STRING for {len(known)} known symbols")

    print("\n[3] STRING interactions...")
    interactions = fetch_string_interactions(known)

    print("\n[4] Build network...")
    G = build_network(interactions, gene_data, symbols)

    print("\n[5] Centralities...")
    metrics = network_metrics(G)

    print("\n[6] Communities...")
    communities = detect_communities(G)

    print("\n[7] Pathway enrichment...")
    enrichment = pathway_enrichment(known)

    print("\n[8] Saving outputs...")
    metrics.to_csv(OUT_RESULTS / "network_metrics.csv", index=False)
    if communities:
        pd.DataFrame([{"Gene": g, "Community": c} for g, c in communities.items()]).to_csv(
            OUT_RESULTS / "community_assignments.csv", index=False
        )
    if not enrichment.empty:
        enrichment.to_csv(OUT_RESULTS / "pathway_enrichment.csv", index=False)
    nx.write_graphml(G, str(OUT_RESULTS / "ppi_network.graphml"))

    plot_summary(G, metrics, communities, enrichment)
    print(f"\nResults: {OUT_RESULTS}\nFigures: {OUT_FIGURES}")


if __name__ == "__main__":
    main()
