#!/usr/bin/env python3
"""
Phase 7: Drug Repurposing Analysis

Query DGIdb (Drug-Gene Interaction Database) for potential drug-gene interactions
based on significant genes from MAGMA analysis.

DGIdb API: https://dgidb.org/api
"""

import json
import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path
import time

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
MAGMA_DIR = BASE_DIR / "results/magma"
RESULTS_DIR = BASE_DIR / "results/drug_repurposing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DGIdb GraphQL API endpoint (v5.0)
DGIDB_GRAPHQL = "https://dgidb.org/api/graphql"

# Significance threshold for genes
P_THRESHOLD = 0.05 / 19000  # Bonferroni correction (~2.6e-6)


def load_significant_genes():
    """Load Bonferroni-significant genes from MAGMA results."""
    print("Loading significant genes from MAGMA results...")

    all_genes = {}
    phenotypes = ["POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence"]

    for pheno in phenotypes:
        results_file = MAGMA_DIR / f"{pheno}_genes.genes.out.txt"
        if results_file.exists():
            df = pd.read_csv(results_file, sep=r'\s+')
            # Filter significant genes
            sig_genes = df[df['P'] < P_THRESHOLD].copy()
            sig_genes['Phenotype'] = pheno

            for _, row in sig_genes.iterrows():
                gene_id = row['GENE']
                if gene_id not in all_genes:
                    all_genes[gene_id] = {
                        'gene_id': gene_id,
                        'phenotypes': [],
                        'min_p': row['P'],
                        'max_z': row['ZSTAT']
                    }
                all_genes[gene_id]['phenotypes'].append(pheno)
                if row['P'] < all_genes[gene_id]['min_p']:
                    all_genes[gene_id]['min_p'] = row['P']
                if row['ZSTAT'] > all_genes[gene_id]['max_z']:
                    all_genes[gene_id]['max_z'] = row['ZSTAT']

    print(f"Found {len(all_genes)} unique Bonferroni-significant genes")
    return all_genes


def load_gene_symbols():
    """Load gene ID to symbol mapping from MAGMA results."""
    top_genes = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
    id_to_symbol = dict(zip(top_genes['GeneID'], top_genes['Symbol']))
    return id_to_symbol


def load_all_gene_annotations():
    """Load NCBI gene annotations for symbol lookup."""
    gene_loc_file = BASE_DIR / "reference/magma/NCBI37.3.gene.loc"
    if gene_loc_file.exists():
        df = pd.read_csv(gene_loc_file, sep='\t', header=None,
                        names=['gene_id', 'chr', 'start', 'end', 'strand', 'symbol'])
        return dict(zip(df['gene_id'], df['symbol']))
    return {}


def query_dgidb_graphql(gene_symbols, batch_size=50):
    """Query DGIdb v5.0 GraphQL API for drug-gene interactions."""
    print(f"\nQuerying DGIdb GraphQL API for {len(gene_symbols)} genes...")

    all_interactions = []

    # GraphQL query template
    query_template = """
    query {
      genes(names: %s) {
        nodes {
          name
          longName
          interactions {
            drug {
              name
              conceptId
              approved
            }
            interactionScore
            interactionTypes {
              type
              directionality
            }
            interactionAttributes {
              name
              value
            }
            sources {
              fullName
            }
            publications {
              pmid
            }
          }
        }
      }
    }
    """

    # Process in batches
    symbols_list = list(gene_symbols)
    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} genes")

        # Format gene list for GraphQL
        genes_str = json.dumps(batch)

        query = query_template % genes_str

        try:
            data = json.dumps({"query": query}).encode('utf-8')
            req = urllib.request.Request(
                DGIDB_GRAPHQL,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0'
                }
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())

                if 'data' in result and 'genes' in result['data']:
                    genes_data = result['data']['genes']['nodes']
                    for gene_node in genes_data:
                        gene_name = gene_node.get('name', '')
                        for interaction in gene_node.get('interactions', []):
                            drug_info = interaction.get('drug', {})
                            interaction_types = [it.get('type', '') for it in interaction.get('interactionTypes', [])]
                            sources = [s.get('fullName', '') for s in interaction.get('sources', [])]
                            pmids = [p.get('pmid', '') for p in interaction.get('publications', [])]

                            all_interactions.append({
                                'gene': gene_name,
                                'drug': drug_info.get('name', ''),
                                'interaction_type': interaction_types,
                                'sources': sources,
                                'pmid': pmids,
                                'drug_concept_id': drug_info.get('conceptId', ''),
                                'approved': drug_info.get('approved', False),
                                'score': interaction.get('interactionScore', 0)
                            })

                    matched_count = len([g for g in genes_data if g.get('interactions')])
                    print(f"    Matched genes with interactions: {matched_count}")

                elif 'errors' in result:
                    print(f"    GraphQL errors: {result['errors']}")

        except Exception as e:
            print(f"    Error: {e}")

        time.sleep(0.3)  # Rate limiting

    print(f"Found {len(all_interactions)} drug-gene interactions")
    return all_interactions


def categorize_drugs(interactions_df):
    """Categorize drugs by interaction type and therapeutic area."""

    # Drug categories for pelvic floor relevance
    pelvic_keywords = {
        'alpha_blockers': ['tamsulosin', 'alfuzosin', 'doxazosin', 'terazosin', 'prazosin', 'silodosin'],
        '5ari': ['finasteride', 'dutasteride'],
        'anticholinergics': ['oxybutynin', 'tolterodine', 'solifenacin', 'darifenacin', 'fesoterodine', 'trospium'],
        'beta3_agonists': ['mirabegron', 'vibegron'],
        'pde5_inhibitors': ['sildenafil', 'tadalafil', 'vardenafil', 'avanafil'],
        'hormones': ['estrogen', 'estradiol', 'testosterone', 'progesterone'],
        'collagen_modulators': ['penicillamine', 'collagenase'],
        'anti_inflammatories': ['ibuprofen', 'naproxen', 'celecoxib', 'diclofenac'],
        'muscle_relaxants': ['baclofen', 'diazepam', 'cyclobenzaprine', 'tizanidine']
    }

    def get_drug_category(drug_name):
        drug_lower = drug_name.lower()
        for category, keywords in pelvic_keywords.items():
            for kw in keywords:
                if kw in drug_lower:
                    return category
        return 'other'

    interactions_df['drug_category'] = interactions_df['drug'].apply(get_drug_category)
    return interactions_df


def identify_repurposing_candidates(interactions_df, gene_info, gene_symbols):
    """Identify promising drug repurposing candidates."""

    candidates = []

    for gene, info in gene_info.items():
        symbol = gene_symbols.get(gene, str(gene))
        gene_drugs = interactions_df[interactions_df['gene'] == symbol]

        if len(gene_drugs) > 0:
            for _, drug in gene_drugs.iterrows():
                candidates.append({
                    'gene_id': gene,
                    'gene_symbol': symbol,
                    'phenotypes': ', '.join(info['phenotypes']),
                    'n_phenotypes': len(info['phenotypes']),
                    'min_p': info['min_p'],
                    'max_z': info['max_z'],
                    'drug': drug['drug'],
                    'interaction_type': ', '.join(drug['interaction_type']) if isinstance(drug['interaction_type'], list) else str(drug['interaction_type']),
                    'drug_category': drug.get('drug_category', 'other'),
                    'sources': ', '.join(drug['sources'][:3]) if isinstance(drug['sources'], list) else str(drug['sources']),
                    'chembl_id': drug.get('drug_chembl_id', '')
                })

    return pd.DataFrame(candidates)


def create_summary_report(candidates_df, interactions_df):
    """Create summary report of drug repurposing analysis."""

    report = []
    report.append("# Drug Repurposing Analysis Summary")
    report.append(f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    report.append("**Phase**: 7 - Drug Repurposing")
    report.append("\n## Overview")
    report.append(f"\n- Total drug-gene interactions found: {len(interactions_df)}")
    report.append(f"- Unique genes with interactions: {interactions_df['gene'].nunique()}")
    report.append(f"- Unique drugs identified: {interactions_df['drug'].nunique()}")
    report.append(f"- Drug repurposing candidates: {len(candidates_df)}")

    if len(candidates_df) > 0:
        report.append("\n## Top Drug-Gene Interactions by Phenotype")

        # Group by phenotype
        for pheno in ['BPH', 'POP', 'FemaleProlapse', 'Incontinence', 'Constipation', 'Bladder']:
            pheno_drugs = candidates_df[candidates_df['phenotypes'].str.contains(pheno)]
            if len(pheno_drugs) > 0:
                report.append(f"\n### {pheno}")
                top = pheno_drugs.nsmallest(10, 'min_p')
                for _, row in top.iterrows():
                    report.append(f"- **{row['gene_symbol']}** → {row['drug']} ({row['interaction_type']})")

        report.append("\n## Drugs by Category")
        if 'drug_category' in candidates_df.columns:
            category_counts = candidates_df['drug_category'].value_counts()
            for cat, count in category_counts.items():
                report.append(f"- {cat}: {count} interactions")

        report.append("\n## Priority Candidates (Multi-phenotype genes)")
        multi = candidates_df[candidates_df['n_phenotypes'] > 1].sort_values('n_phenotypes', ascending=False)
        if len(multi) > 0:
            for _, row in multi.head(20).iterrows():
                report.append(f"- **{row['gene_symbol']}** ({row['phenotypes']}): {row['drug']}")
        else:
            report.append("No genes significant in multiple phenotypes have drug interactions.")

    report.append("\n## Clinical Relevance")
    report.append("""
### Currently Used Drugs for Pelvic Floor Disorders

| Drug Class | Examples | Target Phenotype |
|------------|----------|------------------|
| Alpha-blockers | Tamsulosin, Alfuzosin | BPH |
| 5-alpha reductase inhibitors | Finasteride, Dutasteride | BPH |
| Anticholinergics | Oxybutynin, Solifenacin | OAB/Incontinence |
| Beta-3 agonists | Mirabegron | OAB |
| PDE5 inhibitors | Tadalafil | BPH (combination) |
| Estrogen (topical) | Estradiol | POP/Incontinence |
""")

    report.append("\n## Output Files")
    report.append(f"- `results/drug_repurposing/dgidb_interactions.csv` - All drug-gene interactions")
    report.append(f"- `results/drug_repurposing/repurposing_candidates.csv` - Prioritized candidates")

    return "\n".join(report)


def main():
    print("="*60)
    print("Phase 7: Drug Repurposing Analysis")
    print("="*60)

    # Load significant genes
    gene_info = load_significant_genes()

    if len(gene_info) == 0:
        print("No Bonferroni-significant genes found. Using top genes instead.")
        # Load top genes as fallback
        top_genes_df = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
        gene_symbols_map = dict(zip(top_genes_df['GeneID'], top_genes_df['Symbol']))
        unique_symbols = set(top_genes_df['Symbol'].tolist())
    else:
        # Load gene symbol mapping
        gene_symbols_map = load_all_gene_annotations()

        # Also include top genes
        top_genes_df = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
        for _, row in top_genes_df.iterrows():
            gene_symbols_map[row['GeneID']] = row['Symbol']

        unique_symbols = set()
        for gene_id in gene_info.keys():
            if gene_id in gene_symbols_map:
                unique_symbols.add(gene_symbols_map[gene_id])

        # Add all top genes
        unique_symbols.update(top_genes_df['Symbol'].tolist())

    print(f"Total unique gene symbols to query: {len(unique_symbols)}")

    # Query DGIdb GraphQL API
    interactions = query_dgidb_graphql(unique_symbols)

    if len(interactions) > 0:
        # Create DataFrame
        interactions_df = pd.DataFrame(interactions)

        # Categorize drugs
        interactions_df = categorize_drugs(interactions_df)

        # Save raw interactions
        interactions_df.to_csv(RESULTS_DIR / "dgidb_interactions.csv", index=False)
        print(f"\nSaved {len(interactions_df)} interactions to dgidb_interactions.csv")

        # Identify repurposing candidates
        candidates_df = identify_repurposing_candidates(interactions_df, gene_info, gene_symbols_map)

        if len(candidates_df) > 0:
            candidates_df = candidates_df.sort_values(['n_phenotypes', 'min_p'], ascending=[False, True])
            candidates_df.to_csv(RESULTS_DIR / "repurposing_candidates.csv", index=False)
            print(f"Saved {len(candidates_df)} candidates to repurposing_candidates.csv")

        # Create summary report
        report = create_summary_report(candidates_df, interactions_df)
        report_path = BASE_DIR / "logs/10_drug_repurposing.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        # Print top findings
        print("\n" + "="*60)
        print("TOP DRUG-GENE INTERACTIONS")
        print("="*60)

        # Group by drug category
        print("\n[Pelvic Floor Relevant Drugs]")
        relevant = interactions_df[interactions_df['drug_category'] != 'other']
        if len(relevant) > 0:
            for _, row in relevant.head(15).iterrows():
                print(f"  {row['gene']}: {row['drug']} ({row['drug_category']})")
        else:
            print("  No currently-used pelvic floor drugs found in interactions.")

        print("\n[Novel Repurposing Candidates]")
        novel = interactions_df[interactions_df['drug_category'] == 'other']
        if len(novel) > 0:
            # Group by gene
            gene_counts = novel['gene'].value_counts().head(10)
            for gene, count in gene_counts.items():
                gene_drugs = novel[novel['gene'] == gene]['drug'].head(3).tolist()
                print(f"  {gene} ({count} drugs): {', '.join(gene_drugs)}")
    else:
        print("\nNo drug-gene interactions found.")

    print("\n" + "="*60)
    print("Drug Repurposing Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
