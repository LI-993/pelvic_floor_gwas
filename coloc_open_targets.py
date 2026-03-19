#!/usr/bin/env python3
"""
Colocalization analysis using Open Targets Genetics API.
Retrieves pre-computed coloc results (PP.H4) for top prioritized genes.
"""

import json
import urllib.request
import urllib.error
import csv
import time
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
OUT_DIR = BASE_DIR / "results" / "coloc_formal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Top prioritized genes from ensemble ranking
TOP_GENES = [
    'FGFR2', 'WT1', 'HNF1B', 'WNT4', 'SMAD3', 'ESR1', 'TNXB',
    'DNAH11', 'LOXL1', 'HOXA13', 'ELN', 'FBN1', 'COL1A1',
    'BCL11A', 'TGFBR2', 'COL3A1', 'FBLN5',
]

# Relevant PFD-related EFO trait IDs from Open Targets
PFD_EFO_TRAITS = {
    'EFO_0009553': 'pelvic organ prolapse',
    'EFO_0003105': 'urinary incontinence',
    'EFO_0004232': 'benign prostatic hyperplasia',
    'EFO_0005537': 'bladder disease',
    'EFO_0003912': 'constipation',
}

# Relevant tissue IDs for eQTL sources
RELEVANT_TISSUES = [
    'Uterus', 'Vagina', 'Prostate', 'Bladder',
    'Colon_Sigmoid', 'Colon_Transverse',
    'Muscle_Skeletal', 'Skin_Sun_Exposed_Lower_leg',
    'Adipose_Subcutaneous', 'Whole_Blood',
]


def query_open_targets_genetics(gene_symbol):
    """Query Open Targets Genetics GraphQL API for colocalization data."""
    # Open Targets Genetics GraphQL endpoint
    url = "https://api.genetics.opentargets.org/graphql"

    # Query for gene's associated studies and colocalizations
    query = """
    query GeneColoc($geneId: String!) {
      geneInfo(geneId: $geneId) {
        id
        symbol
        chromosome
        start
        end
      }
      studiesAndLeadVariantsForGeneByL2G(geneId: $geneId, pageSize: 50) {
        rows {
          study {
            studyId
            traitReported
            traitEfos
            source
            pmid
          }
          variant {
            id
            rsId
          }
          pval
          yProbaModel
        }
      }
    }
    """

    # First we need the Ensembl ID for the gene
    # Try looking up via gene symbol
    gene_lookup_query = """
    query GeneSearch($queryString: String!) {
      search(queryString: $queryString) {
        genes {
          id
          symbol
        }
      }
    }
    """

    try:
        # Step 1: Look up gene
        data = json.dumps({
            "query": gene_lookup_query,
            "variables": {"queryString": gene_symbol}
        }).encode('utf-8')

        req = urllib.request.Request(url, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())

        genes = result.get('data', {}).get('search', {}).get('genes', [])
        gene_id = None
        for g in genes:
            if g['symbol'] == gene_symbol:
                gene_id = g['id']
                break

        if not gene_id:
            return None, f"Gene {gene_symbol} not found"

        # Step 2: Get coloc data
        data = json.dumps({
            "query": query,
            "variables": {"geneId": gene_id}
        }).encode('utf-8')

        req = urllib.request.Request(url, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())

        return result, None

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def query_coloc_for_variant(variant_id, gene_id):
    """Query colocalization results for a specific variant-gene pair."""
    url = "https://api.genetics.opentargets.org/graphql"

    query = """
    query ColocForVariant($variantId: String!) {
      colocalisationsForGene(geneId: $variantId) {
        leftVariant { id rsId }
        leftStudy { studyId traitReported source }
        rightVariant { id rsId }
        rightStudy { studyId traitReported source }
        h3
        h4
        log2h4h3
      }
    }
    """

    # Actually use the genes endpoint for coloc
    query2 = """
    query GeneColoc($geneId: String!) {
      colocalisationsForGene(geneId: $geneId) {
        leftVariant { id rsId }
        leftStudy { studyId traitReported source }
        rightVariant { id rsId }
        rightStudy { studyId traitReported source }
        h3
        h4
        log2h4h3
      }
    }
    """

    try:
        data = json.dumps({
            "query": query2,
            "variables": {"geneId": gene_id}
        }).encode('utf-8')

        req = urllib.request.Request(url, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())

        return result, None

    except Exception as e:
        return None, str(e)


def main():
    print("=" * 60)
    print("Colocalization Analysis via Open Targets Genetics API")
    print("=" * 60)

    all_results = []
    coloc_results = []

    for gene in TOP_GENES:
        print(f"\n[{gene}] Querying Open Targets...")
        result, error = query_open_targets_genetics(gene)

        if error:
            print(f"  Error: {error}")
            continue

        gene_info = result.get('data', {}).get('geneInfo', {})
        l2g_data = result.get('data', {}).get('studiesAndLeadVariantsForGeneByL2G', {})
        rows = l2g_data.get('rows', []) if l2g_data else []

        gene_id = gene_info.get('id', '') if gene_info else ''
        print(f"  Gene ID: {gene_id}")
        print(f"  L2G associations: {len(rows)}")

        # Filter for relevant studies
        for row in rows:
            study = row.get('study', {})
            variant = row.get('variant', {})
            l2g_score = row.get('yProbaModel', 0)
            pval = row.get('pval', 1)

            trait = study.get('traitReported', '')
            source = study.get('source', '')
            study_id = study.get('studyId', '')
            rs_id = variant.get('rsId', '') if variant else ''
            var_id = variant.get('id', '') if variant else ''

            all_results.append({
                'gene': gene,
                'gene_id': gene_id,
                'study_id': study_id,
                'trait': trait,
                'source': source,
                'variant': rs_id,
                'variant_id': var_id,
                'pval': pval,
                'l2g_score': l2g_score,
            })

        # Get coloc data for this gene
        if gene_id:
            coloc_data, coloc_error = query_coloc_for_variant("", gene_id)
            if coloc_error:
                print(f"  Coloc query error: {coloc_error}")
            else:
                coloc_rows = coloc_data.get('data', {}).get('colocalisationsForGene', [])
                if coloc_rows:
                    print(f"  Colocalization results: {len(coloc_rows)}")
                    for cr in coloc_rows:
                        h4 = cr.get('h4', 0)
                        h3 = cr.get('h3', 0)
                        log2h4h3 = cr.get('log2h4h3', 0)
                        left_study = cr.get('leftStudy', {})
                        right_study = cr.get('rightStudy', {})
                        left_var = cr.get('leftVariant', {})
                        right_var = cr.get('rightVariant', {})

                        coloc_results.append({
                            'gene': gene,
                            'gene_id': gene_id,
                            'left_study': left_study.get('studyId', ''),
                            'left_trait': left_study.get('traitReported', ''),
                            'left_source': left_study.get('source', ''),
                            'right_study': right_study.get('studyId', ''),
                            'right_trait': right_study.get('traitReported', ''),
                            'right_source': right_study.get('source', ''),
                            'left_variant': left_var.get('rsId', '') if left_var else '',
                            'right_variant': right_var.get('rsId', '') if right_var else '',
                            'PP_H3': h3,
                            'PP_H4': h4,
                            'log2_H4_H3': log2h4h3,
                        })

                        # Print notable ones
                        if h4 and h4 > 0.5:
                            print(f"    PP.H4={h4:.3f}: {left_study.get('traitReported', '?')}"
                                  f" <-> {right_study.get('traitReported', '?')}"
                                  f" ({right_study.get('source', '?')})")
                else:
                    print(f"  No colocalization results found")

        time.sleep(0.5)  # Rate limiting

    # Save L2G results
    if all_results:
        outfile = OUT_DIR / "opentargets_l2g_results.csv"
        with open(outfile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n\nSaved L2G results: {outfile} ({len(all_results)} rows)")

    # Save coloc results
    if coloc_results:
        outfile = OUT_DIR / "opentargets_coloc_results.csv"
        with open(outfile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=coloc_results[0].keys())
            writer.writeheader()
            writer.writerows(coloc_results)
        print(f"Saved coloc results: {outfile} ({len(coloc_results)} rows)")

        # Summary
        print("\n\n=== Colocalization Summary ===")
        print(f"Total coloc results: {len(coloc_results)}")
        strong_coloc = [r for r in coloc_results if r['PP_H4'] and r['PP_H4'] > 0.8]
        print(f"Strong coloc (PP.H4 > 0.8): {len(strong_coloc)}")
        moderate_coloc = [r for r in coloc_results if r['PP_H4'] and 0.5 < r['PP_H4'] <= 0.8]
        print(f"Moderate coloc (0.5 < PP.H4 ≤ 0.8): {len(moderate_coloc)}")

        if strong_coloc:
            print("\nStrong colocalizations (PP.H4 > 0.8):")
            for r in strong_coloc:
                eqtl_trait = r['right_trait'] if 'eqtl' in r['right_source'].lower() or 'gtex' in r['right_source'].lower() else r['left_trait']
                gwas_trait = r['left_trait'] if eqtl_trait == r['right_trait'] else r['right_trait']
                print(f"  {r['gene']}: {gwas_trait} <-> {eqtl_trait} (PP.H4={r['PP_H4']:.3f})")
    else:
        print("\n\nNo colocalization results retrieved. This may be due to API limitations.")
        print("Consider downloading Open Targets Genetics bulk data for comprehensive coloc analysis.")

    # Save summary for manuscript
    summary_file = OUT_DIR / "coloc_summary_for_manuscript.txt"
    with open(summary_file, 'w') as f:
        f.write("Colocalization Summary for Manuscript\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Genes queried: {len(TOP_GENES)}\n")
        f.write(f"Total L2G associations: {len(all_results)}\n")
        f.write(f"Total coloc results: {len(coloc_results)}\n")
        if coloc_results:
            f.write(f"Strong coloc (PP.H4 > 0.8): {len([r for r in coloc_results if r['PP_H4'] and r['PP_H4'] > 0.8])}\n")
            f.write(f"Moderate coloc (PP.H4 > 0.5): {len([r for r in coloc_results if r['PP_H4'] and r['PP_H4'] > 0.5])}\n")
    print(f"\nSaved summary: {summary_file}")


if __name__ == '__main__':
    main()
