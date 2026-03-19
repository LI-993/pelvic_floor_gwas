#!/usr/bin/env python3
"""
Colocalization via Open Targets Genetics API (with SSL workaround).
"""

import requests
import csv
import time
import ssl
import urllib3
from pathlib import Path

# Disable SSL warnings for this session
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
OUT_DIR = BASE_DIR / "results" / "coloc_formal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_GENES = [
    'FGFR2', 'WT1', 'HNF1B', 'WNT4', 'SMAD3', 'ESR1', 'TNXB',
    'DNAH11', 'LOXL1', 'HOXA13', 'ELN', 'FBN1', 'COL1A1',
    'BCL11A', 'TGFBR2', 'COL3A1', 'FBLN5',
]

API_URL = "https://api.genetics.opentargets.org/graphql"


def query_gene_id(session, gene_symbol):
    """Look up Ensembl gene ID via Open Targets search."""
    query = """
    query GeneSearch($q: String!) {
      search(queryString: $q) {
        genes { id symbol }
      }
    }
    """
    try:
        resp = session.post(API_URL,
                            json={"query": query, "variables": {"q": gene_symbol}},
                            timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
        genes = data.get('data', {}).get('search', {}).get('genes', [])
        for g in genes:
            if g['symbol'] == gene_symbol:
                return g['id']
    except Exception as e:
        print(f"    Search error: {e}")
    return None


def query_coloc(session, gene_id):
    """Get colocalization results for a gene."""
    query = """
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
        resp = session.post(API_URL,
                            json={"query": query, "variables": {"geneId": gene_id}},
                            timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
        return data.get('data', {}).get('colocalisationsForGene', [])
    except Exception as e:
        print(f"    Coloc error: {e}")
        return []


def query_l2g(session, gene_id):
    """Get L2G (locus-to-gene) associations."""
    query = """
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
    try:
        resp = session.post(API_URL,
                            json={"query": query, "variables": {"geneId": gene_id}},
                            timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
        l2g = data.get('data', {}).get('studiesAndLeadVariantsForGeneByL2G', {})
        return l2g.get('rows', []) if l2g else []
    except Exception as e:
        print(f"    L2G error: {e}")
        return []


def main():
    print("=" * 60)
    print("Colocalization Analysis via Open Targets Genetics API")
    print("(with SSL workaround)")
    print("=" * 60)

    session = requests.Session()

    all_coloc = []
    all_l2g = []
    gene_summary = []

    for gene in TOP_GENES:
        print(f"\n[{gene}]", end=" ")

        gene_id = query_gene_id(session, gene)
        if not gene_id:
            print("Not found")
            continue
        print(f"({gene_id})")

        # L2G
        l2g_rows = query_l2g(session, gene_id)
        print(f"  L2G associations: {len(l2g_rows)}")

        # Filter eQTL-related L2G
        eqtl_l2g = [r for r in l2g_rows
                     if r.get('study', {}).get('source', '') in ('GTEx', 'eQTL Catalogue')]
        pfd_l2g = [r for r in l2g_rows
                    if any(kw in r.get('study', {}).get('traitReported', '').lower()
                           for kw in ['prolapse', 'incontinence', 'prostat', 'bladder',
                                      'constipation', 'urinary', 'pelvic'])]

        for row in l2g_rows:
            study = row.get('study', {})
            variant = row.get('variant', {})
            all_l2g.append({
                'gene': gene,
                'gene_id': gene_id,
                'study_id': study.get('studyId', ''),
                'trait': study.get('traitReported', ''),
                'source': study.get('source', ''),
                'variant': variant.get('rsId', '') if variant else '',
                'pval': row.get('pval', ''),
                'l2g_score': row.get('yProbaModel', ''),
            })

        # Coloc
        coloc_rows = query_coloc(session, gene_id)
        print(f"  Colocalizations: {len(coloc_rows)}")

        n_strong = 0
        n_moderate = 0
        best_h4 = 0
        best_coloc_trait = ''

        for cr in coloc_rows:
            h4 = cr.get('h4', 0) or 0
            h3 = cr.get('h3', 0) or 0
            ls = cr.get('leftStudy', {})
            rs = cr.get('rightStudy', {})
            lv = cr.get('leftVariant', {})
            rv = cr.get('rightVariant', {})

            all_coloc.append({
                'gene': gene,
                'gene_id': gene_id,
                'left_study': ls.get('studyId', ''),
                'left_trait': ls.get('traitReported', ''),
                'left_source': ls.get('source', ''),
                'right_study': rs.get('studyId', ''),
                'right_trait': rs.get('traitReported', ''),
                'right_source': rs.get('source', ''),
                'left_variant': lv.get('rsId', '') if lv else '',
                'right_variant': rv.get('rsId', '') if rv else '',
                'PP_H3': h3,
                'PP_H4': h4,
                'log2_H4_H3': cr.get('log2h4h3', ''),
            })

            if h4 > 0.8:
                n_strong += 1
            elif h4 > 0.5:
                n_moderate += 1

            if h4 > best_h4:
                best_h4 = h4
                # Determine which is eQTL vs GWAS
                if 'eqtl' in rs.get('source', '').lower() or 'gtex' in rs.get('source', '').lower():
                    best_coloc_trait = f"{ls.get('traitReported', '?')} <-> {rs.get('traitReported', '?')} ({rs.get('source', '')})"
                else:
                    best_coloc_trait = f"{ls.get('traitReported', '?')} <-> {rs.get('traitReported', '?')}"

        if n_strong > 0 or n_moderate > 0:
            print(f"  Strong (H4>0.8): {n_strong}, Moderate (H4>0.5): {n_moderate}")
            if best_h4 > 0.5:
                print(f"  Best: PP.H4={best_h4:.3f} -- {best_coloc_trait}")

        gene_summary.append({
            'gene': gene,
            'gene_id': gene_id,
            'n_l2g': len(l2g_rows),
            'n_coloc_total': len(coloc_rows),
            'n_coloc_strong': n_strong,
            'n_coloc_moderate': n_moderate,
            'best_h4': best_h4,
            'best_coloc': best_coloc_trait,
        })

        time.sleep(0.3)

    # Save results
    if all_l2g:
        f = OUT_DIR / "opentargets_l2g_results.csv"
        with open(f, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=all_l2g[0].keys())
            w.writeheader()
            w.writerows(all_l2g)
        print(f"\nSaved: {f} ({len(all_l2g)} rows)")

    if all_coloc:
        f = OUT_DIR / "opentargets_coloc_results.csv"
        with open(f, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=all_coloc[0].keys())
            w.writeheader()
            w.writerows(all_coloc)
        print(f"Saved: {f} ({len(all_coloc)} rows)")

    if gene_summary:
        f = OUT_DIR / "coloc_gene_summary.csv"
        with open(f, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=gene_summary[0].keys())
            w.writeheader()
            w.writerows(gene_summary)
        print(f"Saved: {f}")

    # Print summary table
    print("\n\n=== Colocalization Summary ===")
    print(f"{'Gene':<10} {'L2G':>4} {'Coloc':>6} {'H4>0.8':>7} {'H4>0.5':>7} {'Best H4':>8}")
    print("-" * 50)
    for gs in gene_summary:
        print(f"{gs['gene']:<10} {gs['n_l2g']:>4} {gs['n_coloc_total']:>6} "
              f"{gs['n_coloc_strong']:>7} {gs['n_coloc_moderate']:>7} "
              f"{gs['best_h4']:>8.3f}")

    total_strong = sum(gs['n_coloc_strong'] for gs in gene_summary)
    total_moderate = sum(gs['n_coloc_moderate'] for gs in gene_summary)
    print(f"\nTotal strong colocalizations (PP.H4 > 0.8): {total_strong}")
    print(f"Total moderate colocalizations (PP.H4 > 0.5): {total_moderate}")


if __name__ == '__main__':
    main()
