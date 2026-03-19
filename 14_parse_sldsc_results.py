#!/usr/bin/env python3
"""Parse S-LDSC log files and create summary tables."""

import re
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas/results/sldsc")
OUTPUT_DIR = RESULTS_DIR

PHENOTYPES = ["POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence"]

def parse_log(log_path):
    """Parse S-LDSC log file to extract results."""
    with open(log_path, 'r') as f:
        content = f.read()

    results = {}

    # Extract total h2
    h2_match = re.search(r'Total Observed scale h2: ([\d.]+) \(([\d.]+)\)', content)
    if h2_match:
        results['h2'] = float(h2_match.group(1))
        results['h2_se'] = float(h2_match.group(2))

    # Extract Lambda GC
    lambda_match = re.search(r'Lambda GC: ([\d.]+)', content)
    if lambda_match:
        results['lambda_gc'] = float(lambda_match.group(1))

    # Extract Mean Chi^2
    chi2_match = re.search(r'Mean Chi\^2: ([\d.]+)', content)
    if chi2_match:
        results['mean_chi2'] = float(chi2_match.group(1))

    # Extract Intercept
    intercept_match = re.search(r'Intercept: ([\d.]+) \(([\d.]+)\)', content)
    if intercept_match:
        results['intercept'] = float(intercept_match.group(1))
        results['intercept_se'] = float(intercept_match.group(2))

    # Extract categories and enrichment
    cat_match = re.search(r'Categories: (.+)', content)
    if cat_match:
        categories = cat_match.group(1).split()
        results['categories'] = categories

    enrich_match = re.search(r'Enrichment: (.+)', content)
    if enrich_match:
        enrichment_str = enrich_match.group(1).strip()
        enrichment = [float(x) for x in enrichment_str.split()]
        results['enrichment'] = enrichment

    return results


def main():
    print("="*60)
    print("S-LDSC Results Summary")
    print("="*60)

    # Parse all logs
    all_results = {}
    for pheno in PHENOTYPES:
        log_path = RESULTS_DIR / f"{pheno}_baselineLD.log"
        if log_path.exists():
            all_results[pheno] = parse_log(log_path)
            print(f"\n{pheno}:")
            print(f"  h2 = {all_results[pheno].get('h2', 'N/A'):.4f} (SE: {all_results[pheno].get('h2_se', 'N/A'):.4f})")
            print(f"  Lambda GC = {all_results[pheno].get('lambda_gc', 'N/A'):.4f}")
            print(f"  Mean Chi^2 = {all_results[pheno].get('mean_chi2', 'N/A'):.4f}")
            print(f"  Intercept = {all_results[pheno].get('intercept', 'N/A'):.4f}")

    # Create h2 summary table
    h2_data = []
    for pheno in PHENOTYPES:
        if pheno in all_results:
            h2_data.append({
                'Phenotype': pheno,
                'h2': all_results[pheno].get('h2'),
                'h2_SE': all_results[pheno].get('h2_se'),
                'Lambda_GC': all_results[pheno].get('lambda_gc'),
                'Mean_Chi2': all_results[pheno].get('mean_chi2'),
                'Intercept': all_results[pheno].get('intercept'),
                'Intercept_SE': all_results[pheno].get('intercept_se')
            })

    h2_df = pd.DataFrame(h2_data)
    h2_df.to_csv(OUTPUT_DIR / "sldsc_h2_summary.csv", index=False)
    print(f"\n\nHeritability summary saved to: {OUTPUT_DIR / 'sldsc_h2_summary.csv'}")

    # Create enrichment table (using POP categories as reference)
    if 'POP' in all_results and 'categories' in all_results['POP']:
        categories = all_results['POP']['categories']
        # Clean category names
        clean_cats = [c.replace('L2_0', '') for c in categories]

        enrich_data = {'Category': clean_cats}
        for pheno in PHENOTYPES:
            if pheno in all_results and 'enrichment' in all_results[pheno]:
                enrich = all_results[pheno]['enrichment']
                # Pad if needed
                if len(enrich) < len(clean_cats):
                    enrich = enrich + [None] * (len(clean_cats) - len(enrich))
                enrich_data[pheno] = enrich[:len(clean_cats)]

        enrich_df = pd.DataFrame(enrich_data)
        enrich_df.to_csv(OUTPUT_DIR / "sldsc_enrichment.csv", index=False)
        print(f"Enrichment results saved to: {OUTPUT_DIR / 'sldsc_enrichment.csv'}")

        # Find top enriched categories (positive enrichment > 10)
        print("\n" + "="*60)
        print("Top Enriched Categories (Enrichment > 50)")
        print("="*60)

        for pheno in PHENOTYPES:
            if pheno in all_results and 'enrichment' in all_results[pheno]:
                enrich = all_results[pheno]['enrichment']
                top_idx = [(i, e) for i, e in enumerate(enrich) if e > 50]
                top_idx.sort(key=lambda x: x[1], reverse=True)

                if top_idx:
                    print(f"\n{pheno}:")
                    for idx, val in top_idx[:5]:
                        if idx < len(clean_cats):
                            print(f"  {clean_cats[idx]}: {val:.1f}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
