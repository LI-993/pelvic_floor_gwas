#!/usr/bin/env python3
"""
Phase 7: Drug Repurposing Analysis - Detailed Summary

Create comprehensive summary and visualization of drug repurposing candidates.
Focus on:
1. FDA-approved drugs for quick translation
2. Drugs already used for related conditions
3. Multi-phenotype targets (shared biology)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
RESULTS_DIR = BASE_DIR / "results/drug_repurposing"
MAGMA_DIR = BASE_DIR / "results/magma"

# Known drugs for pelvic floor conditions
KNOWN_DRUGS = {
    'BPH': {
        'alpha_blockers': ['TAMSULOSIN', 'ALFUZOSIN', 'DOXAZOSIN', 'TERAZOSIN', 'SILODOSIN'],
        '5ari': ['FINASTERIDE', 'DUTASTERIDE'],
        'pde5i': ['TADALAFIL', 'SILDENAFIL'],
    },
    'Incontinence/OAB': {
        'anticholinergics': ['OXYBUTYNIN', 'TOLTERODINE', 'SOLIFENACIN', 'DARIFENACIN', 'FESOTERODINE', 'TROSPIUM'],
        'beta3_agonists': ['MIRABEGRON', 'VIBEGRON'],
    },
    'POP/Prolapse': {
        'hormones': ['ESTRADIOL', 'ESTROGEN', 'PROGESTERONE'],
    }
}

# Pharmacologically relevant interaction types
RELEVANT_TYPES = ['inhibitor', 'agonist', 'antagonist', 'modulator', 'activator', 'blocker']


def load_data():
    """Load DGIdb interactions and MAGMA results."""
    interactions = pd.read_csv(RESULTS_DIR / "dgidb_interactions.csv")
    candidates = pd.read_csv(RESULTS_DIR / "repurposing_candidates.csv")
    top_genes = pd.read_csv(MAGMA_DIR / "magma_top_genes.csv")
    return interactions, candidates, top_genes


def identify_approved_drugs(interactions):
    """Filter for FDA-approved drugs."""
    approved = interactions[interactions['approved'] == True].copy()
    print(f"Approved drugs with interactions: {approved['drug'].nunique()}")
    return approved


def find_known_drugs(interactions):
    """Find interactions involving known pelvic floor drugs."""
    found = []

    for condition, classes in KNOWN_DRUGS.items():
        for drug_class, drugs in classes.items():
            for drug in drugs:
                matches = interactions[interactions['drug'].str.upper().str.contains(drug, na=False)]
                for _, row in matches.iterrows():
                    found.append({
                        'gene': row['gene'],
                        'drug': row['drug'],
                        'condition': condition,
                        'drug_class': drug_class,
                        'interaction_type': row['interaction_type']
                    })

    return pd.DataFrame(found)


def prioritize_candidates(candidates, interactions):
    """Prioritize drug repurposing candidates."""

    priorities = []

    for _, cand in candidates.iterrows():
        # Get interaction details
        gene_int = interactions[interactions['gene'] == cand['gene_symbol']]

        # Calculate priority score
        score = 0

        # Multi-phenotype gene (shared biology)
        score += cand['n_phenotypes'] * 3

        # Strong genetic association
        if cand['min_p'] < 1e-10:
            score += 5
        elif cand['min_p'] < 1e-6:
            score += 3

        # Approved drug
        if gene_int[gene_int['drug'] == cand['drug']]['approved'].any():
            score += 4

        # Has clear mechanism
        int_type = str(cand.get('interaction_type', ''))
        if any(t in int_type.lower() for t in RELEVANT_TYPES):
            score += 2

        priorities.append({
            **cand.to_dict(),
            'priority_score': score
        })

    df = pd.DataFrame(priorities)
    df = df.sort_values('priority_score', ascending=False)
    return df


def generate_summary_report(interactions, candidates, top_genes, known_drugs, prioritized):
    """Generate comprehensive summary report."""

    lines = []
    lines.append("# Drug Repurposing Analysis - Detailed Summary")
    lines.append(f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append("**Phase**: 7 - Drug Repurposing (Extended Analysis)")

    # Overview statistics
    lines.append("\n## Executive Summary")
    lines.append(f"""
| Metric | Count |
|--------|-------|
| Total drug-gene interactions | {len(interactions)} |
| Unique drugs | {interactions['drug'].nunique()} |
| Unique genes with interactions | {interactions['gene'].nunique()} |
| FDA-approved drugs | {interactions[interactions['approved']==True]['drug'].nunique()} |
| Known pelvic floor drugs found | {len(known_drugs)} |
""")

    # Key findings
    lines.append("\n## Key Findings")

    # 1. Known drugs validation
    lines.append("\n### 1. Validation: Known Drugs in Results")
    if len(known_drugs) > 0:
        lines.append("\nOur analysis identified known pelvic floor drugs, validating the approach:")
        for _, row in known_drugs.drop_duplicates(['drug', 'gene']).iterrows():
            lines.append(f"- **{row['drug']}** ({row['drug_class']}) → {row['gene']} ({row['condition']})")
    else:
        lines.append("\nNo currently-used pelvic floor drugs found in the DGIdb interactions.")

    # 2. Top priority candidates
    lines.append("\n### 2. Top Priority Repurposing Candidates")
    lines.append("\nPrioritized by: multi-phenotype targets, genetic significance, FDA approval, clear mechanism")

    top_priority = prioritized.head(30)
    lines.append("\n| Rank | Gene | Drug | Phenotypes | P-value | Score | Mechanism |")
    lines.append("|------|------|------|------------|---------|-------|-----------|")

    for i, (_, row) in enumerate(top_priority.iterrows(), 1):
        gene = row['gene_symbol']
        drug = row['drug'][:30]  # Truncate long names
        phenos = row['phenotypes'][:25] if len(str(row['phenotypes'])) > 25 else row['phenotypes']
        pval = f"{row['min_p']:.2e}"
        score = row['priority_score']
        mechanism = str(row.get('interaction_type', ''))[:20]
        lines.append(f"| {i} | {gene} | {drug} | {phenos} | {pval} | {score} | {mechanism} |")

    # 3. Druggable genes by phenotype
    lines.append("\n### 3. Druggable Genes by Phenotype")

    for pheno in ['BPH', 'POP', 'FemaleProlapse', 'Incontinence']:
        pheno_cands = prioritized[prioritized['phenotypes'].str.contains(pheno, na=False)]
        if len(pheno_cands) > 0:
            genes = pheno_cands['gene_symbol'].unique()[:5]
            lines.append(f"\n**{pheno}**: {', '.join(genes)}")
            # Top drug for each gene
            for gene in genes[:3]:
                gene_drugs = pheno_cands[pheno_cands['gene_symbol'] == gene].head(2)
                drugs = gene_drugs['drug'].tolist()
                lines.append(f"  - {gene}: {', '.join(drugs[:3])}")

    # 4. Approved drug candidates
    lines.append("\n### 4. FDA-Approved Drug Candidates")
    lines.append("\nApproved drugs targeting significant pelvic floor genes:")

    approved_cands = prioritized[
        prioritized['drug'].isin(interactions[interactions['approved']==True]['drug'])
    ]

    if len(approved_cands) > 0:
        for _, row in approved_cands.head(20).iterrows():
            lines.append(f"- **{row['drug']}** → {row['gene_symbol']} ({row['phenotypes']})")
    else:
        # Show genes that have approved drug interactions
        approved_int = interactions[interactions['approved']==True]
        for gene in approved_int['gene'].unique()[:10]:
            drugs = approved_int[approved_int['gene']==gene]['drug'].head(3).tolist()
            lines.append(f"- {gene}: {', '.join(drugs)}")

    # 5. Novel targets
    lines.append("\n### 5. Novel Drug Targets")
    lines.append("\nGenes with many drug interactions but not currently targeted for pelvic floor:")

    gene_drug_counts = interactions.groupby('gene')['drug'].nunique().sort_values(ascending=False)
    for gene, count in gene_drug_counts.head(10).items():
        gene_phenos = candidates[candidates['gene_symbol']==gene]['phenotypes'].iloc[0] if len(candidates[candidates['gene_symbol']==gene]) > 0 else 'Unknown'
        lines.append(f"- **{gene}** ({count} drugs, {gene_phenos})")

    # 6. Biological interpretation
    lines.append("\n## Biological Interpretation")
    lines.append("""
### Key Druggable Pathways

Based on the gene-drug interactions, several druggable pathways emerge:

1. **FGFR Signaling** (FGFR2, FGFR3)
   - Multiple approved FGFR inhibitors available
   - Role in tissue development and fibroblast function
   - Relevant for connective tissue disorders

2. **WNT Signaling** (WNT4)
   - WNT4 is top gene for POP
   - Limited direct drug targets but pathway modulators available
   - Critical for reproductive tract development

3. **Extracellular Matrix** (COL17A1, LOXL1)
   - Collagen and elastin-related genes
   - Potential for collagen modulators
   - Relevant for pelvic floor support

4. **Hormonal Regulation** (WT1, androgen-responsive genes)
   - Already targeted by hormone therapy
   - Supports estrogen use in POP
   - Validates current clinical practice

5. **Kallikrein System** (KLK3/PSA)
   - Top gene for BPH
   - Multiple drugs targeting this pathway
   - Validates prostate-focused therapy
""")

    # 7. Recommendations
    lines.append("\n## Recommendations for Further Investigation")
    lines.append("""
### Priority 1: Repurpose existing approved drugs
1. **FGFR inhibitors** (Erdafitinib, Infigratinib) for BPH/prolapse - investigate safety profile
2. **Curcumin** - natural compound with WT1 interactions, low risk for trial
3. **Tretinoin** - retinoid with connective tissue effects

### Priority 2: Validate in silico findings
1. Expression analysis in pelvic floor tissues (GTEx, HPA)
2. Protein-protein interaction network analysis
3. Molecular docking of top candidates

### Priority 3: Novel target development
1. EHMT2 (G9a) - epigenetic regulator with 800+ drug interactions
2. SMAD3 - TGF-beta signaling node
3. BDNF - neurotrophic factor for muscle/nerve function
""")

    lines.append("\n## Output Files")
    lines.append(f"- `{RESULTS_DIR}/dgidb_interactions.csv` - All interactions")
    lines.append(f"- `{RESULTS_DIR}/repurposing_candidates.csv` - All candidates")
    lines.append(f"- `{RESULTS_DIR}/prioritized_candidates.csv` - Prioritized list")

    return "\n".join(lines)


def main():
    print("="*60)
    print("Drug Repurposing Detailed Analysis")
    print("="*60)

    # Load data
    print("\nLoading data...")
    interactions, candidates, top_genes = load_data()

    # Identify approved drugs
    print("\nAnalyzing approved drugs...")
    approved = identify_approved_drugs(interactions)

    # Find known pelvic floor drugs
    print("\nSearching for known pelvic floor drugs...")
    known_drugs = find_known_drugs(interactions)
    print(f"Found {len(known_drugs)} interactions with known drugs")

    # Prioritize candidates
    print("\nPrioritizing candidates...")
    prioritized = prioritize_candidates(candidates, interactions)
    prioritized.to_csv(RESULTS_DIR / "prioritized_candidates.csv", index=False)

    # Generate report
    print("\nGenerating summary report...")
    report = generate_summary_report(
        interactions, candidates, top_genes, known_drugs, prioritized
    )

    # Save report
    report_path = BASE_DIR / "logs/10_drug_repurposing.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Print top findings
    print("\n" + "="*60)
    print("TOP REPURPOSING CANDIDATES")
    print("="*60)

    print("\n[Priority Score Ranking]")
    for i, (_, row) in enumerate(prioritized.head(15).iterrows(), 1):
        print(f"  {i}. {row['gene_symbol']} → {row['drug'][:40]} (score: {row['priority_score']}, P={row['min_p']:.2e})")

    print("\n[Known Drugs Found]")
    for _, row in known_drugs.drop_duplicates(['drug']).head(10).iterrows():
        print(f"  {row['drug']} ({row['drug_class']}) → {row['gene']}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
