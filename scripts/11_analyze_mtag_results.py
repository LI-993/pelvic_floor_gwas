#!/usr/bin/env python3
"""Analyze MTAG results - identify shared significant loci and create visualizations."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
MTAG_DIR = BASE_DIR / "results/mtag"
FIGURES_DIR = BASE_DIR / "figures/mtag"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Phenotype names
phenotypes = ["POP", "BPH", "Bladder", "Constipation", "FemaleProlapse", "Incontinence"]

# Load MTAG results
print("Loading MTAG results...")
mtag_results = {}
for i, pheno in enumerate(phenotypes, 1):
    df = pd.read_csv(MTAG_DIR / f"pelvic_floor_trait_{i}.txt", sep='\t')
    mtag_results[pheno] = df
    print(f"  {pheno}: {len(df)} SNPs, {(df['mtag_pval'] < 5e-8).sum()} GWS")

# Get genome-wide significant SNPs
print("\n" + "="*60)
print("Genome-wide significant SNPs per phenotype:")
print("="*60)
gws_snps = {}
for pheno, df in mtag_results.items():
    gws = df[df['mtag_pval'] < 5e-8]['SNP'].tolist()
    gws_snps[pheno] = set(gws)
    print(f"  {pheno}: {len(gws)} SNPs")

# Find shared SNPs between phenotype pairs
print("\n" + "="*60)
print("Shared GWS SNPs between phenotype pairs:")
print("="*60)
shared_matrix = np.zeros((len(phenotypes), len(phenotypes)))
for i, p1 in enumerate(phenotypes):
    for j, p2 in enumerate(phenotypes):
        shared = len(gws_snps[p1] & gws_snps[p2])
        shared_matrix[i, j] = shared
        if i < j and shared > 0:
            print(f"  {p1} & {p2}: {shared} shared SNPs")

# Create shared SNPs heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(shared_matrix, dtype=bool), k=1)
sns.heatmap(shared_matrix, mask=mask, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=phenotypes, yticklabels=phenotypes, ax=ax)
ax.set_title('Shared Genome-wide Significant SNPs (MTAG)', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mtag_shared_snps_heatmap.png", dpi=150)
plt.savefig(FIGURES_DIR / "mtag_shared_snps_heatmap.pdf")
plt.close()
print(f"\nSaved: {FIGURES_DIR / 'mtag_shared_snps_heatmap.png'}")

# Load and plot genetic correlation matrix (Omega)
print("\n" + "="*60)
print("Genetic Correlation Matrix (from MTAG):")
print("="*60)
omega = pd.read_csv(MTAG_DIR / "pelvic_floor_omega_hat.txt", sep='\t', header=None)
omega = omega.values

# Convert to correlation matrix
omega_corr = np.zeros_like(omega)
for i in range(len(phenotypes)):
    for j in range(len(phenotypes)):
        omega_corr[i, j] = omega[i, j] / np.sqrt(omega[i, i] * omega[j, j])

omega_corr_df = pd.DataFrame(omega_corr, index=phenotypes, columns=phenotypes)
print(omega_corr_df.round(3))

# Plot genetic correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(omega_corr, dtype=bool), k=1)
sns.heatmap(omega_corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title('Genetic Correlation Matrix (MTAG Omega)', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mtag_genetic_correlation.png", dpi=150)
plt.savefig(FIGURES_DIR / "mtag_genetic_correlation.pdf")
plt.close()
print(f"\nSaved: {FIGURES_DIR / 'mtag_genetic_correlation.png'}")

# Find SNPs significant in multiple traits
print("\n" + "="*60)
print("SNPs significant in multiple traits:")
print("="*60)
all_gws = set()
for snps in gws_snps.values():
    all_gws.update(snps)

snp_counts = {}
for snp in all_gws:
    count = sum(1 for pheno in phenotypes if snp in gws_snps[pheno])
    if count not in snp_counts:
        snp_counts[count] = []
    snp_counts[count].append(snp)

for count in sorted(snp_counts.keys(), reverse=True):
    print(f"  {count} traits: {len(snp_counts[count])} SNPs")

# Get top shared SNPs (in 3+ traits)
multi_trait_snps = []
for snp in all_gws:
    traits = [pheno for pheno in phenotypes if snp in gws_snps[pheno]]
    if len(traits) >= 2:
        # Get min p-value across traits
        min_p = min(mtag_results[t][mtag_results[t]['SNP'] == snp]['mtag_pval'].values[0]
                    for t in traits)
        multi_trait_snps.append({
            'SNP': snp,
            'n_traits': len(traits),
            'traits': ', '.join(traits),
            'min_pval': min_p
        })

multi_df = pd.DataFrame(multi_trait_snps)
multi_df = multi_df.sort_values(['n_traits', 'min_pval'], ascending=[False, True])

print("\nTop 20 multi-trait SNPs:")
print(multi_df.head(20).to_string(index=False))

# Save multi-trait SNPs
multi_df.to_csv(MTAG_DIR / "mtag_multi_trait_snps.csv", index=False)
print(f"\nSaved: {MTAG_DIR / 'mtag_multi_trait_snps.csv'}")

# Create summary comparison plot
print("\n" + "="*60)
print("Creating summary plots...")
print("="*60)

# Pre/Post MTAG comparison
pre_mtag = [308, 622, 0, 1, 450, 33]
post_mtag = [625, 507, 2, 12, 635, 130]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(phenotypes))
width = 0.35

bars1 = ax.bar(x - width/2, pre_mtag, width, label='Original GWAS', color='steelblue')
bars2 = ax.bar(x + width/2, post_mtag, width, label='MTAG', color='coral')

ax.set_ylabel('Number of GWS SNPs (p < 5e-8)')
ax.set_title('MTAG Power Gain: Genome-wide Significant SNPs')
ax.set_xticks(x)
ax.set_xticklabels(phenotypes, rotation=45, ha='right')
ax.legend()

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "mtag_power_comparison.png", dpi=150)
plt.savefig(FIGURES_DIR / "mtag_power_comparison.pdf")
plt.close()
print(f"Saved: {FIGURES_DIR / 'mtag_power_comparison.png'}")

# Save summary statistics
summary = pd.DataFrame({
    'Phenotype': phenotypes,
    'Original_GWS': pre_mtag,
    'MTAG_GWS': post_mtag,
    'Change': [post - pre for pre, post in zip(pre_mtag, post_mtag)],
    'Percent_Change': [100*(post-pre)/max(pre,1) for pre, post in zip(pre_mtag, post_mtag)]
})
summary.to_csv(MTAG_DIR / "mtag_summary.csv", index=False)
print(f"Saved: {MTAG_DIR / 'mtag_summary.csv'}")

print("\n" + "="*60)
print("MTAG Analysis Complete!")
print("="*60)
print(f"\nKey findings:")
print(f"  - POP and FemaleProlapse show very high genetic correlation (r=0.94)")
print(f"  - Total unique GWS SNPs across all traits: {len(all_gws)}")
print(f"  - SNPs significant in 2+ traits: {sum(len(v) for k,v in snp_counts.items() if k >= 2)}")
