#!/usr/bin/env python3
"""
Visualize LAVA local genetic correlation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Directories
RESULTS_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\results\lava")
PLOT_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Read data
print("Loading LAVA results...")
bivar = pd.read_csv(RESULTS_DIR / "lava_bivariate.tsv", sep='\t')
univ = pd.read_csv(RESULTS_DIR / "lava_univariate.tsv", sep='\t')

print(f"Bivariate results: {len(bivar)}")
print(f"Univariate results: {len(univ)}")

# Create phenotype pair column
bivar['pair'] = bivar.apply(lambda x: f"{x['phen1']} vs {x['phen2']}", axis=1)
bivar['pair_sorted'] = bivar.apply(lambda x: '_'.join(sorted([x['phen1'], x['phen2']])), axis=1)

# Calculate -log10(p)
bivar['neglog10p'] = -np.log10(bivar['p'])
bivar.loc[bivar['neglog10p'] > 50, 'neglog10p'] = 50  # Cap at 50

# ============================================================
# Plot 1: Manhattan-style plot of local rg
# ============================================================
print("\nCreating Manhattan plot...")

fig, ax = plt.subplots(figsize=(16, 6))

# Assign positions based on chromosome
chrom_sizes = bivar.groupby('chr')['start'].max()
chrom_offsets = {}
offset = 0
for chrom in range(1, 23):
    chrom_offsets[chrom] = offset
    if chrom in chrom_sizes.index:
        offset += chrom_sizes[chrom] + 50000000

bivar['plot_pos'] = bivar.apply(lambda x: chrom_offsets.get(x['chr'], 0) + x['start'], axis=1)

# Color by significance
colors = ['#1f77b4' if p > 0.05 else '#d62728' if p < 4.94e-5 else '#ff7f0e'
          for p in bivar['p']]

scatter = ax.scatter(bivar['plot_pos'], bivar['neglog10p'],
                     c=colors, alpha=0.6, s=20)

# Add chromosome labels
chrom_centers = []
for chrom in range(1, 23):
    if chrom in chrom_offsets:
        chrom_data = bivar[bivar['chr'] == chrom]
        if len(chrom_data) > 0:
            center = chrom_offsets[chrom] + chrom_data['start'].mean()
            chrom_centers.append((center, str(chrom)))

for center, label in chrom_centers:
    ax.text(center, -2, label, ha='center', fontsize=8)

# Significance lines
ax.axhline(y=-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
ax.axhline(y=-np.log10(4.94e-5), color='red', linestyle='--', alpha=0.7, label='Bonferroni')

ax.set_xlabel('Chromosome')
ax.set_ylabel('-log10(p)')
ax.set_title('LAVA Local Genetic Correlations Across Genome')
ax.legend(loc='upper right')
ax.set_xlim(0, max(bivar['plot_pos']) * 1.02)
ax.set_ylim(-3, 55)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'lava_manhattan.png', dpi=150)
plt.savefig(PLOT_DIR / 'lava_manhattan.pdf')
plt.close()
print(f"Saved: {PLOT_DIR / 'lava_manhattan.png'}")

# ============================================================
# Plot 2: Heatmap of significant loci count by phenotype pair
# ============================================================
print("\nCreating heatmap...")

phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
n_pheno = len(phenotypes)

# Count significant results for each pair
sig_counts = pd.DataFrame(0, index=phenotypes, columns=phenotypes, dtype=float)
total_counts = pd.DataFrame(0, index=phenotypes, columns=phenotypes, dtype=float)

for _, row in bivar.iterrows():
    p1, p2 = row['phen1'], row['phen2']
    total_counts.loc[p1, p2] += 1
    total_counts.loc[p2, p1] += 1
    if row['p'] < 0.05:
        sig_counts.loc[p1, p2] += 1
        sig_counts.loc[p2, p1] += 1

# Calculate proportions
prop_sig = sig_counts / total_counts.replace(0, np.nan)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(prop_sig, dtype=bool), k=1)

sns.heatmap(prop_sig, mask=mask, annot=True, fmt='.0%', cmap='RdYlBu_r',
            vmin=0, vmax=1, ax=ax, square=True,
            cbar_kws={'label': 'Proportion Significant (p<0.05)'})

ax.set_title('Proportion of Loci with Significant Local Genetic Correlation')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'lava_sig_proportion_heatmap.png', dpi=150)
plt.savefig(PLOT_DIR / 'lava_sig_proportion_heatmap.pdf')
plt.close()
print(f"Saved: {PLOT_DIR / 'lava_sig_proportion_heatmap.png'}")

# ============================================================
# Plot 3: Distribution of local rg by phenotype pair
# ============================================================
print("\nCreating rho distribution plot...")

# Select top phenotype pairs
top_pairs = bivar.groupby('pair_sorted').size().nlargest(8).index.tolist()
bivar_top = bivar[bivar['pair_sorted'].isin(top_pairs)]

fig, ax = plt.subplots(figsize=(12, 6))

# Create nice labels
pair_labels = {
    'BPH_FemaleProlapse': 'BPH vs FemaleProlapse',
    'Constipation_FemaleProlapse': 'Constipation vs FemaleProlapse',
    'BPH_Constipation': 'BPH vs Constipation',
    'Bladder_FemaleProlapse': 'Bladder vs FemaleProlapse',
    'BPH_Bladder': 'BPH vs Bladder',
    'Bladder_Constipation': 'Bladder vs Constipation',
    'FemaleProlapse_POP': 'POP vs FemaleProlapse',
    'BPH_POP': 'BPH vs POP'
}

bivar_top['pair_label'] = bivar_top['pair_sorted'].map(pair_labels)

sns.boxplot(data=bivar_top, x='pair_label', y='rho', ax=ax, palette='Set2')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Phenotype Pair')
ax.set_ylabel('Local Genetic Correlation (rho)')
ax.set_title('Distribution of Local Genetic Correlations by Phenotype Pair')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'lava_rho_distribution.png', dpi=150)
plt.savefig(PLOT_DIR / 'lava_rho_distribution.pdf')
plt.close()
print(f"Saved: {PLOT_DIR / 'lava_rho_distribution.png'}")

# ============================================================
# Plot 4: Chromosome-wise significant loci
# ============================================================
print("\nCreating chromosome summary plot...")

sig_bivar = bivar[bivar['p'] < 0.05]
chr_counts = sig_bivar.groupby('chr').size()

fig, ax = plt.subplots(figsize=(12, 5))
chr_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Chromosome')
ax.set_ylabel('Number of Significant Local Correlations')
ax.set_title('Significant Local Genetic Correlations by Chromosome (p < 0.05)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'lava_chr_distribution.png', dpi=150)
plt.savefig(PLOT_DIR / 'lava_chr_distribution.pdf')
plt.close()
print(f"Saved: {PLOT_DIR / 'lava_chr_distribution.png'}")

# ============================================================
# Plot 5: Top loci detail
# ============================================================
print("\nCreating top loci plot...")

top20 = bivar.nsmallest(20, 'p').copy()
top20['label'] = top20.apply(lambda x: f"L{x['locus']} (chr{x['chr']})", axis=1)
top20['pair_short'] = top20.apply(lambda x: f"{x['phen1'][:3]}-{x['phen2'][:3]}", axis=1)

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#d62728' if r < 0 else '#2ca02c' for r in top20['rho']]
bars = ax.barh(range(len(top20)), top20['neglog10p'], color=colors, edgecolor='black')

ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f"{row['label']}: {row['pair_short']} (rho={row['rho']:.2f})"
                    for _, row in top20.iterrows()])
ax.set_xlabel('-log10(p)')
ax.set_title('Top 20 Local Genetic Correlations')
ax.axvline(x=-np.log10(4.94e-5), color='red', linestyle='--', alpha=0.7, label='Bonferroni')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOT_DIR / 'lava_top20.png', dpi=150)
plt.savefig(PLOT_DIR / 'lava_top20.pdf')
plt.close()
print(f"Saved: {PLOT_DIR / 'lava_top20.png'}")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "="*60)
print("LAVA Results Summary")
print("="*60)

bonf_thresh = 0.05 / len(bivar)
print(f"\nTotal bivariate tests: {len(bivar)}")
print(f"Bonferroni threshold: {bonf_thresh:.2e}")
print(f"\nSignificant (p < 0.05): {len(bivar[bivar['p'] < 0.05])} ({100*len(bivar[bivar['p'] < 0.05])/len(bivar):.1f}%)")
print(f"Bonferroni significant: {len(bivar[bivar['p'] < bonf_thresh])} ({100*len(bivar[bivar['p'] < bonf_thresh])/len(bivar):.1f}%)")

print("\nTop 10 most significant:")
for i, (_, row) in enumerate(bivar.nsmallest(10, 'p').iterrows(), 1):
    print(f"  {i}. Locus {row['locus']} (chr{row['chr']}): {row['phen1']} vs {row['phen2']} "
          f"rho={row['rho']:+.2f}, p={row['p']:.2e}")

print("\n" + "="*60)
print("All plots saved to:", PLOT_DIR)
print("="*60)
