#!/usr/bin/env python3
"""
regenerate_figures_revised.py - Regenerate key paper figures with colorblind-friendly palettes

Replaces the old hardcoded color scheme with modern, colorblind-friendly colors
using matplotlib's tab10 for categorical and viridis for sequential data.

Generates revised versions of:
  Fig 2: LDSC heatmap + heritability bar
  Fig 3: MAGMA manhattan (multi-panel)
  Fig 4: GNN gene prioritization
  Fig 5: Cross-ancestry scatter
  Fig 6: Drug-gene network

Output: figures/revised/

Author: Revised palette version
Date: 2026-03-17
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Global settings
# ============================================================
sns.set_style('ticks')
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "revised"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# NEW colorblind-friendly palette (tab10-based)
# ============================================================
_tab10 = plt.cm.tab10.colors
PHENOTYPE_COLORS = {
    'POP':             _tab10[0],  # blue
    'BPH':             _tab10[1],  # orange
    'Bladder':         _tab10[2],  # green
    'Constipation':    _tab10[3],  # red
    'FemaleProlapse':  _tab10[4],  # purple
    'Incontinence':    _tab10[5],  # brown
}

# Hex versions for convenience
PHENOTYPE_COLORS_HEX = {k: matplotlib.colors.rgb2hex(v) for k, v in PHENOTYPE_COLORS.items()}

PHENOTYPE_SHORT = {
    'POP': 'POP',
    'BPH': 'BPH',
    'Bladder': 'Bladder',
    'Constipation': 'Constip.',
    'FemaleProlapse': 'F.Prolapse',
    'Incontinence': 'Incontin.',
}

PHENOTYPE_NAMES = {
    'POP': 'POP',
    'BPH': 'BPH',
    'Bladder': 'Bladder\nDysfunction',
    'Constipation': 'Constipation',
    'FemaleProlapse': 'Female\nProlapse',
    'Incontinence': 'Incontinence',
}

INTERACTION_COLORS = {
    'inhibitor':  _tab10[3],  # red
    'agonist':    _tab10[0],  # blue
    'antagonist': _tab10[2],  # green
    'modulator':  _tab10[4],  # purple
    'other':      _tab10[7],  # gray
}

# Chromosome lengths (GRCh38)
CHR_LENGTHS = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555,
    5: 181538259, 6: 170805979, 7: 159345973, 8: 145138636,
    9: 138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16: 90338345,
    17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}


def check_file(path):
    """Check file exists and print status."""
    exists = path.exists()
    tag = "OK" if exists else "MISSING"
    print(f"  [{tag}] {path}")
    return exists


# ============================================================
# Fig 2: LDSC heatmap + heritability bar  (combined figure)
# ============================================================
def fig2_ldsc(ldsc_dir):
    """LDSC genetic correlation heatmap + heritability bar chart."""
    rg_file = ldsc_dir / "genetic_correlation_summary.tsv"
    if not check_file(rg_file):
        return
    rg_df = pd.read_csv(rg_file, sep='\t')

    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
    n = len(phenotypes)

    # Build matrices
    rg_matrix = pd.DataFrame(np.eye(n), index=phenotypes, columns=phenotypes)
    p_matrix = pd.DataFrame(np.zeros((n, n)), index=phenotypes, columns=phenotypes)
    for _, row in rg_df.iterrows():
        p1, p2 = row['phenotype1'], row['phenotype2']
        rg_matrix.loc[p1, p2] = row['rg']
        rg_matrix.loc[p2, p1] = row['rg']
        p_matrix.loc[p1, p2] = row['p']
        p_matrix.loc[p2, p1] = row['p']

    # Extract heritability
    h2_data = {}
    for _, row in rg_df.iterrows():
        p1, p2 = row['phenotype1'], row['phenotype2']
        if p1 not in h2_data:
            h2_data[p1] = {'h2': row['h2_p1'], 'se': row['h2_p1_se']}
        if p2 not in h2_data:
            h2_data[p2] = {'h2': row['h2_p2'], 'se': row['h2_p2_se']}
    h2_df = pd.DataFrame(h2_data).T.reset_index().rename(columns={'index': 'phenotype'})
    h2_df = h2_df.sort_values('h2', ascending=True)

    # --- Combined figure ---
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(16, 7),
                                           gridspec_kw={'width_ratios': [1.2, 1]})

    # (A) Heatmap  -- use viridis-derived diverging cmap via RdYlBu_r for rg
    labels = [PHENOTYPE_SHORT.get(p, p) for p in rg_matrix.index]
    mask = np.triu(np.ones_like(rg_matrix, dtype=bool), k=1)

    sns.heatmap(rg_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='crest',    # modern seaborn sequential colormap, colorblind-safe
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Genetic Correlation (rg)', 'shrink': 0.75},
                xticklabels=labels, yticklabels=labels, ax=ax_heat)

    # Significance stars
    for i in range(n):
        for j in range(i):
            pv = p_matrix.iloc[i, j]
            stars = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
            if stars:
                ax_heat.text(j + 0.5, i + 0.75, stars, ha='center', va='center',
                             fontsize=8, color='black')

    ax_heat.set_title('A  Genetic Correlation Matrix (LDSC)',
                      fontsize=13, fontweight='bold', loc='left')
    ax_heat.text(1.0, -0.08, '*** p<0.001  ** p<0.01  * p<0.05',
                 transform=ax_heat.transAxes, fontsize=8, ha='right', va='top')

    # (B) Heritability bar
    colors = [PHENOTYPE_COLORS.get(p, (0.5, 0.5, 0.5)) for p in h2_df['phenotype']]
    y_pos = np.arange(len(h2_df))
    ax_bar.barh(y_pos, h2_df['h2'], xerr=h2_df['se'], color=colors, alpha=0.85,
                error_kw=dict(ecolor='gray', capsize=3, capthick=1), edgecolor='white')
    bar_labels = [PHENOTYPE_SHORT.get(p, p) for p in h2_df['phenotype']]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels)
    for i, (h2, se) in enumerate(zip(h2_df['h2'], h2_df['se'])):
        ax_bar.text(h2 + se + 0.001, i, f'{h2:.4f}', va='center', fontsize=9)
    ax_bar.set_xlabel('SNP-based Heritability ($h^2$)', fontsize=11)
    ax_bar.set_title('B  SNP Heritability Estimates', fontsize=13, fontweight='bold', loc='left')
    ax_bar.set_xlim(0, max(h2_df['h2'] + h2_df['se']) * 1.4)
    ax_bar.axvline(x=0, color='gray', linewidth=0.5)
    sns.despine(ax=ax_bar)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES_DIR / f'Fig2_ldsc_revised.{ext}', bbox_inches='tight')
    plt.close()
    print("  -> Saved Fig2_ldsc_revised.png/pdf")


# ============================================================
# Fig 3: MAGMA manhattan (multi-panel) -- most important for reviewer
# ============================================================
def fig3_magma(magma_dir):
    """MAGMA gene-based Manhattan plot with revised palette."""
    phenotypes_ordered = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']

    # Load per-phenotype results
    full_results = {}
    for pheno in phenotypes_ordered:
        f = magma_dir / f"{pheno}_genes.genes.out.txt"
        if f.exists():
            df = pd.read_csv(f, sep=r'\s+', comment='#')
            df['Phenotype'] = pheno
            full_results[pheno] = df

    if not full_results:
        print("  No MAGMA results found, skipping Fig 3")
        return

    # Chromosome offsets
    offsets = {}
    cumulative = 0
    for c in range(1, 23):
        offsets[c] = cumulative
        cumulative += CHR_LENGTHS.get(c, 1e8) + 5e6

    # Use two alternating viridis-derived colors for chromosome shading
    chr_color_even = cm.viridis(0.3)
    chr_color_odd  = cm.viridis(0.7)

    fig, axes = plt.subplots(3, 2, figsize=(18, 13))
    axes = axes.flatten()

    for idx, pheno in enumerate(phenotypes_ordered):
        ax = axes[idx]
        if pheno not in full_results:
            ax.set_title(f'{pheno} (No data)')
            continue

        df = full_results[pheno].copy()
        df = df[df['P'].notna() & (df['P'] > 0)]
        df['neglog10p'] = -np.log10(df['P'])
        df['CHR_num'] = pd.to_numeric(df['CHR'], errors='coerce')
        df = df.dropna(subset=['CHR_num'])
        df['CHR_num'] = df['CHR_num'].astype(int)
        df = df[df['CHR_num'].between(1, 22)]
        df['plot_pos'] = df.apply(lambda r: offsets.get(r['CHR_num'], 0) + r['START'], axis=1)

        colors = [chr_color_even if c % 2 == 0 else chr_color_odd for c in df['CHR_num']]
        ax.scatter(df['plot_pos'], df['neglog10p'], c=colors, alpha=0.5, s=10, edgecolors='none')

        # Bonferroni threshold
        bonf = -np.log10(0.05 / len(df))
        ax.axhline(y=bonf, color=PHENOTYPE_COLORS['Constipation'], linestyle='--',
                   linewidth=0.9, alpha=0.8, label='Bonferroni')

        # Suggestive line
        ax.axhline(y=-np.log10(1e-4), color='gray', linestyle=':', linewidth=0.7, alpha=0.6)

        # Label top 5 genes above suggestive
        top5 = df.nlargest(5, 'neglog10p')
        for _, row in top5.iterrows():
            if row['neglog10p'] > 4:
                ax.annotate(str(int(row['GENE'])) if isinstance(row['GENE'], float) else str(row['GENE']),
                            xy=(row['plot_pos'], row['neglog10p']),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, alpha=0.85, fontstyle='italic')

        pheno_color = PHENOTYPE_COLORS.get(pheno, (0, 0, 0))
        ax.set_title(pheno, fontsize=12, fontweight='bold', color=pheno_color)
        ax.set_ylabel('$-\\log_{10}(P)$', fontsize=10)
        ax.set_xlabel('Chromosome', fontsize=9)

        # Chromosome tick labels
        chr_centers = {c: offsets[c] + CHR_LENGTHS[c] / 2 for c in range(1, 23)}
        ax.set_xticks([chr_centers[c] for c in range(1, 23)])
        ax.set_xticklabels([str(c) for c in range(1, 23)], fontsize=6, rotation=0)
        sns.despine(ax=ax)

    fig.suptitle('Gene-based Association Analysis (MAGMA)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES_DIR / f'Fig3_magma_manhattan_revised.{ext}', bbox_inches='tight')
    plt.close()
    print("  -> Saved Fig3_magma_manhattan_revised.png/pdf")


# ============================================================
# Fig 4: GNN gene prioritization
# ============================================================
def fig4_gnn(gnn_dir):
    """GNN gene prioritization -- lollipop / dot plot."""
    ranking_file = gnn_dir / "ensemble_ranking.csv"
    if not check_file(ranking_file):
        return
    df = pd.read_csv(ranking_file)

    # Top 25 genes
    top = df.head(25).copy()
    top = top.sort_values('Ensemble_Score', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    # (A) Lollipop chart of ensemble scores
    y_pos = np.arange(len(top))
    colors_arr = [PHENOTYPE_COLORS['POP'] if pos else PHENOTYPE_COLORS['BPH']
                  for pos in top['Is_Positive']]
    ax1.hlines(y_pos, 0, top['Ensemble_Score'], color='gray', alpha=0.4, linewidth=1)
    ax1.scatter(top['Ensemble_Score'], y_pos, c=colors_arr, s=80, zorder=5, edgecolors='white', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top['Gene'], fontsize=9)
    ax1.set_xlabel('Ensemble Score', fontsize=11)
    ax1.set_title('A  GNN Ensemble Gene Ranking (Top 25)', fontsize=13, fontweight='bold', loc='left')
    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PHENOTYPE_COLORS['POP'],
               markersize=8, label='Known positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PHENOTYPE_COLORS['BPH'],
               markersize=8, label='Novel candidate'),
    ]
    ax1.legend(handles=legend_el, loc='lower right', fontsize=9)
    sns.despine(ax=ax1)

    # (B) Scatter: GNN score vs -log10(p)
    from adjustText import adjust_text
    ax2.scatter(df['neglog10p'], df['Ensemble_Score'],
                c=df['n_phenotypes'], cmap='viridis', s=30, alpha=0.6, edgecolors='white', linewidth=0.3)
    # Highlight top 10
    top10 = df.head(10)
    ax2.scatter(top10['neglog10p'], top10['Ensemble_Score'],
                c='none', s=80, edgecolors=PHENOTYPE_COLORS_HEX['Constipation'], linewidth=1.5)
    texts = []
    for _, row in top10.iterrows():
        texts.append(ax2.text(row['neglog10p'], row['Ensemble_Score'], row['Gene'],
                              fontsize=7, fontstyle='italic'))
    adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                expand=(2.0, 2.0), force_text=(1.5, 1.5), force_points=(1.5, 1.5),
                ensure_inside_axes=True, max_move=50)
    ax2.set_xlabel('$-\\log_{10}(P_{MAGMA})$', fontsize=11)
    ax2.set_ylabel('Ensemble Score', fontsize=11)
    ax2.set_title('B  GNN Score vs MAGMA Significance', fontsize=13, fontweight='bold', loc='left')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2, shrink=0.7, label='# Phenotypes')
    sns.despine(ax=ax2)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES_DIR / f'Fig4_gnn_prioritization_revised.{ext}', bbox_inches='tight')
    plt.close()
    print("  -> Saved Fig4_gnn_prioritization_revised.png/pdf")


# ============================================================
# Fig 5: Cross-ancestry scatter
# ============================================================
def fig5_cross_ancestry(ca_dir):
    """Cross-ancestry effect size comparison scatter plot."""
    snp_file = ca_dir / "top10_snps.csv"
    summary_file = ca_dir / "n81_summary.csv"
    if not check_file(snp_file):
        return

    snps = pd.read_csv(snp_file)

    # Summary stats
    summary = None
    if summary_file.exists():
        summary = pd.read_csv(summary_file)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (A) EUR vs AFR beta scatter
    ax = axes[0]
    ax.scatter(snps['beta_EUR'], snps['beta_AFR'], c=snps['neglog10_pval_EUR'],
               cmap='viridis', s=60, edgecolors='white', linewidth=0.5, zorder=5)
    # Identity line
    lims = [min(snps['beta_EUR'].min(), snps['beta_AFR'].min()) * 1.2,
            max(snps['beta_EUR'].max(), snps['beta_AFR'].max()) * 1.2]
    ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Effect size (EUR)', fontsize=11)
    ax.set_ylabel('Effect size (AFR)', fontsize=11)
    ax.set_title('A  EUR vs AFR Effect Sizes', fontsize=13, fontweight='bold', loc='left')
    plt.colorbar(ax.collections[0], ax=ax, shrink=0.7, label='$-\\log_{10}(P_{EUR})$')
    sns.despine(ax=ax)

    # (B) EUR vs CSA
    ax = axes[1]
    ax.scatter(snps['beta_EUR'], snps['beta_CSA'], c=snps['neglog10_pval_EUR'],
               cmap='viridis', s=60, edgecolors='white', linewidth=0.5, zorder=5)
    lims = [min(snps['beta_EUR'].min(), snps['beta_CSA'].min()) * 1.2,
            max(snps['beta_EUR'].max(), snps['beta_CSA'].max()) * 1.2]
    ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Effect size (EUR)', fontsize=11)
    ax.set_ylabel('Effect size (CSA)', fontsize=11)
    ax.set_title('B  EUR vs CSA Effect Sizes', fontsize=13, fontweight='bold', loc='left')
    plt.colorbar(ax.collections[0], ax=ax, shrink=0.7, label='$-\\log_{10}(P_{EUR})$')
    sns.despine(ax=ax)

    # Add correlation annotation if summary exists
    if summary is not None and 'corr_eur_afr_all' in summary.columns:
        r_afr = summary['corr_eur_afr_all'].iloc[0]
        r_csa = summary['corr_eur_csa_all'].iloc[0]
        axes[0].text(0.05, 0.95, f'r = {r_afr:.4f}', transform=axes[0].transAxes,
                     fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        axes[1].text(0.05, 0.95, f'r = {r_csa:.4f}', transform=axes[1].transAxes,
                     fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig.suptitle('Cross-Ancestry Comparison of Top GWAS Loci', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES_DIR / f'Fig5_cross_ancestry_revised.{ext}', bbox_inches='tight')
    plt.close()
    print("  -> Saved Fig5_cross_ancestry_revised.png/pdf")


# ============================================================
# Fig 6: Drug-gene network
# ============================================================
def fig6_drug_network(drug_dir):
    """Drug-gene interaction network with revised colors."""
    pri_file = drug_dir / "prioritized_candidates.csv"
    if not check_file(pri_file):
        return
    prioritized = pd.read_csv(pri_file)

    top_drugs = prioritized.nlargest(30, 'priority_score')

    G = nx.Graph()
    genes = top_drugs['gene_symbol'].unique()
    for gene in genes:
        G.add_node(gene, node_type='gene')
    for _, row in top_drugs.iterrows():
        drug = row['drug']
        gene = row['gene_symbol']
        itype = row.get('interaction_type', 'other')
        G.add_node(drug, node_type='drug', interaction=itype if pd.notna(itype) else 'other')
        G.add_edge(drug, gene, weight=row['priority_score'])

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    gene_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'gene']
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'drug']

    fig, ax = plt.subplots(figsize=(16, 12))

    # Gene nodes (squares) -- use tab10[0]
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes,
                           node_color=[PHENOTYPE_COLORS['POP']], node_size=1000,
                           node_shape='s', alpha=0.9, ax=ax)

    # Drug nodes (circles) colored by interaction type
    drug_colors = []
    for node in drug_nodes:
        interaction = G.nodes[node].get('interaction', 'other')
        if pd.isna(interaction):
            interaction = 'other'
        drug_colors.append(INTERACTION_COLORS.get(interaction, INTERACTION_COLORS['other']))

    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes,
                           node_color=drug_colors, node_size=500,
                           node_shape='o', alpha=0.8, ax=ax)

    # Edges
    edges = G.edges(data=True)
    edge_widths = [d.get('weight', 1) / 5 for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=edge_widths,
                           edge_color='gray', ax=ax)

    # Labels
    gene_labels = {n: n for n in gene_nodes}
    drug_labels = {n: (n[:15] + '...' if len(n) > 15 else n) for n in drug_nodes}
    nx.draw_networkx_labels(G, pos, gene_labels, font_size=9, font_weight='bold', ax=ax)
    nx.draw_networkx_labels(G, pos, drug_labels, font_size=7, ax=ax)

    # Legend
    legend_elements = [
        Patch(facecolor=PHENOTYPE_COLORS['POP'], label='Gene (target)', alpha=0.9),
        Patch(facecolor=INTERACTION_COLORS['inhibitor'], label='Inhibitor'),
        Patch(facecolor=INTERACTION_COLORS['agonist'], label='Agonist'),
        Patch(facecolor=INTERACTION_COLORS['antagonist'], label='Antagonist'),
        Patch(facecolor=INTERACTION_COLORS['other'], label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    ax.set_title('Drug-Gene Interaction Network\n(Top 30 Candidates by Priority Score)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES_DIR / f'Fig6_drug_network_revised.{ext}', bbox_inches='tight')
    plt.close()
    print("  -> Saved Fig6_drug_network_revised.png/pdf")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Regenerating key paper figures with revised color palette")
    print("  Output directory:", FIGURES_DIR)
    print("=" * 70)

    # Print color palette info
    print("\nNew phenotype color palette (tab10-based, colorblind-friendly):")
    for pheno, col in PHENOTYPE_COLORS_HEX.items():
        print(f"  {pheno:20s} -> {col}")

    # --- Fig 2: LDSC ---
    print("\n[Fig 2] LDSC heatmap + heritability bar...")
    fig2_ldsc(RESULTS_DIR / "ldsc")

    # --- Fig 3: MAGMA manhattan ---
    print("\n[Fig 3] MAGMA gene-based Manhattan plots...")
    fig3_magma(RESULTS_DIR / "magma")

    # --- Fig 4: GNN prioritization ---
    print("\n[Fig 4] GNN gene prioritization...")
    fig4_gnn(RESULTS_DIR / "gnn_prioritization")

    # --- Fig 5: Cross-ancestry ---
    print("\n[Fig 5] Cross-ancestry comparison...")
    fig5_cross_ancestry(RESULTS_DIR / "cross_ancestry")

    # --- Fig 6: Drug-gene network ---
    print("\n[Fig 6] Drug-gene network...")
    fig6_drug_network(RESULTS_DIR / "drug_repurposing")

    print("\n" + "=" * 70)
    print("  All revised figures saved to:", FIGURES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
