#!/usr/bin/env python3
"""
24_visualize_magma.py - MAGMA基因分析结果可视化

生成图表:
1. 基因级别Manhattan图
2. Top基因条形图
3. 共享基因UpSet图
4. 表型-基因热图

Author: Claude
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 路径设置
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "magma"
FIGURES_DIR = BASE_DIR / "figures" / "magma"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 表型颜色
PHENOTYPE_COLORS = {
    'POP': '#E64B35',
    'BPH': '#4DBBD5',
    'Bladder': '#00A087',
    'Constipation': '#3C5488',
    'FemaleProlapse': '#F39B7F',
    'Incontinence': '#8491B4'
}

# 染色体长度（GRCh38，用于Manhattan图）
CHR_LENGTHS = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555,
    5: 181538259, 6: 170805979, 7: 159345973, 8: 145138636,
    9: 138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16: 90338345,
    17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468
}


def load_data():
    """加载MAGMA结果数据"""
    # 加载top基因
    top_genes = pd.read_csv(RESULTS_DIR / "magma_top_genes.csv")

    # 尝试加载完整的magma_summary
    summary_file = RESULTS_DIR / "magma_summary.csv"
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
    else:
        summary_df = None

    # 加载每个表型的完整MAGMA输出
    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
    full_results = {}

    for pheno in phenotypes:
        genes_file = RESULTS_DIR / f"{pheno}_genes.genes.out.txt"
        if genes_file.exists():
            df = pd.read_csv(genes_file, sep=r'\s+', comment='#')
            df['Phenotype'] = pheno
            full_results[pheno] = df

    return top_genes, summary_df, full_results


def plot_manhattan_gene(full_results):
    """绘制基因级别Manhattan图（所有表型叠加）"""
    if not full_results:
        print("  Warning: No full MAGMA results available for Manhattan plot")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']

    # 计算染色体偏移
    offsets = {}
    cumulative = 0
    for chrom in range(1, 23):
        offsets[chrom] = cumulative
        cumulative += CHR_LENGTHS.get(chrom, 100000000) + 5000000

    for idx, pheno in enumerate(phenotypes):
        ax = axes[idx]

        if pheno not in full_results:
            ax.set_title(f'{pheno} (No data)', fontsize=12)
            continue

        df = full_results[pheno].copy()

        # 过滤有效数据
        df = df[df['P'].notna() & (df['P'] > 0)]
        df['neglog10p'] = -np.log10(df['P'])

        # 处理染色体（可能是X=23）
        df['CHR_num'] = df['CHR'].replace({'X': 23, 'Y': 24}).astype(int)
        df = df[df['CHR_num'] <= 22]

        # 计算绘图位置（使用基因起始位置）
        df['plot_pos'] = df.apply(
            lambda row: offsets.get(row['CHR_num'], 0) + row['START'],
            axis=1
        )

        # 按染色体着色
        colors = ['#4DBBD5' if c % 2 == 0 else '#3C5488' for c in df['CHR_num']]

        ax.scatter(df['plot_pos'], df['neglog10p'],
                  c=colors, alpha=0.5, s=8, edgecolors='none')

        # Bonferroni阈值
        bonf_threshold = -np.log10(0.05 / len(df))
        ax.axhline(y=bonf_threshold, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

        # 标注top 5基因
        top5 = df.nlargest(5, 'neglog10p')
        for _, row in top5.iterrows():
            if row['neglog10p'] > 4:  # 只标注p < 1e-4的
                ax.annotate(row['GENE'],
                           xy=(row['plot_pos'], row['neglog10p']),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=7, alpha=0.8)

        ax.set_title(pheno, fontsize=12, fontweight='bold',
                    color=PHENOTYPE_COLORS.get(pheno, 'black'))
        ax.set_xlabel('Chromosome', fontsize=9)
        ax.set_ylabel('-log10(P)', fontsize=9)

        # 简化x轴刻度
        chr_centers = {c: offsets[c] + CHR_LENGTHS[c]/2 for c in range(1, 23)}
        ax.set_xticks([chr_centers[c] for c in [1, 5, 10, 15, 20]])
        ax.set_xticklabels([1, 5, 10, 15, 20])

    plt.suptitle('Gene-based Association Analysis (MAGMA)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'magma_manhattan.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'magma_manhattan.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: magma_manhattan.png/pdf")


def plot_top_genes_bar(top_genes):
    """绘制各表型Top 10基因条形图"""
    phenotypes = top_genes['Phenotype'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, pheno in enumerate(phenotypes):
        ax = axes[idx]
        pheno_data = top_genes[top_genes['Phenotype'] == pheno].sort_values('P')

        # 计算-log10(P)
        pheno_data['neglog10p'] = -np.log10(pheno_data['P'])

        # 条形图
        y_pos = np.arange(len(pheno_data))
        color = PHENOTYPE_COLORS.get(pheno, '#888888')

        bars = ax.barh(y_pos, pheno_data['neglog10p'],
                       color=color, alpha=0.8, edgecolor='white')

        # Bonferroni阈值线（假设~19000个基因）
        bonf_line = -np.log10(0.05 / 19000)
        ax.axvline(x=bonf_line, color='red', linestyle='--', linewidth=1,
                   label=f'Bonferroni')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pheno_data['Symbol'])
        ax.invert_yaxis()

        ax.set_xlabel('-log10(P)', fontsize=10)
        ax.set_title(pheno, fontsize=12, fontweight='bold', color=color)

        # 在条形末端添加P值
        for i, (p, neglogp) in enumerate(zip(pheno_data['P'], pheno_data['neglog10p'])):
            ax.text(neglogp + 0.3, i, f'{p:.1e}', va='center', fontsize=8)

    plt.suptitle('Top 10 Genes per Phenotype (MAGMA Gene-based Analysis)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'magma_top_genes_bar.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'magma_top_genes_bar.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: magma_top_genes_bar.png/pdf")


def plot_shared_genes_upset(full_results):
    """绘制共享基因UpSet图"""
    if not full_results:
        print("  Warning: No data for UpSet plot")
        return

    # 为每个表型获取显著基因（Bonferroni）
    sig_genes = {}
    for pheno, df in full_results.items():
        bonf_threshold = 0.05 / len(df)
        sig = df[df['P'] < bonf_threshold]['GENE'].tolist()
        sig_genes[pheno] = set(sig)

    # 如果有些表型没有显著基因，使用较宽松的阈值
    for pheno in sig_genes:
        if len(sig_genes[pheno]) == 0:
            df = full_results[pheno]
            # 使用suggestive阈值
            sig = df[df['P'] < 1e-4]['GENE'].tolist()
            sig_genes[pheno] = set(sig)

    # 计算交集
    phenotypes = list(sig_genes.keys())
    all_genes = set().union(*sig_genes.values())

    if len(all_genes) == 0:
        print("  Warning: No significant genes found")
        return

    # 创建二进制矩阵
    gene_matrix = pd.DataFrame(index=list(all_genes))
    for pheno in phenotypes:
        gene_matrix[pheno] = gene_matrix.index.isin(sig_genes[pheno]).astype(int)

    # 计算各种组合的交集大小
    intersections = {}

    # 单个表型
    for pheno in phenotypes:
        key = (pheno,)
        count = (gene_matrix[pheno] == 1).sum()
        if count > 0:
            intersections[key] = count

    # 两两组合
    for p1, p2 in combinations(phenotypes, 2):
        shared = ((gene_matrix[p1] == 1) & (gene_matrix[p2] == 1)).sum()
        if shared > 0:
            intersections[(p1, p2)] = shared

    # 三三组合
    for combo in combinations(phenotypes, 3):
        mask = (gene_matrix[list(combo)] == 1).all(axis=1)
        shared = mask.sum()
        if shared > 0:
            intersections[combo] = shared

    # 绘制简化版UpSet图
    fig, (ax_bar, ax_matrix) = plt.subplots(2, 1, figsize=(14, 8),
                                            gridspec_kw={'height_ratios': [3, 1]})

    # 排序交集（按大小）
    sorted_intersections = sorted(intersections.items(), key=lambda x: -x[1])[:20]

    # 条形图
    x_pos = np.arange(len(sorted_intersections))
    heights = [v for k, v in sorted_intersections]
    labels = [' & '.join(k) if len(k) <= 2 else f'{len(k)} traits' for k, v in sorted_intersections]

    colors = []
    for key, val in sorted_intersections:
        if len(key) == 1:
            colors.append(PHENOTYPE_COLORS.get(key[0], '#888888'))
        else:
            colors.append('#3C5488')

    ax_bar.bar(x_pos, heights, color=colors, alpha=0.8, edgecolor='white')
    ax_bar.set_ylabel('Number of Genes', fontsize=12)
    ax_bar.set_title('Shared Significant Genes Across Phenotypes (MAGMA)',
                    fontsize=14, fontweight='bold')

    # 在条形上添加数值
    for i, h in enumerate(heights):
        ax_bar.text(i, h + 0.5, str(h), ha='center', fontsize=9)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    # 矩阵指示器（简化）
    ax_matrix.axis('off')

    # 添加图例
    legend_elements = [Patch(facecolor=PHENOTYPE_COLORS[p], label=p, alpha=0.8)
                      for p in phenotypes if p in PHENOTYPE_COLORS]
    legend_elements.append(Patch(facecolor='#3C5488', label='Multiple', alpha=0.8))
    ax_bar.legend(handles=legend_elements, loc='upper right', ncol=3)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'magma_shared_genes_upset.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'magma_shared_genes_upset.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: magma_shared_genes_upset.png/pdf")


def plot_gene_phenotype_heatmap(full_results):
    """绘制基因-表型关联热图（Top基因）"""
    if not full_results:
        print("  Warning: No data for heatmap")
        return

    # 获取所有表型中最显著的基因
    all_top_genes = []
    for pheno, df in full_results.items():
        top = df.nsmallest(20, 'P')[['GENE', 'P']].copy()
        top['Phenotype'] = pheno
        all_top_genes.append(top)

    combined = pd.concat(all_top_genes)
    unique_genes = combined.groupby('GENE')['P'].min().nsmallest(50).index.tolist()

    # 创建热图矩阵
    phenotypes = list(full_results.keys())
    heatmap_data = pd.DataFrame(index=unique_genes, columns=phenotypes)

    for pheno, df in full_results.items():
        gene_p = df.set_index('GENE')['P']
        for gene in unique_genes:
            if gene in gene_p.index:
                heatmap_data.loc[gene, pheno] = -np.log10(gene_p[gene])
            else:
                heatmap_data.loc[gene, pheno] = 0

    heatmap_data = heatmap_data.astype(float)

    # 按最大值排序
    heatmap_data['max_val'] = heatmap_data.max(axis=1)
    heatmap_data = heatmap_data.sort_values('max_val', ascending=False)
    heatmap_data = heatmap_data.drop('max_val', axis=1)

    # 只取top 30
    heatmap_data = heatmap_data.head(30)

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(heatmap_data,
                cmap='YlOrRd',
                linewidths=0.5,
                cbar_kws={'label': '-log10(P-value)'},
                ax=ax)

    # Bonferroni线（约-log10(2.6e-6) = 5.58）
    ax.axhline(y=0, color='black', linewidth=2)

    ax.set_xlabel('Phenotype', fontsize=12)
    ax.set_ylabel('Gene', fontsize=12)
    ax.set_title('Gene-Phenotype Association Heatmap (MAGMA)\nTop 30 Genes',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'magma_gene_phenotype_heatmap.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'magma_gene_phenotype_heatmap.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: magma_gene_phenotype_heatmap.png/pdf")


def plot_significant_genes_summary(full_results):
    """绘制显著基因数量汇总图"""
    if not full_results:
        return

    summary_data = []
    for pheno, df in full_results.items():
        n_total = len(df)
        n_bonf = (df['P'] < 0.05 / n_total).sum()
        n_sugg = ((df['P'] >= 0.05 / n_total) & (df['P'] < 1e-4)).sum()
        n_nominal = ((df['P'] >= 1e-4) & (df['P'] < 0.05)).sum()

        summary_data.append({
            'Phenotype': pheno,
            'Bonferroni': n_bonf,
            'Suggestive': n_sugg,
            'Nominal': n_nominal
        })

    summary_df = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(summary_df))
    width = 0.25

    bars1 = ax.bar(x - width, summary_df['Bonferroni'], width,
                   label='Bonferroni (p<2.6e-6)', color='#E64B35', alpha=0.8)
    bars2 = ax.bar(x, summary_df['Suggestive'], width,
                   label='Suggestive (p<1e-4)', color='#4DBBD5', alpha=0.8)
    bars3 = ax.bar(x + width, summary_df['Nominal'], width,
                   label='Nominal (p<0.05)', color='#00A087', alpha=0.8)

    ax.set_xlabel('Phenotype', fontsize=12)
    ax.set_ylabel('Number of Significant Genes', fontsize=12)
    ax.set_title('Significant Genes per Phenotype (MAGMA)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Phenotype'], rotation=45, ha='right')
    ax.legend()

    # 在条形上添加数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'magma_significant_summary.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'magma_significant_summary.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: magma_significant_summary.png/pdf")


def main():
    """主函数"""
    print("=" * 60)
    print("MAGMA Gene Analysis Visualization")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    top_genes, summary_df, full_results = load_data()
    print(f"  Top genes: {len(top_genes)} entries")
    print(f"  Full results available for: {list(full_results.keys())}")

    # 生成可视化
    print("\n[2] Generating visualizations...")

    print("\n  2.1 Manhattan plots...")
    plot_manhattan_gene(full_results)

    print("\n  2.2 Top genes bar charts...")
    plot_top_genes_bar(top_genes)

    print("\n  2.3 Shared genes UpSet plot...")
    plot_shared_genes_upset(full_results)

    print("\n  2.4 Gene-phenotype heatmap...")
    plot_gene_phenotype_heatmap(full_results)

    print("\n  2.5 Significant genes summary...")
    plot_significant_genes_summary(full_results)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
