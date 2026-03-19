#!/usr/bin/env python3
"""
23_visualize_lava.py - LAVA局部遗传相关性结果可视化

生成图表:
1. 局部相关性Manhattan图
2. Top显著位点热图
3. 染色体汇总条形图
4. 显著局部相关性分布图

Author: Claude
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
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
RESULTS_DIR = BASE_DIR / "results" / "lava"
FIGURES_DIR = BASE_DIR / "figures" / "lava"
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

# 表型对颜色（用于区分不同配对）
PAIR_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4',
    '#91D1C2', '#DC0000', '#7E6148', '#B09C85', '#FACC52', '#9ECAE1'
]


def load_data():
    """加载LAVA结果数据"""
    bivar_file = RESULTS_DIR / "lava_bivariate.tsv"
    bivar_df = pd.read_csv(bivar_file, sep='\t')

    # 创建表型对标识
    bivar_df['pair'] = bivar_df['phen1'] + ' vs ' + bivar_df['phen2']

    # 计算-log10(p)
    bivar_df['neglog10p'] = -np.log10(bivar_df['p'].clip(lower=1e-300))

    return bivar_df


def get_chromosome_offsets(bivar_df):
    """计算染色体偏移量用于Manhattan图"""
    # 获取每个染色体的最大位置
    chr_max = bivar_df.groupby('chr')['stop'].max()

    # 计算累积偏移
    offsets = {}
    cumulative = 0
    for chrom in range(1, 23):
        offsets[chrom] = cumulative
        if chrom in chr_max.index:
            cumulative += chr_max[chrom] + 10000000  # 添加间隔

    return offsets


def plot_manhattan(bivar_df):
    """绘制局部相关性Manhattan图"""
    fig, ax = plt.subplots(figsize=(16, 6))

    # 计算染色体偏移
    offsets = get_chromosome_offsets(bivar_df)

    # 计算绘图位置
    bivar_df = bivar_df.copy()
    bivar_df['plot_pos'] = bivar_df.apply(
        lambda row: offsets.get(row['chr'], 0) + (row['start'] + row['stop']) / 2,
        axis=1
    )

    # Bonferroni阈值
    bonf_threshold = -np.log10(0.05 / len(bivar_df))
    suggestive_threshold = -np.log10(1e-4)

    # 按染色体着色
    colors = []
    for chrom in bivar_df['chr']:
        colors.append('#4DBBD5' if chrom % 2 == 0 else '#3C5488')

    # 根据相关性方向调整颜色深浅
    alphas = []
    for rho in bivar_df['rho']:
        if rho > 0:
            alphas.append(0.8)
        else:
            alphas.append(0.4)

    # 绘制散点
    scatter = ax.scatter(bivar_df['plot_pos'], bivar_df['neglog10p'],
                        c=colors, alpha=0.6, s=20, edgecolors='none')

    # 添加阈值线
    ax.axhline(y=bonf_threshold, color='red', linestyle='--', linewidth=1,
               label=f'Bonferroni (p={0.05/len(bivar_df):.2e})')
    ax.axhline(y=suggestive_threshold, color='orange', linestyle='--', linewidth=1,
               label='Suggestive (p=1e-4)')

    # 标注top位点
    top_loci = bivar_df.nlargest(10, 'neglog10p')
    for _, row in top_loci.iterrows():
        if row['neglog10p'] > bonf_threshold:
            ax.annotate(f"chr{row['chr']}:{row['locus']}\n{row['phen1']}-{row['phen2']}",
                       xy=(row['plot_pos'], row['neglog10p']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8,
                       arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

    # 设置x轴刻度（染色体中心）
    chr_centers = {}
    for chrom in range(1, 23):
        chr_data = bivar_df[bivar_df['chr'] == chrom]
        if len(chr_data) > 0:
            chr_centers[chrom] = chr_data['plot_pos'].mean()

    ax.set_xticks(list(chr_centers.values()))
    ax.set_xticklabels(list(chr_centers.keys()))

    ax.set_xlabel('Chromosome', fontsize=12)
    ax.set_ylabel('-log10(P-value)', fontsize=12)
    ax.set_title('LAVA Local Genetic Correlation Manhattan Plot\n(All Phenotype Pairs)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # 设置y轴范围
    ax.set_ylim(0, max(bivar_df['neglog10p']) * 1.1)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'lava_manhattan.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'lava_manhattan.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: lava_manhattan.png/pdf")


def plot_top_loci_heatmap(bivar_df):
    """绘制Top显著位点的多表型相关性热图"""
    # 获取Bonferroni显著位点
    bonf_threshold = 0.05 / len(bivar_df)
    sig_df = bivar_df[bivar_df['p'] < bonf_threshold].copy()

    if len(sig_df) == 0:
        print("  No Bonferroni significant loci found, using top 50 by p-value")
        sig_df = bivar_df.nsmallest(50, 'p').copy()

    # 创建位点标识
    sig_df['locus_id'] = 'chr' + sig_df['chr'].astype(str) + ':' + sig_df['locus'].astype(str)

    # 获取唯一位点
    unique_loci = sig_df.groupby('locus_id').agg({
        'p': 'min',
        'chr': 'first',
        'start': 'first',
        'stop': 'first'
    }).reset_index()
    unique_loci = unique_loci.nsmallest(30, 'p')  # 取top 30位点

    # 获取唯一表型对
    pairs = sig_df['pair'].unique()

    # 创建热图矩阵
    heatmap_data = pd.DataFrame(index=unique_loci['locus_id'], columns=pairs)
    heatmap_data = heatmap_data.fillna(0)

    for _, row in sig_df.iterrows():
        if row['locus_id'] in heatmap_data.index:
            heatmap_data.loc[row['locus_id'], row['pair']] = row['rho']

    # 转换为数值
    heatmap_data = heatmap_data.astype(float)

    # 只保留有显著结果的列
    heatmap_data = heatmap_data.loc[:, (heatmap_data != 0).any(axis=0)]

    if heatmap_data.empty or len(heatmap_data.columns) == 0:
        print("  Warning: No data for heatmap")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # 创建自定义颜色映射
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(heatmap_data,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                cbar_kws={'label': 'Local Genetic Correlation (ρ)'},
                ax=ax)

    ax.set_xlabel('Phenotype Pair', fontsize=12)
    ax.set_ylabel('Genomic Locus', fontsize=12)
    ax.set_title('Top Significant Loci: Local Genetic Correlations\n(LAVA Bivariate Analysis)',
                fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'lava_top_loci_heatmap.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'lava_top_loci_heatmap.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: lava_top_loci_heatmap.png/pdf")


def plot_chromosome_summary(bivar_df):
    """绘制各染色体显著位点数量条形图"""
    # Bonferroni阈值
    bonf_threshold = 0.05 / len(bivar_df)
    suggestive_threshold = 1e-4

    # 统计每个染色体的显著位点数
    chr_stats = []
    for chrom in range(1, 23):
        chr_data = bivar_df[bivar_df['chr'] == chrom]
        n_bonf = (chr_data['p'] < bonf_threshold).sum()
        n_sugg = ((chr_data['p'] >= bonf_threshold) & (chr_data['p'] < suggestive_threshold)).sum()
        n_total = len(chr_data)
        chr_stats.append({
            'chr': chrom,
            'Bonferroni': n_bonf,
            'Suggestive': n_sugg,
            'Total_tests': n_total
        })

    chr_df = pd.DataFrame(chr_stats)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(chr_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, chr_df['Bonferroni'], width,
                   label='Bonferroni significant', color='#E64B35', alpha=0.8)
    bars2 = ax.bar(x + width/2, chr_df['Suggestive'], width,
                   label='Suggestive', color='#4DBBD5', alpha=0.8)

    ax.set_xlabel('Chromosome', fontsize=12)
    ax.set_ylabel('Number of Significant Loci', fontsize=12)
    ax.set_title('Distribution of Significant Local Genetic Correlations by Chromosome\n(LAVA)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(chr_df['chr'])
    ax.legend()

    # 在条形上添加数值
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'lava_chromosome_summary.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'lava_chromosome_summary.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: lava_chromosome_summary.png/pdf")


def plot_rho_distribution(bivar_df):
    """绘制局部相关性分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：所有结果的rho分布
    ax1 = axes[0]
    sns.histplot(bivar_df['rho'], bins=50, kde=True, ax=ax1, color='#3C5488', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Local Genetic Correlation (ρ)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Local Genetic Correlations\n(All Tests)', fontsize=12, fontweight='bold')

    # 右图：按表型对分组的箱线图
    ax2 = axes[1]

    # 计算每个表型对的中位rho
    pair_order = bivar_df.groupby('pair')['rho'].median().sort_values(ascending=False).index

    # 只显示top 10对
    if len(pair_order) > 10:
        pair_order = pair_order[:10]

    plot_data = bivar_df[bivar_df['pair'].isin(pair_order)]

    sns.boxplot(data=plot_data, x='rho', y='pair', order=pair_order,
                palette='RdBu_r', ax=ax2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Local Genetic Correlation (ρ)', fontsize=12)
    ax2.set_ylabel('Phenotype Pair', fontsize=12)
    ax2.set_title('Local Genetic Correlations by Phenotype Pair\n(Top 10 Pairs)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'lava_rho_distribution.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'lava_rho_distribution.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: lava_rho_distribution.png/pdf")


def plot_pair_specific_manhattan(bivar_df):
    """为每个表型对绘制单独的Manhattan图（合并为一个多面板图）"""
    pairs = bivar_df['pair'].unique()

    # 选择top 6个表型对（按显著位点数）
    bonf_threshold = 0.05 / len(bivar_df)
    pair_sig_counts = bivar_df[bivar_df['p'] < bonf_threshold].groupby('pair').size()
    top_pairs = pair_sig_counts.nlargest(6).index.tolist()

    if len(top_pairs) < 6:
        # 如果显著的不足6个，补充其他的
        remaining = [p for p in pairs if p not in top_pairs]
        top_pairs.extend(remaining[:6-len(top_pairs)])

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    offsets = get_chromosome_offsets(bivar_df)

    for idx, pair in enumerate(top_pairs[:6]):
        ax = axes[idx]
        pair_data = bivar_df[bivar_df['pair'] == pair].copy()

        pair_data['plot_pos'] = pair_data.apply(
            lambda row: offsets.get(row['chr'], 0) + (row['start'] + row['stop']) / 2,
            axis=1
        )

        # 按染色体着色
        colors = ['#4DBBD5' if c % 2 == 0 else '#3C5488' for c in pair_data['chr']]

        ax.scatter(pair_data['plot_pos'], pair_data['neglog10p'],
                  c=colors, alpha=0.6, s=15, edgecolors='none')

        # 阈值线
        pair_bonf = -np.log10(0.05 / len(pair_data))
        ax.axhline(y=pair_bonf, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

        ax.set_title(pair, fontsize=10, fontweight='bold')
        ax.set_xlabel('Chromosome', fontsize=9)
        ax.set_ylabel('-log10(P)', fontsize=9)

        # 简化x轴
        ax.set_xticks([])

    plt.suptitle('LAVA Local Genetic Correlations by Phenotype Pair',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'lava_pair_manhattan.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'lava_pair_manhattan.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: lava_pair_manhattan.png/pdf")


def generate_summary_stats(bivar_df):
    """生成汇总统计"""
    bonf_threshold = 0.05 / len(bivar_df)

    summary = {
        'Total tests': len(bivar_df),
        'Bonferroni threshold': f'{bonf_threshold:.2e}',
        'Bonferroni significant': (bivar_df['p'] < bonf_threshold).sum(),
        'Suggestive (p<1e-4)': (bivar_df['p'] < 1e-4).sum(),
        'Nominal (p<0.05)': (bivar_df['p'] < 0.05).sum(),
        'Positive correlations': (bivar_df['rho'] > 0).sum(),
        'Negative correlations': (bivar_df['rho'] < 0).sum(),
        'Mean |rho|': abs(bivar_df['rho']).mean(),
        'Max rho': bivar_df['rho'].max(),
        'Min rho': bivar_df['rho'].min(),
    }

    print("\n  Summary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("LAVA Local Genetic Correlation Visualization")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    bivar_df = load_data()
    print(f"  Loaded {len(bivar_df)} bivariate tests")
    print(f"  Phenotype pairs: {bivar_df['pair'].nunique()}")
    print(f"  Unique loci: {bivar_df['locus'].nunique()}")

    # 生成汇总统计
    print("\n[2] Computing summary statistics...")
    summary = generate_summary_stats(bivar_df)

    # 生成可视化
    print("\n[3] Generating visualizations...")

    print("\n  3.1 Manhattan plot...")
    plot_manhattan(bivar_df)

    print("\n  3.2 Top loci heatmap...")
    plot_top_loci_heatmap(bivar_df)

    print("\n  3.3 Chromosome summary...")
    plot_chromosome_summary(bivar_df)

    print("\n  3.4 Rho distribution...")
    plot_rho_distribution(bivar_df)

    print("\n  3.5 Pair-specific Manhattan plots...")
    plot_pair_specific_manhattan(bivar_df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
