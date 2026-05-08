#!/usr/bin/env python3
"""
22_visualize_ldsc.py - LDSC遗传相关性结果可视化

生成图表:
1. 遗传相关性热图 (带显著性标记)
2. 表型相关性网络图
3. 遗传力条形图

Author: Claude
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 路径设置
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "ldsc"
FIGURES_DIR = BASE_DIR / "figures" / "ldsc"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 表型显示名称映射
PHENOTYPE_NAMES = {
    'POP': 'POP',
    'BPH': 'BPH',
    'Bladder': 'Bladder\nDysfunction',
    'Constipation': 'Constipation',
    'FemaleProlapse': 'Female\nProlapse',
    'Incontinence': 'Incontinence'
}

# 表型简短名称（用于热图）
PHENOTYPE_SHORT = {
    'POP': 'POP',
    'BPH': 'BPH',
    'Bladder': 'Bladder',
    'Constipation': 'Constip.',
    'FemaleProlapse': 'F.Prolapse',
    'Incontinence': 'Incontin.'
}

# 表型颜色
PHENOTYPE_COLORS = {
    'POP': '#E64B35',
    'BPH': '#4DBBD5',
    'Bladder': '#00A087',
    'Constipation': '#3C5488',
    'FemaleProlapse': '#F39B7F',
    'Incontinence': '#8491B4'
}


def load_data():
    """加载LDSC结果数据"""
    # 加载遗传相关性摘要
    rg_file = RESULTS_DIR / "genetic_correlation_summary.tsv"
    rg_df = pd.read_csv(rg_file, sep='\t')

    return rg_df


def create_correlation_matrix(rg_df):
    """从成对相关性创建完整矩阵"""
    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
    n = len(phenotypes)

    # 初始化矩阵
    rg_matrix = pd.DataFrame(np.eye(n), index=phenotypes, columns=phenotypes)
    p_matrix = pd.DataFrame(np.zeros((n, n)), index=phenotypes, columns=phenotypes)
    se_matrix = pd.DataFrame(np.zeros((n, n)), index=phenotypes, columns=phenotypes)

    # 填充矩阵
    for _, row in rg_df.iterrows():
        p1, p2 = row['phenotype1'], row['phenotype2']
        rg_matrix.loc[p1, p2] = row['rg']
        rg_matrix.loc[p2, p1] = row['rg']
        p_matrix.loc[p1, p2] = row['p']
        p_matrix.loc[p2, p1] = row['p']
        se_matrix.loc[p1, p2] = row['rg_se']
        se_matrix.loc[p2, p1] = row['rg_se']

    return rg_matrix, p_matrix, se_matrix


def extract_heritability(rg_df):
    """从LDSC结果提取遗传力估计"""
    h2_data = {}

    # 从每对相关性中提取h2
    for _, row in rg_df.iterrows():
        p1, p2 = row['phenotype1'], row['phenotype2']

        if p1 not in h2_data:
            h2_data[p1] = {'h2': row['h2_p1'], 'se': row['h2_p1_se']}
        if p2 not in h2_data:
            h2_data[p2] = {'h2': row['h2_p2'], 'se': row['h2_p2_se']}

    h2_df = pd.DataFrame(h2_data).T
    h2_df.index.name = 'phenotype'
    h2_df = h2_df.reset_index()

    return h2_df


def plot_correlation_heatmap(rg_matrix, p_matrix):
    """绘制遗传相关性热图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建用于显示的标签
    labels = [PHENOTYPE_SHORT.get(p, p) for p in rg_matrix.index]

    # 创建mask（上三角，不包括对角线）
    mask = np.triu(np.ones_like(rg_matrix, dtype=bool), k=1)

    # 绘制热图
    sns.heatmap(rg_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Genetic Correlation (rg)', 'shrink': 0.8},
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)

    # 添加显著性标记
    for i in range(len(rg_matrix)):
        for j in range(i):  # 下三角
            p_val = p_matrix.iloc[i, j]
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            else:
                stars = ''

            if stars:
                # 在数值下方添加星号
                ax.text(j + 0.5, i + 0.75, stars,
                       ha='center', va='center',
                       fontsize=8, color='black')

    ax.set_title('Genetic Correlation Matrix of Pelvic Floor Disorders\n(LDSC)',
                 fontsize=14, fontweight='bold', pad=20)

    # 添加图例说明
    ax.text(1.02, -0.15, '*** p<0.001, ** p<0.01, * p<0.05',
            transform=ax.transAxes, fontsize=8, va='top')

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'ldsc_correlation_heatmap.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'ldsc_correlation_heatmap.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: ldsc_correlation_heatmap.png/pdf")


def plot_correlation_network(rg_df):
    """绘制遗传相关性网络图"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建图
    G = nx.Graph()

    # 添加节点
    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
    for p in phenotypes:
        G.add_node(p)

    # 添加边（只添加显著相关性）
    for _, row in rg_df.iterrows():
        if row['p'] < 0.05:  # 只显示显著的相关性
            G.add_edge(row['phenotype1'], row['phenotype2'],
                      weight=abs(row['rg']),
                      rg=row['rg'],
                      p=row['p'])

    # 使用spring布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 绘制边
    edges = G.edges(data=True)
    for (u, v, d) in edges:
        rg = d['rg']
        weight = abs(rg)
        color = '#E64B35' if rg > 0 else '#4DBBD5'  # 红色正相关，蓝色负相关
        alpha = min(1.0, 0.3 + weight * 0.7)
        width = 1 + weight * 8

        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
               color=color, alpha=alpha, linewidth=width, zorder=1)

        # 在边上标注rg值
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        ax.text(mid_x, mid_y, f'{rg:.2f}', fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 绘制节点
    node_sizes = 3000
    for node in G.nodes():
        color = PHENOTYPE_COLORS.get(node, '#888888')
        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                              node_color=[color],
                              node_size=node_sizes,
                              alpha=0.9,
                              ax=ax)

    # 添加节点标签
    labels = {p: PHENOTYPE_NAMES.get(p, p) for p in phenotypes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#E64B35', linewidth=4, label='Positive rg'),
        Line2D([0], [0], color='#4DBBD5', linewidth=4, label='Negative rg'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title('Genetic Correlation Network of Pelvic Floor Disorders\n(Edge width proportional to |rg|, only significant pairs shown)',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'ldsc_correlation_network.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'ldsc_correlation_network.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: ldsc_correlation_network.png/pdf")


def plot_heritability_bar(h2_df):
    """绘制遗传力条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 排序
    h2_df = h2_df.sort_values('h2', ascending=True)

    # 颜色
    colors = [PHENOTYPE_COLORS.get(p, '#888888') for p in h2_df['phenotype']]

    # 绘制条形图
    y_pos = np.arange(len(h2_df))
    bars = ax.barh(y_pos, h2_df['h2'], xerr=h2_df['se'],
                   color=colors, alpha=0.8,
                   error_kw=dict(ecolor='gray', capsize=3, capthick=1))

    # 设置y轴标签
    labels = [PHENOTYPE_SHORT.get(p, p) for p in h2_df['phenotype']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # 在条形上添加数值
    for i, (h2, se) in enumerate(zip(h2_df['h2'], h2_df['se'])):
        ax.text(h2 + se + 0.002, i, f'{h2:.3f}', va='center', fontsize=9)

    ax.set_xlabel('SNP-based Heritability (h²)', fontsize=12)
    ax.set_title('SNP Heritability Estimates for Pelvic Floor Disorders\n(LDSC)',
                fontsize=14, fontweight='bold')

    # 添加垂直参考线
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    ax.set_xlim(0, max(h2_df['h2'] + h2_df['se']) * 1.3)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'ldsc_heritability_bar.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'ldsc_heritability_bar.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: ldsc_heritability_bar.png/pdf")


def plot_correlation_clustered_heatmap(rg_matrix, p_matrix):
    """绘制聚类热图"""
    # 创建用于显示的标签
    labels = [PHENOTYPE_SHORT.get(p, p) for p in rg_matrix.index]

    # 创建注释矩阵（显著性星号）
    annot_matrix = rg_matrix.copy().astype(str)
    for i in range(len(rg_matrix)):
        for j in range(len(rg_matrix)):
            rg = rg_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            if i == j:
                annot_matrix.iloc[i, j] = '1.00'
            else:
                if p_val < 0.001:
                    annot_matrix.iloc[i, j] = f'{rg:.2f}***'
                elif p_val < 0.01:
                    annot_matrix.iloc[i, j] = f'{rg:.2f}**'
                elif p_val < 0.05:
                    annot_matrix.iloc[i, j] = f'{rg:.2f}*'
                else:
                    annot_matrix.iloc[i, j] = f'{rg:.2f}'

    # 创建clustermap
    g = sns.clustermap(rg_matrix,
                       annot=annot_matrix,
                       fmt='',
                       cmap='RdBu_r',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       linewidths=0.5,
                       figsize=(10, 8),
                       cbar_kws={'label': 'Genetic Correlation (rg)'},
                       xticklabels=labels,
                       yticklabels=labels)

    g.fig.suptitle('Hierarchically Clustered Genetic Correlation Matrix',
                   fontsize=14, fontweight='bold', y=1.02)

    # 保存
    g.savefig(FIGURES_DIR / 'ldsc_correlation_clustered.png', bbox_inches='tight')
    g.savefig(FIGURES_DIR / 'ldsc_correlation_clustered.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: ldsc_correlation_clustered.png/pdf")


def main():
    """主函数"""
    print("=" * 60)
    print("LDSC Genetic Correlation Visualization")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    rg_df = load_data()
    print(f"  Loaded {len(rg_df)} pairwise correlations")

    # 创建相关矩阵
    print("\n[2] Creating correlation matrix...")
    rg_matrix, p_matrix, se_matrix = create_correlation_matrix(rg_df)

    # 提取遗传力
    print("\n[3] Extracting heritability estimates...")
    h2_df = extract_heritability(rg_df)
    print(h2_df.to_string(index=False))

    # 生成可视化
    print("\n[4] Generating visualizations...")

    print("\n  4.1 Correlation heatmap...")
    plot_correlation_heatmap(rg_matrix, p_matrix)

    print("\n  4.2 Correlation network...")
    plot_correlation_network(rg_df)

    print("\n  4.3 Heritability bar chart...")
    plot_heritability_bar(h2_df)

    print("\n  4.4 Clustered heatmap...")
    plot_correlation_clustered_heatmap(rg_matrix, p_matrix)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
