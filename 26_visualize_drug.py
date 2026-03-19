#!/usr/bin/env python3
"""
26_visualize_drug.py - 药物重定位结果可视化

生成图表:
1. 药物-基因相互作用网络图
2. Top候选药物优先级条形图
3. 药物类别分布图
4. 表型-药物热图

Author: Claude
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from matplotlib.patches import Patch
from collections import Counter
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
RESULTS_DIR = BASE_DIR / "results" / "drug_repurposing"
FIGURES_DIR = BASE_DIR / "figures" / "drug_repurposing"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
PHENOTYPE_COLORS = {
    'POP': '#E64B35',
    'BPH': '#4DBBD5',
    'Bladder': '#00A087',
    'Constipation': '#3C5488',
    'FemaleProlapse': '#F39B7F',
    'Incontinence': '#8491B4'
}

INTERACTION_COLORS = {
    'inhibitor': '#E64B35',
    'agonist': '#4DBBD5',
    'antagonist': '#00A087',
    'modulator': '#3C5488',
    'other': '#888888'
}


def load_data():
    """加载药物重定位结果"""
    prioritized = pd.read_csv(RESULTS_DIR / "prioritized_candidates.csv")

    # 尝试加载完整的相互作用数据
    interactions_file = RESULTS_DIR / "dgidb_interactions.csv"
    if interactions_file.exists():
        interactions = pd.read_csv(interactions_file)
    else:
        interactions = None

    return prioritized, interactions


def plot_drug_gene_network(prioritized):
    """绘制药物-基因相互作用网络图"""
    fig, ax = plt.subplots(figsize=(16, 12))

    # 创建图
    G = nx.Graph()

    # 获取top药物（按优先级分数）
    top_drugs = prioritized.nlargest(30, 'priority_score')

    # 添加基因节点
    genes = top_drugs['gene_symbol'].unique()
    for gene in genes:
        G.add_node(gene, node_type='gene')

    # 添加药物节点和边
    for _, row in top_drugs.iterrows():
        drug = row['drug']
        gene = row['gene_symbol']
        G.add_node(drug, node_type='drug', interaction=row.get('interaction_type', 'other'))
        G.add_edge(drug, gene, weight=row['priority_score'])

    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 分离基因和药物节点
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'gene']
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'drug']

    # 绘制基因节点（正方形）
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes,
                          node_color='#E64B35', node_size=1000,
                          node_shape='s', alpha=0.9, ax=ax)

    # 绘制药物节点（圆形）
    drug_colors = []
    for node in drug_nodes:
        interaction = G.nodes[node].get('interaction', 'other')
        if pd.isna(interaction):
            interaction = 'other'
        drug_colors.append(INTERACTION_COLORS.get(interaction, '#888888'))

    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes,
                          node_color=drug_colors, node_size=500,
                          node_shape='o', alpha=0.8, ax=ax)

    # 绘制边
    edges = G.edges(data=True)
    edge_widths = [d.get('weight', 1) / 5 for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_widths, ax=ax)

    # 标签
    gene_labels = {n: n for n in gene_nodes}
    drug_labels = {n: n[:15] + '...' if len(n) > 15 else n for n in drug_nodes}

    nx.draw_networkx_labels(G, pos, gene_labels, font_size=9, font_weight='bold', ax=ax)
    nx.draw_networkx_labels(G, pos, drug_labels, font_size=7, ax=ax)

    # 图例
    legend_elements = [
        Patch(facecolor='#E64B35', label='Gene (target)', alpha=0.9),
        Patch(facecolor=INTERACTION_COLORS['inhibitor'], label='Inhibitor'),
        Patch(facecolor=INTERACTION_COLORS['agonist'], label='Agonist'),
        Patch(facecolor=INTERACTION_COLORS['other'], label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title('Drug-Gene Interaction Network\n(Top 30 Drug Candidates by Priority Score)',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'drug_gene_network.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'drug_gene_network.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: drug_gene_network.png/pdf")


def plot_priority_bar(prioritized):
    """绘制Top候选药物优先级条形图"""
    # 获取每个药物的最高优先级分数（去重）
    drug_priority = prioritized.groupby('drug').agg({
        'priority_score': 'max',
        'gene_symbol': lambda x: ', '.join(x.unique()[:3]),  # 最多3个基因
        'phenotypes': 'first',
        'interaction_type': 'first'
    }).reset_index()

    drug_priority = drug_priority.nlargest(25, 'priority_score')

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(drug_priority))

    # 颜色根据相互作用类型
    colors = []
    for itype in drug_priority['interaction_type']:
        if pd.isna(itype):
            colors.append('#888888')
        else:
            colors.append(INTERACTION_COLORS.get(itype, '#888888'))

    bars = ax.barh(y_pos, drug_priority['priority_score'],
                   color=colors, alpha=0.8, edgecolor='white')

    # 设置y轴标签（药物名称）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(drug_priority['drug'])
    ax.invert_yaxis()

    # 在条形末端添加靶基因
    for i, (score, genes) in enumerate(zip(drug_priority['priority_score'],
                                           drug_priority['gene_symbol'])):
        ax.text(score + 0.3, i, f'→ {genes}', va='center', fontsize=8, alpha=0.8)

    ax.set_xlabel('Priority Score', fontsize=12)
    ax.set_title('Top 25 Drug Repurposing Candidates\n(Ranked by Priority Score)',
                fontsize=14, fontweight='bold')

    # 图例
    legend_elements = [
        Patch(facecolor=INTERACTION_COLORS['inhibitor'], label='Inhibitor'),
        Patch(facecolor=INTERACTION_COLORS['agonist'], label='Agonist'),
        Patch(facecolor=INTERACTION_COLORS['other'], label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'drug_priority_bar.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'drug_priority_bar.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: drug_priority_bar.png/pdf")


def plot_interaction_type_pie(prioritized):
    """绘制相互作用类型分布饼图"""
    # 统计相互作用类型
    interaction_counts = prioritized['interaction_type'].fillna('unknown').value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：饼图
    ax1 = axes[0]
    colors = [INTERACTION_COLORS.get(t, '#888888') for t in interaction_counts.index]

    wedges, texts, autotexts = ax1.pie(interaction_counts.values,
                                        labels=interaction_counts.index,
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        pctdistance=0.8)
    ax1.set_title('Drug-Gene Interaction Types', fontsize=12, fontweight='bold')

    # 右图：按表型的药物数量
    ax2 = axes[1]

    # 统计每个表型的药物数量
    phenotype_counts = {}
    for _, row in prioritized.iterrows():
        phenotypes = str(row['phenotypes']).split(', ')
        for p in phenotypes:
            p = p.strip()
            if p and p != 'nan':
                phenotype_counts[p] = phenotype_counts.get(p, 0) + 1

    if phenotype_counts:
        phenotypes = list(phenotype_counts.keys())
        counts = [phenotype_counts[p] for p in phenotypes]
        colors = [PHENOTYPE_COLORS.get(p, '#888888') for p in phenotypes]

        bars = ax2.barh(phenotypes, counts, color=colors, alpha=0.8)
        ax2.set_xlabel('Number of Drug-Gene Interactions', fontsize=12)
        ax2.set_title('Drug Candidates by Target Phenotype', fontsize=12, fontweight='bold')

        # 添加数值
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontsize=10)

    plt.suptitle('Drug Repurposing Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'drug_interaction_distribution.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'drug_interaction_distribution.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: drug_interaction_distribution.png/pdf")


def plot_gene_drug_heatmap(prioritized):
    """绘制基因-药物热图"""
    # 获取top基因和药物
    top_genes = prioritized.groupby('gene_symbol')['priority_score'].sum().nlargest(15).index
    top_drugs = prioritized.groupby('drug')['priority_score'].sum().nlargest(20).index

    # 创建热图数据
    heatmap_data = pd.DataFrame(0, index=top_genes, columns=top_drugs)

    for _, row in prioritized.iterrows():
        if row['gene_symbol'] in top_genes and row['drug'] in top_drugs:
            heatmap_data.loc[row['gene_symbol'], row['drug']] = row['priority_score']

    # 只保留有值的列
    heatmap_data = heatmap_data.loc[:, (heatmap_data != 0).any(axis=0)]

    if heatmap_data.empty or len(heatmap_data.columns) == 0:
        print("  Warning: No data for heatmap")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(heatmap_data,
                cmap='YlOrRd',
                linewidths=0.5,
                cbar_kws={'label': 'Priority Score'},
                ax=ax)

    ax.set_xlabel('Drug', fontsize=12)
    ax.set_ylabel('Target Gene', fontsize=12)
    ax.set_title('Gene-Drug Interaction Heatmap\n(Top Genes and Drugs by Priority Score)',
                fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'drug_gene_heatmap.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'drug_gene_heatmap.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: drug_gene_heatmap.png/pdf")


def plot_source_distribution(prioritized):
    """绘制数据来源分布图"""
    # 统计来源
    all_sources = []
    for sources in prioritized['sources'].dropna():
        source_list = str(sources).split(', ')
        all_sources.extend([s.strip()[:30] for s in source_list])  # 截断长名称

    source_counts = Counter(all_sources)
    top_sources = dict(sorted(source_counts.items(), key=lambda x: -x[1])[:15])

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top_sources))
    bars = ax.barh(y_pos, list(top_sources.values()),
                   color='#3C5488', alpha=0.8, edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(top_sources.keys()))
    ax.invert_yaxis()

    ax.set_xlabel('Number of Interactions', fontsize=12)
    ax.set_title('Drug-Gene Interaction Sources (Top 15)',
                fontsize=14, fontweight='bold')

    # 添加数值
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{int(width)}', va='center', fontsize=9)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'drug_source_distribution.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'drug_source_distribution.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: drug_source_distribution.png/pdf")


def generate_summary(prioritized):
    """生成汇总统计"""
    summary = {
        'Total drug-gene interactions': len(prioritized),
        'Unique drugs': prioritized['drug'].nunique(),
        'Unique target genes': prioritized['gene_symbol'].nunique(),
        'Max priority score': prioritized['priority_score'].max(),
        'Mean priority score': prioritized['priority_score'].mean(),
    }

    # 按相互作用类型
    interaction_counts = prioritized['interaction_type'].fillna('unknown').value_counts()
    for itype, count in interaction_counts.items():
        summary[f'Interaction type - {itype}'] = count

    print("\n  Summary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.2f}")
        else:
            print(f"    {key}: {value}")

    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("Drug Repurposing Visualization")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    prioritized, interactions = load_data()
    print(f"  Prioritized candidates: {len(prioritized)}")
    if interactions is not None:
        print(f"  Total interactions: {len(interactions)}")

    # 生成汇总
    print("\n[2] Computing summary...")
    summary = generate_summary(prioritized)

    # 生成可视化
    print("\n[3] Generating visualizations...")

    print("\n  3.1 Drug-gene network...")
    plot_drug_gene_network(prioritized)

    print("\n  3.2 Priority bar chart...")
    plot_priority_bar(prioritized)

    print("\n  3.3 Interaction type distribution...")
    plot_interaction_type_pie(prioritized)

    print("\n  3.4 Gene-drug heatmap...")
    plot_gene_drug_heatmap(prioritized)

    print("\n  3.5 Source distribution...")
    plot_source_distribution(prioritized)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
