#!/usr/bin/env python3
"""
32_ppi_network_analysis.py - PPI网络/模块分析

使用STRING数据库进行蛋白质相互作用网络分析:
1. 从MAGMA显著基因获取PPI网络
2. 网络拓扑分析（度、中心性）
3. 社区/模块检测
4. 模块功能富集分析

Author: Claude
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import requests
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# 路径设置
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "ppi_network"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = BASE_DIR / "figures" / "ppi_network"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# MAGMA结果
MAGMA_DIR = BASE_DIR / "results" / "magma"

# 表型颜色
PHENOTYPE_COLORS = {
    'POP': '#E64B35',
    'BPH': '#4DBBD5',
    'Bladder': '#00A087',
    'Constipation': '#3C5488',
    'FemaleProlapse': '#F39B7F',
    'Incontinence': '#8491B4'
}


def load_magma_genes():
    """加载MAGMA显著基因（使用基因符号）"""
    print("  Loading MAGMA significant genes...")

    # 使用top_genes文件获取基因符号
    top_genes_file = MAGMA_DIR / "magma_top_genes.csv"
    all_genes = {}
    gene_phenotype_map = {}

    if top_genes_file.exists():
        top_df = pd.read_csv(top_genes_file)
        print(f"    Loaded {len(top_df)} top genes from magma_top_genes.csv")

        for _, row in top_df.iterrows():
            gene_symbol = row['Symbol']
            pheno = row['Phenotype']

            if gene_symbol not in all_genes:
                all_genes[gene_symbol] = {'P': row['P'], 'Z': row['Z'], 'phenotypes': []}
            if pheno not in all_genes[gene_symbol]['phenotypes']:
                all_genes[gene_symbol]['phenotypes'].append(pheno)
            gene_phenotype_map[gene_symbol] = gene_phenotype_map.get(gene_symbol, []) + [pheno]

    # 还可以从完整的MAGMA结果中获取更多基因符号
    # 但由于MAGMA输出只有Entrez ID，我们需要映射
    # 暂时先使用top_genes

    gene_df = pd.DataFrame([
        {'Gene': g, 'P': v['P'], 'Z': v['Z'], 'n_phenotypes': len(set(v['phenotypes'])),
         'phenotypes': ','.join(set(v['phenotypes']))}
        for g, v in all_genes.items()
    ])

    print(f"    Total genes with symbols: {len(gene_df)}")
    print(f"    Multi-phenotype genes: {(gene_df['n_phenotypes'] > 1).sum()}")

    return gene_df, gene_phenotype_map


def get_string_network(genes, score_threshold=400):
    """从STRING数据库获取PPI网络"""
    print(f"  Fetching PPI network from STRING for {len(genes)} genes...")

    string_api_url = "https://string-db.org/api"

    # 分批查询（STRING API限制）
    batch_size = 200
    all_interactions = []

    gene_list = [str(g) for g in genes]  # 确保是字符串

    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i+batch_size]

        params = {
            "identifiers": "%0d".join(batch),
            "species": 9606,  # Human
            "caller_identity": "pelvic_floor_gwas",
            "required_score": score_threshold
        }

        try:
            # 获取网络
            response = requests.post(
                f"{string_api_url}/json/network",
                data=params,
                timeout=60
            )

            if response.status_code == 200:
                interactions = response.json()
                all_interactions.extend(interactions)
                print(f"    Batch {i//batch_size + 1}: {len(interactions)} interactions")
            else:
                print(f"    Warning: STRING API returned {response.status_code}")

        except Exception as e:
            print(f"    Warning: Error fetching batch {i//batch_size + 1}: {e}")

        time.sleep(1)  # Rate limiting

    print(f"    Total interactions: {len(all_interactions)}")
    return all_interactions


def build_network(interactions, gene_df):
    """构建NetworkX图"""
    print("  Building network graph...")

    G = nx.Graph()

    # 添加节点（基因符号）
    input_genes = set(gene_df['Gene'].str.upper())  # 大写以便匹配
    for gene in gene_df['Gene']:
        G.add_node(gene)

    # 调试：打印第一个交互的结构
    if interactions:
        print(f"    Sample interaction keys: {list(interactions[0].keys())[:10]}")

    # 添加边
    edges_added = 0
    gene_pairs_found = set()

    for interaction in interactions:
        # STRING API可能返回不同的键名
        gene1 = interaction.get('preferredName_A') or interaction.get('preferredName')
        gene2 = interaction.get('preferredName_B') or interaction.get('preferredName_B')

        if gene1 is None or gene2 is None:
            # 尝试其他可能的键
            gene1 = interaction.get('stringId_A', '').split('.')[-1] if '.' in str(interaction.get('stringId_A', '')) else interaction.get('stringId_A', '')
            gene2 = interaction.get('stringId_B', '').split('.')[-1] if '.' in str(interaction.get('stringId_B', '')) else interaction.get('stringId_B', '')

        score = interaction.get('score', 0)

        # 大小写不敏感匹配
        if gene1 and gene2:
            gene_pairs_found.add((gene1, gene2))
            # 查找匹配的节点
            gene1_match = None
            gene2_match = None
            for node in G.nodes():
                if node.upper() == str(gene1).upper():
                    gene1_match = node
                if node.upper() == str(gene2).upper():
                    gene2_match = node

            if gene1_match and gene2_match and gene1_match != gene2_match:
                G.add_edge(gene1_match, gene2_match, weight=score)
                edges_added += 1

    print(f"    Gene pairs from STRING: {len(gene_pairs_found)}")
    print(f"    Edges successfully added: {edges_added}")

    # 移除孤立节点
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    print(f"    Final nodes: {G.number_of_nodes()}")
    print(f"    Final edges: {G.number_of_edges()}")
    print(f"    Isolated nodes removed: {len(isolated)}")

    return G


def calculate_network_metrics(G):
    """计算网络拓扑指标"""
    print("  Calculating network metrics...")

    metrics = {}

    # 度
    degrees = dict(G.degree())
    metrics['degree'] = degrees

    # 介数中心性
    if G.number_of_nodes() > 0:
        betweenness = nx.betweenness_centrality(G)
        metrics['betweenness'] = betweenness

        # 接近中心性
        closeness = nx.closeness_centrality(G)
        metrics['closeness'] = closeness

        # 特征向量中心性
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            metrics['eigenvector'] = eigenvector
        except:
            metrics['eigenvector'] = {n: 0 for n in G.nodes()}

        # 聚类系数
        clustering = nx.clustering(G)
        metrics['clustering'] = clustering
    else:
        for m in ['betweenness', 'closeness', 'eigenvector', 'clustering']:
            metrics[m] = {}

    # 创建汇总DataFrame
    nodes = list(G.nodes())
    metrics_df = pd.DataFrame({
        'Gene': nodes,
        'Degree': [metrics['degree'].get(n, 0) for n in nodes],
        'Betweenness': [metrics['betweenness'].get(n, 0) for n in nodes],
        'Closeness': [metrics['closeness'].get(n, 0) for n in nodes],
        'Eigenvector': [metrics['eigenvector'].get(n, 0) for n in nodes],
        'Clustering': [metrics['clustering'].get(n, 0) for n in nodes]
    })

    return metrics_df


def detect_communities(G):
    """检测网络社区/模块"""
    print("  Detecting network communities...")

    if G.number_of_nodes() == 0:
        return {}, []

    # 使用Louvain算法
    try:
        from networkx.algorithms import community
        communities = community.louvain_communities(G, seed=42)
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i

        print(f"    Found {len(communities)} communities")
        for i, comm in enumerate(communities):
            print(f"      Community {i+1}: {len(comm)} genes")

        return community_map, communities
    except Exception as e:
        print(f"    Warning: Community detection failed: {e}")
        return {n: 0 for n in G.nodes()}, [set(G.nodes())]


def functional_enrichment(genes, background_size=20000):
    """简单的功能富集分析（使用STRING的富集API）"""
    print("  Running functional enrichment...")

    if len(genes) == 0:
        return pd.DataFrame()

    string_api_url = "https://string-db.org/api"

    params = {
        "identifiers": "%0d".join(genes),
        "species": 9606,
        "caller_identity": "pelvic_floor_gwas"
    }

    try:
        response = requests.post(
            f"{string_api_url}/json/enrichment",
            data=params,
            timeout=60
        )

        if response.status_code == 200:
            enrichment = response.json()
            if enrichment:
                enrich_df = pd.DataFrame(enrichment)
                if len(enrich_df) > 0:
                    enrich_df = enrich_df.sort_values('fdr', ascending=True)
                    return enrich_df
        else:
            print(f"    Warning: Enrichment API returned {response.status_code}")

    except Exception as e:
        print(f"    Warning: Enrichment analysis failed: {e}")

    return pd.DataFrame()


def create_visualizations(G, metrics_df, community_map, communities, gene_phenotype_map, enrichment_df):
    """生成可视化图表"""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300

    if G.number_of_nodes() == 0:
        print("    Warning: Empty network, skipping visualizations")
        return

    # 1. 网络图
    fig, ax = plt.subplots(figsize=(14, 14))

    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 节点颜色（按社区）
    node_colors = [community_map.get(n, 0) for n in G.nodes()]
    cmap = plt.cm.Set3

    # 节点大小（按度）
    degrees = dict(G.degree())
    node_sizes = [100 + degrees.get(n, 0) * 50 for n in G.nodes()]

    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                    node_size=node_sizes, cmap=cmap, alpha=0.8, ax=ax)

    # 只标注高度节点
    top_nodes = metrics_df.nlargest(15, 'Degree')['Gene'].tolist()
    labels = {n: n for n in top_nodes if n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(f'PPI Network of MAGMA Significant Genes\n(Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'ppi_network_graph.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'ppi_network_graph.pdf', bbox_inches='tight')
    plt.close()

    # 2. 度分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    degree_values = list(degrees.values())
    sns.histplot(degree_values, bins=30, kde=True, ax=ax1, color='#3C5488')
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Degree Distribution', fontsize=12, fontweight='bold')

    # 3. Hub基因条形图
    ax2 = axes[1]
    top_hubs = metrics_df.nlargest(20, 'Degree')
    y_pos = np.arange(len(top_hubs))
    ax2.barh(y_pos, top_hubs['Degree'], color='#E64B35', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_hubs['Gene'])
    ax2.invert_yaxis()
    ax2.set_xlabel('Degree', fontsize=12)
    ax2.set_title('Top 20 Hub Genes', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'degree_distribution.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'degree_distribution.pdf', bbox_inches='tight')
    plt.close()

    # 4. 中心性比较
    fig, ax = plt.subplots(figsize=(10, 10))

    if len(metrics_df) > 0:
        scatter = ax.scatter(metrics_df['Degree'], metrics_df['Betweenness'],
                            c=metrics_df['Closeness'], cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Closeness Centrality')

        # 标注top基因
        top_genes = metrics_df.nlargest(10, 'Betweenness')
        for _, row in top_genes.iterrows():
            ax.annotate(row['Gene'], (row['Degree'], row['Betweenness']),
                       fontsize=8, alpha=0.8)

    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Betweenness Centrality', fontsize=12)
    ax.set_title('Network Centrality Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'centrality_comparison.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'centrality_comparison.pdf', bbox_inches='tight')
    plt.close()

    # 5. 社区/模块大小
    if communities:
        fig, ax = plt.subplots(figsize=(10, 6))
        comm_sizes = [len(c) for c in communities]
        x = range(1, len(comm_sizes) + 1)
        bars = ax.bar(x, comm_sizes, color=plt.cm.Set3(np.linspace(0, 1, len(comm_sizes))))
        ax.set_xlabel('Community', fontsize=12)
        ax.set_ylabel('Number of Genes', fontsize=12)
        ax.set_title('Network Community Sizes', fontsize=14, fontweight='bold')

        for bar, size in zip(bars, comm_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(size), ha='center', fontsize=10)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'community_sizes.png', bbox_inches='tight')
        fig.savefig(FIGURES_DIR / 'community_sizes.pdf', bbox_inches='tight')
        plt.close()

    # 6. 功能富集点图
    if len(enrichment_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))

        top_terms = enrichment_df.head(15)
        if 'description' in top_terms.columns and 'fdr' in top_terms.columns:
            y_pos = np.arange(len(top_terms))
            colors = -np.log10(top_terms['fdr'].values + 1e-300)

            scatter = ax.scatter(-np.log10(top_terms['fdr'] + 1e-300), y_pos,
                               c=colors, cmap='YlOrRd', s=100, alpha=0.8)

            ax.set_yticks(y_pos)
            labels = [d[:50] + '...' if len(d) > 50 else d for d in top_terms['description']]
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel('-log10(FDR)', fontsize=12)
            ax.set_title('Functional Enrichment of Network Genes',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()
            fig.savefig(FIGURES_DIR / 'functional_enrichment.png', bbox_inches='tight')
            fig.savefig(FIGURES_DIR / 'functional_enrichment.pdf', bbox_inches='tight')
            plt.close()

    print(f"  Saved visualizations to {FIGURES_DIR}")


def write_log(G, metrics_df, communities, enrichment_df, gene_df):
    """写入分析日志"""
    log_file = BASE_DIR / "logs" / "14_ppi_network.md"

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("# Log 14: PPI Network Analysis\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write("**Status**: Completed\n\n")
        f.write("---\n\n")

        f.write("## Objectives\n\n")
        f.write("1. Construct PPI network from MAGMA significant genes\n")
        f.write("2. Identify hub genes (high connectivity)\n")
        f.write("3. Detect functional modules/communities\n")
        f.write("4. Perform pathway enrichment analysis\n\n")

        f.write("---\n\n")

        f.write("## Methods\n\n")
        f.write("### Data Sources\n")
        f.write("- **Genes**: MAGMA significant genes (p < 1e-4)\n")
        f.write("- **PPI Database**: STRING v12 (score > 400)\n")
        f.write("- **Enrichment**: STRING functional enrichment API\n\n")

        f.write("### Network Analysis\n")
        f.write("- **Graph Library**: NetworkX\n")
        f.write("- **Community Detection**: Louvain algorithm\n")
        f.write("- **Centrality Measures**: Degree, Betweenness, Closeness, Eigenvector\n\n")

        f.write("---\n\n")

        f.write("## Results\n\n")

        f.write("### Network Statistics\n")
        f.write(f"- Input genes: {len(gene_df)}\n")
        f.write(f"- Network nodes: {G.number_of_nodes()}\n")
        f.write(f"- Network edges: {G.number_of_edges()}\n")

        if G.number_of_nodes() > 0:
            density = nx.density(G)
            f.write(f"- Network density: {density:.4f}\n")

            if nx.is_connected(G):
                f.write(f"- Average path length: {nx.average_shortest_path_length(G):.2f}\n")
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subG = G.subgraph(largest_cc)
                f.write(f"- Largest component: {len(largest_cc)} nodes\n")

            f.write(f"- Average clustering: {nx.average_clustering(G):.4f}\n")
        f.write("\n")

        f.write("### Top Hub Genes\n")
        f.write("| Rank | Gene | Degree | Betweenness | Phenotypes |\n")
        f.write("|------|------|--------|-------------|------------|\n")
        top_hubs = metrics_df.nlargest(15, 'Degree')
        for i, (_, row) in enumerate(top_hubs.iterrows()):
            gene = row['Gene']
            phenos = gene_df[gene_df['Gene'] == gene]['phenotypes'].values
            pheno_str = phenos[0] if len(phenos) > 0 else ''
            f.write(f"| {i+1} | {gene} | {row['Degree']} | {row['Betweenness']:.4f} | {pheno_str[:30]} |\n")
        f.write("\n")

        f.write("### Network Communities\n")
        f.write(f"- Total communities: {len(communities)}\n\n")
        for i, comm in enumerate(communities[:5]):
            f.write(f"**Community {i+1}** ({len(comm)} genes):\n")
            genes_list = list(comm)[:10]
            f.write(f"- {', '.join(genes_list)}")
            if len(comm) > 10:
                f.write(f"... (+{len(comm)-10} more)")
            f.write("\n\n")

        if len(enrichment_df) > 0:
            f.write("### Top Enriched Pathways\n")
            f.write("| Term | Category | FDR | Genes |\n")
            f.write("|------|----------|-----|-------|\n")
            for _, row in enrichment_df.head(10).iterrows():
                desc = row.get('description', 'N/A')[:40]
                cat = row.get('category', 'N/A')
                fdr = row.get('fdr', 1)
                n_genes = row.get('number_of_genes', 0)
                f.write(f"| {desc} | {cat} | {fdr:.2e} | {n_genes} |\n")
            f.write("\n")

        f.write("---\n\n")

        f.write("## Output Files\n\n")
        f.write("```\n")
        f.write("results/ppi_network/\n")
        f.write("├── ppi_network.graphml           # Network in GraphML format\n")
        f.write("├── network_metrics.csv           # Node centrality metrics\n")
        f.write("├── community_assignments.csv     # Community membership\n")
        f.write("├── hub_genes.csv                 # Top hub genes\n")
        f.write("└── functional_enrichment.csv     # Pathway enrichment\n")
        f.write("```\n\n")

        f.write("---\n\n")

        f.write("## Conclusions\n\n")
        if G.number_of_nodes() > 0:
            f.write(f"1. Successfully constructed PPI network with {G.number_of_nodes()} genes\n")
            f.write(f"2. Identified {len(communities)} functional communities\n")
            f.write("3. Hub genes represent potential key regulators of pelvic floor biology\n")
        else:
            f.write("1. Limited network connections found - genes may act through non-direct mechanisms\n")

    print(f"  Log saved to: {log_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("PPI Network Analysis")
    print("=" * 60)

    # 加载MAGMA基因
    print("\n[1] Loading MAGMA significant genes...")
    gene_df, gene_phenotype_map = load_magma_genes()

    # 获取STRING网络
    print("\n[2] Fetching PPI network from STRING...")
    genes = gene_df['Gene'].tolist()
    interactions = get_string_network(genes)

    # 构建网络
    print("\n[3] Building network graph...")
    G = build_network(interactions, gene_df)

    # 计算网络指标
    print("\n[4] Calculating network metrics...")
    metrics_df = calculate_network_metrics(G)

    # 检测社区
    print("\n[5] Detecting communities...")
    community_map, communities = detect_communities(G)

    # 功能富集
    print("\n[6] Running functional enrichment...")
    if G.number_of_nodes() > 0:
        enrichment_df = functional_enrichment(list(G.nodes()))
    else:
        enrichment_df = pd.DataFrame()

    # 保存结果
    print("\n[7] Saving results...")

    # 保存网络
    if G.number_of_nodes() > 0:
        nx.write_graphml(G, str(RESULTS_DIR / "ppi_network.graphml"))

    # 保存指标
    metrics_df.to_csv(RESULTS_DIR / "network_metrics.csv", index=False)

    # 保存社区
    community_df = pd.DataFrame([
        {'Gene': gene, 'Community': comm}
        for gene, comm in community_map.items()
    ])
    community_df.to_csv(RESULTS_DIR / "community_assignments.csv", index=False)

    # 保存hub基因
    hub_genes = metrics_df.nlargest(50, 'Degree')
    hub_genes.to_csv(RESULTS_DIR / "hub_genes.csv", index=False)

    # 保存富集结果
    if len(enrichment_df) > 0:
        enrichment_df.to_csv(RESULTS_DIR / "functional_enrichment.csv", index=False)

    print(f"  Results saved to: {RESULTS_DIR}")

    # 生成可视化
    print("\n[8] Generating visualizations...")
    create_visualizations(G, metrics_df, community_map, communities,
                         gene_phenotype_map, enrichment_df)

    # 写入日志
    print("\n[9] Writing analysis log...")
    write_log(G, metrics_df, communities, enrichment_df, gene_df)

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
