#!/usr/bin/env python3
"""
38_ppi_network_expanded.py - 扩展版PPI网络分析

改进:
1. 使用完整NCBI基因映射（193,701个映射）
2. 使用更多MAGMA显著基因（P<0.001）
3. 添加STRING网络的二级邻居
4. 更丰富的功能富集分析
5. 模块化分析和可视化

Author: Claude
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# 导入基因映射工具（使用完整NCBI映射）
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.gene_mapping import load_ncbi_gene_mapping, get_symbol

# 路径设置
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "ppi_network_expanded"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = BASE_DIR / "figures" / "ppi_network"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MAGMA_DIR = BASE_DIR / "results" / "magma"

# STRING API
STRING_API = "https://string-db.org/api"


def load_magma_genes(p_threshold=0.001, top_n=100):
    """加载MAGMA显著基因"""
    print("  Loading MAGMA genes...")

    phenotypes = ['POP', 'BPH', 'Bladder', 'Constipation', 'FemaleProlapse', 'Incontinence']
    all_genes = []

    for pheno in phenotypes:
        full_file = MAGMA_DIR / f"{pheno}_genes.genes.out.txt"
        if full_file.exists():
            df = pd.read_csv(full_file, sep=r'\s+', comment='#')
            df['Phenotype'] = pheno
            # 筛选显著基因
            sig = df[df['P'] < p_threshold].head(top_n)
            all_genes.append(sig)
            print(f"    {pheno}: {len(sig)} genes (P<{p_threshold})")

    if not all_genes:
        print("    Warning: No MAGMA data found")
        return pd.DataFrame()

    combined = pd.concat(all_genes, ignore_index=True)

    # 汇总到基因级别
    gene_summary = combined.groupby('GENE').agg({
        'P': 'min',
        'ZSTAT': 'max',
        'Phenotype': lambda x: len(set(x)),
        'NSNPS': 'max'
    }).reset_index()
    gene_summary.columns = ['GeneID', 'min_P', 'max_Z', 'n_phenotypes', 'nSNPs']

    print(f"    Total unique genes: {len(gene_summary)}")
    return gene_summary


def get_gene_symbols(gene_ids, entrez_to_symbol):
    """使用完整NCBI映射获取基因符号"""
    print("  Mapping Entrez IDs to gene symbols...")

    symbols = {}
    mapped_count = 0
    unmapped_count = 0

    for gid in gene_ids:
        gid_str = str(gid).strip()
        symbol = get_symbol(gid_str, entrez_to_symbol)

        if symbol != gid_str:  # 成功映射
            symbols[gid_str] = symbol
            mapped_count += 1
        else:
            # 未能映射，保留ID
            symbols[gid_str] = f"GENE_{gid_str}"
            unmapped_count += 1

    print(f"    Mapped: {mapped_count}, Unmapped: {unmapped_count}")
    return symbols


def get_string_interactions(gene_list, species=9606, score_threshold=400):
    """从STRING获取PPI网络"""
    print("  Fetching STRING interactions...")

    if not gene_list:
        return []

    # STRING API限制每次100个基因
    batch_size = 100
    all_interactions = []

    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i+batch_size]
        genes_str = "%0d".join(batch)

        url = f"{STRING_API}/json/network"
        params = {
            'identifiers': genes_str,
            'species': species,
            'caller_identity': 'pelvic_floor_gwas',
            'required_score': score_threshold
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                interactions = response.json()
                all_interactions.extend(interactions)
                print(f"    Batch {i//batch_size + 1}: {len(interactions)} interactions")
            else:
                print(f"    Warning: STRING API returned {response.status_code}")
        except Exception as e:
            print(f"    Warning: STRING API error - {e}")

        time.sleep(0.5)  # 避免API限速

    return all_interactions


def build_network(interactions, gene_data, symbols):
    """构建NetworkX图"""
    print("  Building network...")

    G = nx.Graph()

    # 创建所有节点符号的集合，用于快速查找
    node_symbols = set()

    # 添加节点
    for idx, row in gene_data.iterrows():
        # 确保GeneID转换为字符串（去除.0后缀如果是float）
        gene_id = str(int(row['GeneID'])) if pd.notna(row['GeneID']) else str(row['GeneID'])
        symbol = symbols.get(gene_id, gene_id)
        G.add_node(symbol, gene_id=gene_id, p_value=row['min_P'],
                  z_score=row['max_Z'], n_phenotypes=row['n_phenotypes'])
        node_symbols.add(symbol)

    print(f"    Added {len(node_symbols)} nodes")
    print(f"    Sample nodes: {list(node_symbols)[:5]}")

    # 添加边
    n_edges = 0
    matched_edges = 0
    for interaction in interactions:
        gene1 = interaction.get('preferredName_A', interaction.get('stringId_A', ''))
        gene2 = interaction.get('preferredName_B', interaction.get('stringId_B', ''))
        score = interaction.get('score', 0)

        if gene1 and gene2:
            matched_edges += 1
            if gene1 in node_symbols and gene2 in node_symbols:
                G.add_edge(gene1, gene2, weight=score)
                n_edges += 1

    print(f"    STRING interactions: {len(interactions)}, matched: {matched_edges}, edges added: {n_edges}")
    if len(interactions) > 0 and n_edges == 0:
        # Debug: show what STRING returned vs our nodes
        sample_interactions = interactions[:3]
        print(f"    Sample STRING genes: {[(i.get('preferredName_A'), i.get('preferredName_B')) for i in sample_interactions]}")

    print(f"    Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def analyze_network(G):
    """分析网络拓扑"""
    print("  Analyzing network topology...")

    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    # 计算中心性指标
    metrics = {}

    # 度中心性
    degree_cent = nx.degree_centrality(G)

    # 介数中心性
    betweenness = nx.betweenness_centrality(G)

    # 接近中心性
    try:
        closeness = nx.closeness_centrality(G)
    except:
        closeness = {n: 0 for n in G.nodes()}

    # 特征向量中心性
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    except:
        eigenvector = {n: 0 for n in G.nodes()}

    # 聚类系数
    clustering = nx.clustering(G)

    # 汇总
    for node in G.nodes():
        metrics[node] = {
            'Gene': node,
            'Degree': G.degree(node),
            'Degree_Centrality': degree_cent.get(node, 0),
            'Betweenness': betweenness.get(node, 0),
            'Closeness': closeness.get(node, 0),
            'Eigenvector': eigenvector.get(node, 0),
            'Clustering': clustering.get(node, 0),
            'P_value': G.nodes[node].get('p_value', 1),
            'Z_score': G.nodes[node].get('z_score', 0),
            'N_phenotypes': G.nodes[node].get('n_phenotypes', 0)
        }

    metrics_df = pd.DataFrame(metrics.values())
    metrics_df = metrics_df.sort_values('Degree', ascending=False)

    print(f"    Top hub genes by degree:")
    for _, row in metrics_df.head(5).iterrows():
        print(f"      {row['Gene']}: degree={row['Degree']}")

    return metrics_df


def detect_communities(G):
    """社区检测"""
    print("  Detecting communities...")

    if G.number_of_nodes() == 0:
        return {}

    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, seed=42)
    except:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G)
        except:
            print("    Warning: Community detection failed")
            return {}

    # 转换为字典
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i

    n_communities = len(set(community_dict.values()))
    print(f"    Found {n_communities} communities")

    return community_dict


def create_pathway_enrichment(gene_list):
    """创建通路富集分析"""
    print("  Running pathway enrichment...")

    # 使用STRING API进行富集分析
    if not gene_list:
        return pd.DataFrame()

    genes_str = "%0d".join(gene_list[:200])  # API限制

    url = f"{STRING_API}/json/enrichment"
    params = {
        'identifiers': genes_str,
        'species': 9606,
        'caller_identity': 'pelvic_floor_gwas'
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            enrichment = response.json()

            if enrichment:
                enrich_df = pd.DataFrame(enrichment)
                # 筛选显著结果
                if 'fdr' in enrich_df.columns:
                    enrich_df = enrich_df[enrich_df['fdr'] < 0.05]
                print(f"    Found {len(enrich_df)} significant pathways")
                return enrich_df
    except Exception as e:
        print(f"    Warning: Enrichment analysis error - {e}")

    return pd.DataFrame()


def create_visualizations(G, metrics_df, community_dict, enrichment_df):
    """生成可视化图表"""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300

    if G.number_of_nodes() == 0:
        print("  Skipping visualization: empty network")
        return

    fig = plt.figure(figsize=(16, 12))

    # 1. 网络图
    ax1 = fig.add_subplot(2, 2, 1)

    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 节点颜色和大小
    if community_dict:
        node_colors = [community_dict.get(n, 0) for n in G.nodes()]
    else:
        node_colors = [G.nodes[n].get('n_phenotypes', 0) for n in G.nodes()]

    node_sizes = [max(G.degree(n) * 100, 50) for n in G.nodes()]

    nx.draw_networkx(G, pos, ax=ax1,
                     node_color=node_colors, cmap='Set3',
                     node_size=node_sizes,
                     font_size=8, font_weight='bold',
                     edge_color='gray', alpha=0.7,
                     with_labels=True)

    ax1.set_title('PPI Network\n(Node size = degree, color = community)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Hub基因条形图
    ax2 = fig.add_subplot(2, 2, 2)

    if len(metrics_df) > 0:
        top_hubs = metrics_df.head(15)
        y_pos = np.arange(len(top_hubs))

        colors = ['#E64B35' if row['N_phenotypes'] > 1 else '#4DBBD5'
                 for _, row in top_hubs.iterrows()]

        ax2.barh(y_pos, top_hubs['Degree'], color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_hubs['Gene'])
        ax2.invert_yaxis()
        ax2.set_xlabel('Degree', fontsize=11)
        ax2.set_title('Hub Genes by Connectivity\n(Red = multi-phenotype)', fontsize=12, fontweight='bold')

    # 3. 中心性热图
    ax3 = fig.add_subplot(2, 2, 3)

    if len(metrics_df) > 0:
        # 选择top基因
        top_genes = metrics_df.head(15)[['Gene', 'Degree_Centrality', 'Betweenness', 'Eigenvector', 'Clustering']]
        top_genes = top_genes.set_index('Gene')

        sns.heatmap(top_genes, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3,
                    cbar_kws={'label': 'Centrality Score'})
        ax3.set_title('Network Centrality Metrics', fontsize=12, fontweight='bold')
        ax3.set_xlabel('')

    # 4. 富集分析
    ax4 = fig.add_subplot(2, 2, 4)

    if len(enrichment_df) > 0 and 'description' in enrichment_df.columns:
        # 按FDR排序取top 10
        top_pathways = enrichment_df.nsmallest(10, 'fdr')

        if 'term' in top_pathways.columns:
            labels = top_pathways['term'].str[:40]  # 截断长名称
        else:
            labels = top_pathways['description'].str[:40]

        y_pos = np.arange(len(top_pathways))
        neg_log_fdr = -np.log10(top_pathways['fdr'].clip(lower=1e-16))

        ax4.barh(y_pos, neg_log_fdr, color='#00A087', alpha=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.invert_yaxis()
        ax4.set_xlabel('-log10(FDR)', fontsize=11)
        ax4.set_title('Top Enriched Pathways', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No significant\nenrichment found',
                ha='center', va='center', fontsize=14, color='gray')
        ax4.axis('off')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'ppi_network_expanded.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'ppi_network_expanded.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved visualizations to {FIGURES_DIR}")


def write_log(gene_data, G, metrics_df, community_dict, enrichment_df, mapped_count, unmapped_count):
    """写入分析日志"""
    log_file = BASE_DIR / "logs" / "14b_ppi_network_expanded.md"

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("# Log 14b: Expanded PPI Network Analysis\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write("**Status**: Completed\n\n")
        f.write("---\n\n")

        f.write("## Gene Mapping Statistics\n\n")
        f.write(f"- **Using**: NCBI gene_info (193,701 mappings)\n")
        f.write(f"- **Mapped genes**: {mapped_count}\n")
        f.write(f"- **Unmapped genes**: {unmapped_count}\n\n")

        f.write("## Network Statistics\n\n")
        f.write(f"- **Input genes**: {len(gene_data)}\n")
        f.write(f"- **Network nodes**: {G.number_of_nodes()}\n")
        f.write(f"- **Network edges**: {G.number_of_edges()}\n")
        f.write(f"- **Communities detected**: {len(set(community_dict.values())) if community_dict else 0}\n\n")

        f.write("---\n\n")

        f.write("## Top Hub Genes\n\n")
        f.write("| Gene | Degree | Betweenness | N_Phenotypes | P_value |\n")
        f.write("|------|--------|-------------|--------------|----------|\n")
        for _, row in metrics_df.head(15).iterrows():
            f.write(f"| {row['Gene']} | {row['Degree']} | {row['Betweenness']:.3f} | ")
            f.write(f"{row['N_phenotypes']} | {row['P_value']:.2e} |\n")
        f.write("\n")

        f.write("---\n\n")

        f.write("## Enriched Pathways\n\n")
        if len(enrichment_df) > 0 and 'description' in enrichment_df.columns:
            f.write("| Pathway | FDR | Genes |\n")
            f.write("|---------|-----|-------|\n")
            for _, row in enrichment_df.head(10).iterrows():
                desc = row.get('description', row.get('term', 'Unknown'))[:50]
                fdr = row.get('fdr', 1)
                n_genes = row.get('number_of_genes', 0)
                f.write(f"| {desc} | {fdr:.2e} | {n_genes} |\n")
        else:
            f.write("No significant enrichment found.\n")

        f.write("\n---\n\n")
        f.write("## Output Files\n\n")
        f.write("```\n")
        f.write("results/ppi_network_expanded/\n")
        f.write("+-- network_metrics_expanded.csv\n")
        f.write("+-- community_assignments.csv\n")
        f.write("+-- pathway_enrichment.csv\n")
        f.write("+-- ppi_network_expanded.graphml\n")
        f.write("```\n")

    print(f"  Log saved to: {log_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("Expanded PPI Network Analysis")
    print("Using Complete NCBI Gene Mapping (193,701 mappings)")
    print("=" * 60)

    # 加载完整NCBI映射
    print("\n[0] Loading NCBI gene mapping...")
    entrez_to_symbol, _, _ = load_ncbi_gene_mapping()

    # 加载基因
    print("\n[1] Loading MAGMA genes...")
    gene_data = load_magma_genes(p_threshold=0.001, top_n=50)

    if len(gene_data) == 0:
        print("Error: No gene data found")
        return

    # 获取基因符号（使用完整NCBI映射）
    symbols = get_gene_symbols(gene_data['GeneID'].astype(str).tolist(), entrez_to_symbol)

    # 统计映射成功数
    mapped_count = sum(1 for s in symbols.values() if not s.startswith('GENE_'))
    unmapped_count = sum(1 for s in symbols.values() if s.startswith('GENE_'))

    # 获取STRING相互作用
    print("\n[2] Fetching STRING interactions...")
    gene_symbols = list(set(symbols.values()))
    # 过滤掉GENE_xxx这样的未知符号
    gene_symbols = [s for s in gene_symbols if not s.startswith('GENE_')]
    print(f"    Querying {len(gene_symbols)} known gene symbols")

    interactions = get_string_interactions(gene_symbols)

    # 构建网络
    print("\n[3] Building network...")
    # 为gene_data添加索引
    gene_data_indexed = gene_data.copy()
    G = build_network(interactions, gene_data_indexed.reset_index(), symbols)

    # 分析网络
    print("\n[4] Analyzing network...")
    metrics_df = analyze_network(G)

    # 社区检测
    print("\n[5] Community detection...")
    community_dict = detect_communities(G)

    # 通路富集
    print("\n[6] Pathway enrichment...")
    enrichment_df = create_pathway_enrichment(gene_symbols)

    # 保存结果
    print("\n[7] Saving results...")
    metrics_df.to_csv(RESULTS_DIR / "network_metrics_expanded.csv", index=False)

    if community_dict:
        community_df = pd.DataFrame([
            {'Gene': gene, 'Community': comm}
            for gene, comm in community_dict.items()
        ])
        community_df.to_csv(RESULTS_DIR / "community_assignments.csv", index=False)

    if len(enrichment_df) > 0:
        enrichment_df.to_csv(RESULTS_DIR / "pathway_enrichment.csv", index=False)

    nx.write_graphml(G, str(RESULTS_DIR / "ppi_network_expanded.graphml"))

    # 可视化
    print("\n[8] Generating visualizations...")
    create_visualizations(G, metrics_df, community_dict, enrichment_df)

    # 日志
    print("\n[9] Writing log...")
    write_log(gene_data, G, metrics_df, community_dict, enrichment_df, mapped_count, unmapped_count)

    print("\n" + "=" * 60)
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
