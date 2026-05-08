#!/usr/bin/env python3
"""
25_visualize_mr.py - 孟德尔随机化结果可视化

生成图表:
1. 因果关系网络图（有向图）
2. MR效应森林图
3. 方法比较图
4. 因果效应热图

Author: Claude
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from matplotlib.patches import FancyArrowPatch, ArrowStyle
import matplotlib.patches as mpatches
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
RESULTS_DIR = BASE_DIR / "results" / "mr"
FIGURES_DIR = BASE_DIR / "figures" / "mr"
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

# 表型简称
PHENOTYPE_SHORT = {
    'POP': 'POP',
    'BPH': 'BPH',
    'Bladder': 'Bladder',
    'Constipation': 'Constip.',
    'FemaleProlapse': 'F.Prolapse',
    'Incontinence': 'Incontin.'
}


def load_data():
    """加载MR结果数据"""
    mr_file = RESULTS_DIR / "mr_bidirectional_results.csv"
    mr_df = pd.read_csv(mr_file)
    return mr_df


def plot_causal_network(mr_df):
    """绘制因果关系网络图"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点
    phenotypes = list(set(mr_df['exposure'].tolist() + mr_df['outcome'].tolist()))
    for p in phenotypes:
        G.add_node(p)

    # 添加边（只显示显著的因果关系）
    sig_threshold = 0.05
    for _, row in mr_df.iterrows():
        if row['ivw_p'] < sig_threshold:
            G.add_edge(row['exposure'], row['outcome'],
                      beta=row['ivw_beta'],
                      p=row['ivw_p'],
                      n_snps=row['n_snps'])

    # 使用circular布局
    pos = nx.circular_layout(G)

    # 绘制节点
    node_sizes = 4000
    node_colors = [PHENOTYPE_COLORS.get(n, '#888888') for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=node_sizes, alpha=0.9, ax=ax)

    # 绘制节点标签
    labels = {p: PHENOTYPE_SHORT.get(p, p) for p in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold', ax=ax)

    # 绘制边（带箭头）
    for edge in G.edges(data=True):
        u, v, d = edge
        beta = d['beta']
        p_val = d['p']

        # 颜色：正效应红色，负效应蓝色
        color = '#E64B35' if beta > 0 else '#4DBBD5'

        # 线宽根据效应大小
        width = 1 + abs(beta) * 5

        # 透明度根据显著性
        alpha = min(1.0, 0.4 - np.log10(p_val) * 0.1)

        # 绘制边
        start = pos[u]
        end = pos[v]

        # 计算箭头位置（缩短以避免覆盖节点）
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = np.sqrt(dx**2 + dy**2)

        # 缩短箭头
        shrink = 0.15
        start_adj = (start[0] + dx * shrink, start[1] + dy * shrink)
        end_adj = (end[0] - dx * shrink, end[1] - dy * shrink)

        arrow = FancyArrowPatch(start_adj, end_adj,
                               connectionstyle="arc3,rad=0.1",
                               arrowstyle=ArrowStyle('->', head_length=10, head_width=6),
                               color=color, alpha=alpha, linewidth=width,
                               mutation_scale=15)
        ax.add_patch(arrow)

        # 在边上添加beta值
        mid_x = (start[0] + end[0]) / 2 + 0.05
        mid_y = (start[1] + end[1]) / 2 + 0.05
        ax.text(mid_x, mid_y, f'β={beta:.2f}', fontsize=8,
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 图例
    legend_elements = [
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color='#E64B35', label='Positive causal effect'),
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color='#4DBBD5', label='Negative causal effect'),
    ]
    ax.legend(handles=[
        plt.Line2D([0], [0], color='#E64B35', linewidth=3, label='Positive effect (β>0)'),
        plt.Line2D([0], [0], color='#4DBBD5', linewidth=3, label='Negative effect (β<0)'),
    ], loc='upper left', fontsize=10)

    ax.set_title('Mendelian Randomization Causal Network\n(Significant at p<0.05, IVW method)',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    # 设置范围
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'mr_causal_network.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'mr_causal_network.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: mr_causal_network.png/pdf")


def plot_forest_plot(mr_df):
    """绘制MR效应森林图"""
    # 筛选显著结果
    sig_df = mr_df[mr_df['ivw_p'] < 0.05].copy()

    if len(sig_df) == 0:
        print("  Warning: No significant MR results for forest plot")
        return

    # 创建标签
    sig_df['label'] = sig_df['exposure'] + ' → ' + sig_df['outcome']

    # 按效应大小排序
    sig_df = sig_df.sort_values('ivw_beta', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sig_df) * 0.5)))

    y_pos = np.arange(len(sig_df))

    # 计算95% CI
    sig_df['ci_lower'] = sig_df['ivw_beta'] - 1.96 * sig_df['ivw_se']
    sig_df['ci_upper'] = sig_df['ivw_beta'] + 1.96 * sig_df['ivw_se']

    # 颜色
    colors = ['#E64B35' if b > 0 else '#4DBBD5' for b in sig_df['ivw_beta']]

    # 绘制点和误差线
    ax.errorbar(sig_df['ivw_beta'], y_pos,
               xerr=[sig_df['ivw_beta'] - sig_df['ci_lower'],
                     sig_df['ci_upper'] - sig_df['ivw_beta']],
               fmt='o', markersize=8, capsize=4, capthick=1.5,
               color='black', ecolor='gray', elinewidth=1.5)

    # 绘制彩色点
    for i, (beta, color) in enumerate(zip(sig_df['ivw_beta'], colors)):
        ax.scatter(beta, i, c=color, s=100, zorder=5, edgecolors='white', linewidths=1)

    # 垂直参考线
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # 设置标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_df['label'])

    ax.set_xlabel('Causal Effect (β) with 95% CI', fontsize=12)
    ax.set_title('Mendelian Randomization Forest Plot\n(IVW Method, p<0.05)',
                fontsize=14, fontweight='bold')

    # 在右侧添加P值和SNP数
    for i, row in enumerate(sig_df.itertuples()):
        ax.text(ax.get_xlim()[1] * 1.02, i,
               f'p={row.ivw_p:.2e} (n={row.n_snps})',
               va='center', fontsize=9)

    # 调整x轴范围
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 1.3)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'mr_forest_plot.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'mr_forest_plot.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: mr_forest_plot.png/pdf")


def plot_method_comparison(mr_df):
    """绘制不同MR方法的比较图"""
    # 选择几个关键的因果对
    key_pairs = [
        ('BPH', 'Incontinence'),
        ('Incontinence', 'BPH'),
        ('POP', 'Incontinence'),
        ('POP', 'FemaleProlapse'),
        ('FemaleProlapse', 'POP'),
        ('FemaleProlapse', 'Incontinence')
    ]

    # 筛选数据
    plot_data = []
    for exp, out in key_pairs:
        row = mr_df[(mr_df['exposure'] == exp) & (mr_df['outcome'] == out)]
        if len(row) > 0:
            row = row.iloc[0]
            plot_data.append({
                'Pair': f'{exp} → {out}',
                'IVW': row['ivw_beta'],
                'IVW_SE': row['ivw_se'],
                'WM': row['wm_beta'],
                'WM_SE': row['wm_se'],
                'Egger': row['egger_beta'],
                'Egger_SE': row['egger_se']
            })

    if not plot_data:
        print("  Warning: No data for method comparison")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(plot_df))
    width = 0.25

    # 绘制三种方法的条形图
    bars1 = ax.bar(x - width, plot_df['IVW'], width, yerr=plot_df['IVW_SE'],
                   label='IVW', color='#E64B35', alpha=0.8, capsize=3)
    bars2 = ax.bar(x, plot_df['WM'], width, yerr=plot_df['WM_SE'],
                   label='Weighted Median', color='#4DBBD5', alpha=0.8, capsize=3)
    bars3 = ax.bar(x + width, plot_df['Egger'], width, yerr=plot_df['Egger_SE'],
                   label='MR-Egger', color='#00A087', alpha=0.8, capsize=3)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Causal Relationship', fontsize=12)
    ax.set_ylabel('Causal Effect (β)', fontsize=12)
    ax.set_title('Comparison of MR Methods for Key Causal Pairs',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Pair'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'mr_method_comparison.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'mr_method_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: mr_method_comparison.png/pdf")


def plot_causal_heatmap(mr_df):
    """绘制因果效应热图"""
    # 获取所有表型
    phenotypes = list(set(mr_df['exposure'].tolist() + mr_df['outcome'].tolist()))

    # 创建效应矩阵
    beta_matrix = pd.DataFrame(np.nan, index=phenotypes, columns=phenotypes)
    p_matrix = pd.DataFrame(1.0, index=phenotypes, columns=phenotypes)

    for _, row in mr_df.iterrows():
        beta_matrix.loc[row['exposure'], row['outcome']] = row['ivw_beta']
        p_matrix.loc[row['exposure'], row['outcome']] = row['ivw_p']

    # 对角线设为0
    for p in phenotypes:
        beta_matrix.loc[p, p] = 0

    # 创建注释矩阵（显著性星号）
    annot_matrix = beta_matrix.copy().astype(str)
    for i in phenotypes:
        for j in phenotypes:
            beta = beta_matrix.loc[i, j]
            p_val = p_matrix.loc[i, j]
            if pd.isna(beta):
                annot_matrix.loc[i, j] = ''
            elif i == j:
                annot_matrix.loc[i, j] = '-'
            else:
                if p_val < 0.001:
                    annot_matrix.loc[i, j] = f'{beta:.2f}***'
                elif p_val < 0.01:
                    annot_matrix.loc[i, j] = f'{beta:.2f}**'
                elif p_val < 0.05:
                    annot_matrix.loc[i, j] = f'{beta:.2f}*'
                else:
                    annot_matrix.loc[i, j] = f'{beta:.2f}'

    fig, ax = plt.subplots(figsize=(10, 8))

    # 短标签
    labels = [PHENOTYPE_SHORT.get(p, p) for p in phenotypes]

    # 绘制热图
    mask = beta_matrix.isna()
    sns.heatmap(beta_matrix,
                mask=mask,
                annot=annot_matrix,
                fmt='',
                cmap='RdBu_r',
                center=0,
                linewidths=0.5,
                cbar_kws={'label': 'Causal Effect (β)'},
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)

    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Exposure', fontsize=12)
    ax.set_title('Bidirectional Mendelian Randomization Causal Effects\n(*** p<0.001, ** p<0.01, * p<0.05)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'mr_causal_heatmap.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'mr_causal_heatmap.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: mr_causal_heatmap.png/pdf")


def plot_pleiotropy_check(mr_df):
    """绘制多效性检验图（MR-Egger intercept）"""
    # 筛选有显著IVW结果的
    sig_df = mr_df[mr_df['ivw_p'] < 0.1].copy()

    if len(sig_df) == 0:
        print("  Warning: No data for pleiotropy check")
        return

    sig_df['label'] = sig_df['exposure'] + ' → ' + sig_df['outcome']

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(sig_df))

    # 绘制intercept
    colors = ['#E64B35' if p < 0.05 else '#00A087'
              for p in sig_df['egger_intercept_p']]

    bars = ax.barh(y_pos, sig_df['egger_intercept'],
                   color=colors, alpha=0.8, edgecolor='white')

    # 垂直参考线
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_df['label'])
    ax.set_xlabel('MR-Egger Intercept', fontsize=12)
    ax.set_title('Horizontal Pleiotropy Check (MR-Egger Intercept)\nRed = significant pleiotropy (p<0.05)',
                fontsize=14, fontweight='bold')

    # 在右侧添加P值
    for i, row in enumerate(sig_df.itertuples()):
        ax.text(ax.get_xlim()[1] * 0.95, i,
               f'p={row.egger_intercept_p:.3f}',
               va='center', ha='right', fontsize=9)

    plt.tight_layout()

    # 保存
    fig.savefig(FIGURES_DIR / 'mr_pleiotropy_check.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'mr_pleiotropy_check.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved: mr_pleiotropy_check.png/pdf")


def main():
    """主函数"""
    print("=" * 60)
    print("Mendelian Randomization Visualization")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    mr_df = load_data()
    print(f"  Loaded {len(mr_df)} MR tests")

    # 统计
    n_sig = (mr_df['ivw_p'] < 0.05).sum()
    print(f"  Significant causal relationships (p<0.05): {n_sig}")

    # 生成可视化
    print("\n[2] Generating visualizations...")

    print("\n  2.1 Causal network...")
    plot_causal_network(mr_df)

    print("\n  2.2 Forest plot...")
    plot_forest_plot(mr_df)

    print("\n  2.3 Method comparison...")
    plot_method_comparison(mr_df)

    print("\n  2.4 Causal heatmap...")
    plot_causal_heatmap(mr_df)

    print("\n  2.5 Pleiotropy check...")
    plot_pleiotropy_check(mr_df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
