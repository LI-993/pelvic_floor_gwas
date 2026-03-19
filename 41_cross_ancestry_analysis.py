#!/usr/bin/env python3
"""
跨人群验证分析

下载Pan-UKBB多人群数据并分析：
1. 下载关键表型的summary statistics
2. 比较不同人群的遗传信号
3. 检验EUR发现在其他人群的复制率
"""

import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# 路径设置
BASE_DIR = Path("d:/Nproject/gwas/pelvic_floor_gwas")
DATA_DIR = BASE_DIR / "data" / "pan_ukbb"
RESULTS_DIR = BASE_DIR / "results" / "cross_ancestry"
FIGURES_DIR = BASE_DIR / "figures" / "cross_ancestry"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Pan-UKBB下载基础URL
S3_BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files"

# 关键表型
KEY_PHENOTYPES = {
    'N81': {
        'name': 'Female_genital_prolapse',
        'file': 'icd10-N81-both_sexes.tsv.bgz',
        'populations': ['EUR', 'AFR', 'CSA'],
    },
    'N39': {
        'name': 'Urinary_disorders',
        'file': 'icd10-N39-both_sexes.tsv.bgz',
        'populations': ['EUR', 'AFR', 'CSA', 'EAS', 'MID'],
    },
    'N32': {
        'name': 'Bladder_disorders',
        'file': 'icd10-N32-both_sexes.tsv.bgz',
        'populations': ['EUR', 'AFR', 'CSA', 'MID'],
    },
    'K59': {
        'name': 'Constipation',
        'file': 'icd10-K59-both_sexes.tsv.bgz',
        'populations': ['EUR', 'AFR', 'CSA', 'EAS', 'MID'],
    },
    '599.4': {
        'name': 'Urinary_incontinence',
        'file': 'phecode-599.4-both_sexes.tsv.bgz',
        'populations': ['EUR', 'AFR', 'CSA'],
    },
}

# 我们EUR发现的top位点 (从现有分析)
EUR_TOP_LOCI = [
    # chr, pos (GRCh38), rsid, gene
    ('1', 22138563, 'rs7527902', 'WNT4'),
    ('11', 32413566, 'rs2276849', 'WT1'),
    ('15', 74217882, 'rs12324955', 'LOXL1'),
    ('22', 38477419, 'rs738722', 'PLA2G6'),
]


def download_file(url, output_path):
    """下载文件"""
    if output_path.exists():
        print(f"    Already exists: {output_path.name}")
        return True

    cmd = f'curl -L -o "{output_path}" "{url}"'
    print(f"    Downloading: {output_path.name}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            size_mb = output_path.stat().st_size / 1e6
            print(f"      Success: {size_mb:.1f} MB")
            return True
        else:
            print(f"      Failed")
            return False
    except Exception as e:
        print(f"      Error: {e}")
        return False


def download_pan_ukbb_data():
    """下载Pan-UKBB数据"""
    print("\n" + "=" * 60)
    print("Downloading Pan-UKBB Multi-Ancestry Data")
    print("=" * 60)

    downloaded = []
    for phenocode, info in KEY_PHENOTYPES.items():
        url = f"{S3_BASE}/{info['file']}"
        output_path = DATA_DIR / info['file']

        print(f"\n  {phenocode}: {info['name']}")
        success = download_file(url, output_path)

        if success:
            downloaded.append({
                'phenocode': phenocode,
                'name': info['name'],
                'file': str(output_path),
                'populations': info['populations'],
            })

    # 保存下载清单
    if downloaded:
        pd.DataFrame(downloaded).to_csv(DATA_DIR / "downloaded_files.csv", index=False)

    return downloaded


def load_pan_ukbb_sumstats(filepath, population='EUR'):
    """
    加载Pan-UKBB summary statistics

    Pan-UKBB文件包含所有人群的结果在同一文件中
    列名格式: beta_{pop}, se_{pop}, pval_{pop}, af_{pop}
    """
    print(f"    Loading {filepath.name} for {population}...")

    # Pan-UKBB使用bgzip压缩的tsv
    df = pd.read_csv(filepath, sep='\t', compression='gzip', nrows=100000)  # 先读一部分看结构

    print(f"      Columns: {list(df.columns)[:15]}...")
    print(f"      Shape: {df.shape}")

    # 检查人群特异性列
    pop_cols = [col for col in df.columns if population.lower() in col.lower()]
    print(f"      {population} columns: {pop_cols}")

    return df


def analyze_cross_ancestry_replication(downloaded):
    """分析跨人群复制"""
    print("\n" + "=" * 60)
    print("Cross-Ancestry Replication Analysis")
    print("=" * 60)

    results = []

    for item in downloaded:
        filepath = Path(item['file'])
        if not filepath.exists():
            continue

        print(f"\n  Analyzing {item['name']}...")

        # 加载数据
        df = load_pan_ukbb_sumstats(filepath, 'EUR')

        if df is None or len(df) == 0:
            continue

        # 分析各人群
        for pop in item['populations']:
            # 检查该人群的数据列
            beta_col = f'beta_{pop}'
            pval_col = f'pval_{pop}'

            if beta_col in df.columns and pval_col in df.columns:
                # 计算基本统计
                valid = df[pval_col].notna()
                n_valid = valid.sum()
                n_sig = (df[pval_col] < 5e-8).sum()

                results.append({
                    'phenotype': item['name'],
                    'population': pop,
                    'n_variants': n_valid,
                    'n_gwas_sig': n_sig,
                })

                print(f"      {pop}: {n_valid:,} variants, {n_sig} GWAS-significant")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / "cross_ancestry_summary.csv", index=False)
        return results_df

    return None


def compare_effect_sizes(downloaded):
    """比较不同人群的效应量"""
    print("\n" + "=" * 60)
    print("Effect Size Comparison Across Ancestries")
    print("=" * 60)

    comparisons = []

    for item in downloaded:
        filepath = Path(item['file'])
        if not filepath.exists():
            continue

        print(f"\n  {item['name']}...")

        # 只读取需要的列，使用chunksize提高效率
        # 先获取列名
        with gzip.open(filepath, 'rt') as f:
            header = f.readline().strip().split('\t')

        # 找出需要的列
        needed_cols = ['chr', 'pos', 'ref', 'alt']
        for pop in item['populations']:
            needed_cols.extend([f'beta_{pop}', f'pval_{pop}', f'af_{pop}'])

        usecols = [c for c in needed_cols if c in header]
        print(f"    Loading columns: {usecols[:6]}...")

        # 分块读取并采样
        chunks = pd.read_csv(filepath, sep='\t', compression='gzip',
                            usecols=usecols, chunksize=500000)
        df = pd.concat([chunk for chunk in chunks], ignore_index=True)
        print(f"    Loaded {len(df):,} variants")

        # EUR vs 其他人群的效应量相关性
        for pop in item['populations']:
            if pop == 'EUR':
                continue

            beta_eur = f'beta_EUR'
            beta_pop = f'beta_{pop}'

            if beta_eur in df.columns and beta_pop in df.columns:
                # 选择两个人群都有效应量的变异
                valid = df[beta_eur].notna() & df[beta_pop].notna()
                valid_df = df[valid]

                if len(valid_df) > 100:
                    # 计算相关性
                    corr = valid_df[beta_eur].corr(valid_df[beta_pop])

                    comparisons.append({
                        'phenotype': item['name'],
                        'comparison': f'EUR_vs_{pop}',
                        'n_variants': len(valid_df),
                        'effect_correlation': corr,
                    })

                    print(f"      EUR vs {pop}: r = {corr:.3f} ({len(valid_df):,} variants)")

    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        comp_df.to_csv(RESULTS_DIR / "effect_size_correlations.csv", index=False)
        return comp_df

    return None


def check_top_loci_replication(downloaded):
    """检查EUR top位点在其他人群的复制"""
    print("\n" + "=" * 60)
    print("Top Loci Replication Check")
    print("=" * 60)

    # 这需要先知道具体的位点位置
    # 暂时跳过，因为需要处理染色体位置匹配

    print("  [Skipped - requires variant position matching]")
    return None


def create_visualizations(results_df, comp_df):
    """创建可视化"""
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    if results_df is not None and len(results_df) > 0:
        # 1. 各人群GWAS显著变异数量
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = results_df.pivot(index='phenotype', columns='population', values='n_gwas_sig')
        pivot.plot(kind='bar', ax=ax)
        ax.set_ylabel('Number of GWAS-significant variants (P < 5e-8)')
        ax.set_title('GWAS Signals Across Ancestries')
        ax.legend(title='Population')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "gwas_signals_by_ancestry.png", dpi=300)
        plt.close()
        print("  Saved: gwas_signals_by_ancestry.png")

    if comp_df is not None and len(comp_df) > 0:
        # 2. 效应量相关性热图
        fig, ax = plt.subplots(figsize=(8, 6))

        # 创建热图数据
        pivot = comp_df.pivot(index='phenotype', columns='comparison', values='effect_correlation')

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Effect Size Correlation: EUR vs Other Ancestries')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "effect_correlation_heatmap.png", dpi=300)
        plt.close()
        print("  Saved: effect_correlation_heatmap.png")


def main():
    print("=" * 60)
    print("Pan-UKBB Cross-Ancestry Validation Analysis")
    print("=" * 60)

    # Step 1: 下载数据
    print("\n[1] Downloading Pan-UKBB data...")
    downloaded = download_pan_ukbb_data()

    if not downloaded:
        print("No data downloaded. Exiting.")
        return

    # Step 2: 跨人群复制分析
    print("\n[2] Analyzing cross-ancestry replication...")
    results_df = analyze_cross_ancestry_replication(downloaded)

    # Step 3: 效应量比较
    print("\n[3] Comparing effect sizes...")
    comp_df = compare_effect_sizes(downloaded)

    # Step 4: Top位点复制检查
    print("\n[4] Checking top loci replication...")
    check_top_loci_replication(downloaded)

    # Step 5: 可视化
    print("\n[5] Creating visualizations...")
    create_visualizations(results_df, comp_df)

    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Downloaded files: {len(downloaded)}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
