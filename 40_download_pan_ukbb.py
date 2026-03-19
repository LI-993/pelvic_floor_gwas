#!/usr/bin/env python3
"""
下载Pan-UKBB多人群GWAS数据

Pan-UKBB提供6个人群的GWAS结果:
- EUR: European
- AFR: African
- EAS: East Asian
- CSA: Central/South Asian
- MID: Middle Eastern
- AMR: Admixed American

数据源: https://pan.ukbb.broadinstitute.org/
"""

import os
import subprocess
from pathlib import Path
import gzip
import pandas as pd

# 路径设置
BASE_DIR = Path("d:/Nproject/gwas/pelvic_floor_gwas")
DATA_DIR = BASE_DIR / "data" / "pan_ukbb"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pan-UKBB AWS S3基础URL
S3_BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com"
MANIFEST_URL = f"{S3_BASE}/sumstats_release/phenotype_manifest.tsv.bgz"

# 目标表型关键词 (用于在manifest中搜索)
TARGET_KEYWORDS = [
    'prolapse',
    'incontinence',
    'prostat',  # 匹配prostate, prostatic等
    'bladder',
    'constipation',
    'urinary',
    'pelvic',
]

# ICD-10 编码
TARGET_ICD10 = ['N81', 'N39', 'N40', 'N32', 'K59', 'R32', 'N31']

# 目标人群
POPULATIONS = ['EUR', 'AFR', 'EAS', 'CSA', 'MID', 'AMR', 'meta']


def download_file(url, output_path):
    """下载文件"""
    cmd = f'curl -L -o "{output_path}" "{url}"'
    print(f"  Downloading: {url.split('/')[-1]}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 100:
            print(f"    Success: {output_path.stat().st_size / 1e6:.2f} MB")
            return True
        else:
            print(f"    Failed")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def download_manifest():
    """下载phenotype manifest"""
    manifest_path = DATA_DIR / "phenotype_manifest.tsv.bgz"

    if manifest_path.exists():
        print(f"  Manifest already exists: {manifest_path}")
    else:
        success = download_file(MANIFEST_URL, manifest_path)
        if not success:
            print("  Failed to download manifest!")
            return None

    # 读取manifest
    print("  Reading manifest...")
    df = pd.read_csv(manifest_path, sep='\t', compression='gzip')
    print(f"    Total phenotypes in manifest: {len(df)}")

    return df


def find_relevant_phenotypes(manifest_df):
    """在manifest中找到相关表型"""
    relevant = []

    for _, row in manifest_df.iterrows():
        phenocode = str(row.get('phenocode', '')).upper()
        description = str(row.get('description', '')).lower()
        trait_type = str(row.get('trait_type', ''))

        # 检查ICD-10编码
        is_target_icd = any(icd in phenocode for icd in TARGET_ICD10)

        # 检查关键词
        has_keyword = any(kw in description for kw in TARGET_KEYWORDS)

        if is_target_icd or has_keyword:
            relevant.append(row)

    df = pd.DataFrame(relevant)
    print(f"\n  Found {len(df)} relevant phenotypes")

    return df


def get_download_urls(relevant_df):
    """从manifest获取下载URL"""
    downloads = []

    # manifest中应该有指向各人群文件的路径
    # 需要检查实际的列名
    print(f"\n  Manifest columns: {list(relevant_df.columns)[:10]}...")

    for _, row in relevant_df.iterrows():
        phenocode = row.get('phenocode', 'unknown')
        description = row.get('description', '')

        # 检查是否有各人群的文件路径列
        for pop in POPULATIONS:
            # 可能的列名格式
            possible_cols = [
                f'{pop}_filename',
                f'filename_{pop}',
                f'{pop.lower()}_path',
                f'aws_path_{pop}',
            ]

            for col in possible_cols:
                if col in row.index and pd.notna(row[col]):
                    downloads.append({
                        'phenocode': phenocode,
                        'description': description,
                        'population': pop,
                        'filename': row[col],
                    })
                    break

        # 如果有通用filename列
        if 'aws_path' in row.index and pd.notna(row['aws_path']):
            downloads.append({
                'phenocode': phenocode,
                'description': description,
                'population': 'all',
                'filename': row['aws_path'],
            })

    return pd.DataFrame(downloads)


def main():
    print("=" * 60)
    print("Pan-UKBB Multi-Ancestry GWAS Data Download")
    print("=" * 60)

    # Step 1: 下载manifest
    print("\n[1] Downloading phenotype manifest...")
    manifest = download_manifest()

    if manifest is None:
        print("Failed to get manifest. Exiting.")
        return

    # 显示manifest结构
    print(f"\n  Manifest columns:")
    for col in manifest.columns:
        print(f"    - {col}")

    # Step 2: 找到相关表型
    print("\n[2] Finding relevant phenotypes...")
    relevant = find_relevant_phenotypes(manifest)

    if len(relevant) == 0:
        print("No relevant phenotypes found!")
        return

    # 保存相关表型信息
    relevant.to_csv(DATA_DIR / "relevant_phenotypes.csv", index=False)
    print(f"\n  Saved relevant phenotypes to: {DATA_DIR / 'relevant_phenotypes.csv'}")

    # 显示找到的表型
    print("\n  Relevant phenotypes found:")
    cols_to_show = ['phenocode', 'description', 'trait_type']
    cols_to_show = [c for c in cols_to_show if c in relevant.columns]
    if cols_to_show:
        print(relevant[cols_to_show].head(20).to_string())

    # Step 3: 检查数据可用性
    print("\n[3] Checking data availability by population...")

    # 检查哪些人群有数据
    pop_cols = [col for col in relevant.columns if any(p.lower() in col.lower() for p in POPULATIONS)]
    print(f"\n  Population-related columns: {pop_cols}")

    # 如果有n_cases列等，按人群统计
    for pop in ['EUR', 'AFR', 'EAS', 'CSA', 'AMR', 'MID']:
        n_col = f'n_cases_{pop}'
        if n_col in relevant.columns:
            has_data = relevant[n_col].notna() & (relevant[n_col] > 0)
            print(f"    {pop}: {has_data.sum()} phenotypes with data")

    # Step 4: 生成下载指南
    print("\n[4] Generating download guide...")

    # Pan-UKBB的实际下载需要通过manifest中的路径
    guide_path = DATA_DIR / "download_guide.txt"
    with open(guide_path, 'w') as f:
        f.write("Pan-UKBB Download Guide\n")
        f.write("=" * 50 + "\n\n")
        f.write("Base URL: https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/\n\n")
        f.write("Relevant phenotypes:\n")
        for _, row in relevant.iterrows():
            f.write(f"\n  {row.get('phenocode', 'NA')}: {row.get('description', 'NA')}\n")
            if 'aws_path' in row.index:
                f.write(f"    Path: {row.get('aws_path', 'NA')}\n")

    print(f"  Download guide saved to: {guide_path}")

    # Step 5: 尝试下载一个文件作为测试
    print("\n[5] Testing download...")

    if 'aws_path' in relevant.columns:
        test_row = relevant.iloc[0]
        test_path = test_row['aws_path']
        if pd.notna(test_path):
            test_url = f"{S3_BASE}/sumstats_release/{test_path}"
            test_output = DATA_DIR / Path(test_path).name
            print(f"  Test URL: {test_url}")
            download_file(test_url, test_output)


if __name__ == "__main__":
    main()
