#!/usr/bin/env python3
"""
基因ID映射工具 - 使用NCBI gene_info完整映射

解决之前Entrez ID无法映射到Symbol的问题
"""

import gzip
from pathlib import Path
import pandas as pd

# 文件路径
GENE_INFO_FILE = Path("d:/Nproject/gwas/pelvic_floor_gwas/data/reference/Homo_sapiens.gene_info.gz")


def load_ncbi_gene_mapping():
    """
    加载NCBI gene_info文件，创建完整的Entrez ID到Symbol映射

    Returns:
        dict: {entrez_id: symbol} 映射字典
        dict: {symbol: entrez_id} 反向映射字典
    """
    if not GENE_INFO_FILE.exists():
        raise FileNotFoundError(f"NCBI gene_info file not found: {GENE_INFO_FILE}")

    entrez_to_symbol = {}
    symbol_to_entrez = {}
    synonyms_to_entrez = {}

    with gzip.open(GENE_INFO_FILE, 'rt') as f:
        header = f.readline()  # 跳过header
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue

            tax_id = fields[0]
            if tax_id != '9606':  # 只要人类基因
                continue

            entrez_id = fields[1]
            symbol = fields[2]
            synonyms = fields[4] if len(fields) > 4 else ''

            entrez_to_symbol[entrez_id] = symbol
            symbol_to_entrez[symbol.upper()] = entrez_id

            # 也记录同义词
            if synonyms and synonyms != '-':
                for syn in synonyms.split('|'):
                    syn = syn.strip()
                    if syn:
                        synonyms_to_entrez[syn.upper()] = entrez_id

    print(f"Loaded {len(entrez_to_symbol)} Entrez-to-Symbol mappings")
    return entrez_to_symbol, symbol_to_entrez, synonyms_to_entrez


def get_symbol(gene_id, entrez_to_symbol):
    """
    根据Entrez ID获取Symbol

    Args:
        gene_id: Entrez ID (字符串或数字)
        entrez_to_symbol: 映射字典

    Returns:
        str: Gene symbol, 或原ID如果找不到
    """
    gene_id_str = str(gene_id).strip()
    return entrez_to_symbol.get(gene_id_str, gene_id_str)


def get_entrez(symbol, symbol_to_entrez, synonyms_to_entrez=None):
    """
    根据Symbol获取Entrez ID

    Args:
        symbol: Gene symbol
        symbol_to_entrez: 映射字典
        synonyms_to_entrez: 同义词映射字典

    Returns:
        str: Entrez ID, 或None如果找不到
    """
    symbol_upper = symbol.upper().strip()

    # 先查主symbol
    if symbol_upper in symbol_to_entrez:
        return symbol_to_entrez[symbol_upper]

    # 再查同义词
    if synonyms_to_entrez and symbol_upper in synonyms_to_entrez:
        return synonyms_to_entrez[symbol_upper]

    return None


def map_magma_genes(magma_df, entrez_to_symbol):
    """
    为MAGMA结果添加Symbol列

    Args:
        magma_df: MAGMA输出DataFrame (需要有GENE列)
        entrez_to_symbol: 映射字典

    Returns:
        DataFrame: 添加了Symbol列的DataFrame
    """
    df = magma_df.copy()
    df['Symbol'] = df['GENE'].astype(str).apply(lambda x: get_symbol(x, entrez_to_symbol))

    # 统计映射成功率
    mapped = (df['Symbol'] != df['GENE'].astype(str)).sum()
    total = len(df)
    print(f"  Mapped {mapped}/{total} ({100*mapped/total:.1f}%) genes to symbols")

    return df


# 全局缓存
_MAPPING_CACHE = None

def get_mapping():
    """获取映射字典（带缓存）"""
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        _MAPPING_CACHE = load_ncbi_gene_mapping()
    return _MAPPING_CACHE


if __name__ == "__main__":
    # 测试
    print("Testing gene mapping...")
    entrez_to_symbol, symbol_to_entrez, synonyms = load_ncbi_gene_mapping()

    # 测试一些MAGMA中的基因
    test_ids = ['54361', '7490', '2099', '354', '1277', '2006']
    print("\nTest Entrez to Symbol:")
    for eid in test_ids:
        print(f"  {eid} -> {get_symbol(eid, entrez_to_symbol)}")

    # 测试反向
    test_symbols = ['WNT4', 'WT1', 'ESR1', 'COL1A1', 'ELN', 'LOXL1']
    print("\nTest Symbol to Entrez:")
    for sym in test_symbols:
        print(f"  {sym} -> {get_entrez(sym, symbol_to_entrez)}")
