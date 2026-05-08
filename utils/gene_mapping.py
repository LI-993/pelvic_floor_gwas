"""NCBI gene-info Entrez<->Symbol mapping.

Loads `Homo_sapiens.gene_info.gz` and exposes lookup helpers. MAGMA outputs
Entrez IDs; downstream tools (DGIdb, STRING, plotting) need symbols.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from config import DATA_DIR

GENE_INFO_FILE = DATA_DIR / "reference" / "Homo_sapiens.gene_info.gz"
HUMAN_TAX_ID = "9606"


def load_ncbi_gene_mapping(
    gene_info_file: Path = GENE_INFO_FILE,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build Entrez<->Symbol dicts from NCBI gene_info.

    Returns (entrez_to_symbol, symbol_to_entrez, synonyms_to_entrez). Synonym
    lookups handle gene aliases reported in the 5th column of gene_info.
    """
    if not gene_info_file.exists():
        raise FileNotFoundError(f"NCBI gene_info file not found: {gene_info_file}")

    entrez_to_symbol: dict[str, str] = {}
    symbol_to_entrez: dict[str, str] = {}
    synonyms_to_entrez: dict[str, str] = {}

    with gzip.open(gene_info_file, "rt") as f:
        next(f)  # header
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[0] != HUMAN_TAX_ID:
                continue

            entrez_id, symbol = fields[1], fields[2]
            synonyms = fields[4] if len(fields) > 4 else ""

            entrez_to_symbol[entrez_id] = symbol
            symbol_to_entrez[symbol.upper()] = entrez_id

            if synonyms and synonyms != "-":
                for syn in synonyms.split("|"):
                    syn = syn.strip().upper()
                    if syn:
                        synonyms_to_entrez.setdefault(syn, entrez_id)

    print(f"Loaded {len(entrez_to_symbol)} Entrez-to-Symbol mappings")
    return entrez_to_symbol, symbol_to_entrez, synonyms_to_entrez


def get_symbol(gene_id: str | int, entrez_to_symbol: dict[str, str]) -> str:
    """Symbol lookup; returns the input ID unchanged if not found."""
    gid = str(gene_id).strip()
    return entrez_to_symbol.get(gid, gid)


def get_entrez(
    symbol: str,
    symbol_to_entrez: dict[str, str],
    synonyms_to_entrez: dict[str, str] | None = None,
) -> str | None:
    """Entrez lookup, falling back to synonyms when provided."""
    key = symbol.upper().strip()
    if key in symbol_to_entrez:
        return symbol_to_entrez[key]
    if synonyms_to_entrez and key in synonyms_to_entrez:
        return synonyms_to_entrez[key]
    return None


def map_magma_genes(magma_df: pd.DataFrame, entrez_to_symbol: dict[str, str]) -> pd.DataFrame:
    """Add a Symbol column to a MAGMA gene-level results DataFrame."""
    df = magma_df.copy()
    df["Symbol"] = df["GENE"].astype(str).map(lambda x: get_symbol(x, entrez_to_symbol))

    mapped = (df["Symbol"] != df["GENE"].astype(str)).sum()
    total = len(df)
    if total:
        print(f"  Mapped {mapped}/{total} ({100 * mapped / total:.1f}%) genes to symbols")
    return df


_CACHE: tuple[dict, dict, dict] | None = None


def get_mapping() -> tuple[dict, dict, dict]:
    """Lazy-cache the mapping dicts; first call reads the gene_info file."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_ncbi_gene_mapping()
    return _CACHE


if __name__ == "__main__":
    entrez_to_symbol, symbol_to_entrez, synonyms = load_ncbi_gene_mapping()

    print("\nEntrez -> Symbol:")
    for eid in ["54361", "7490", "2099", "354", "1277", "2006"]:
        print(f"  {eid} -> {get_symbol(eid, entrez_to_symbol)}")

    print("\nSymbol -> Entrez:")
    for sym in ["WNT4", "WT1", "ESR1", "COL1A1", "ELN", "LOXL1"]:
        print(f"  {sym} -> {get_entrez(sym, symbol_to_entrez, synonyms)}")
