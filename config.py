"""Project-wide constants and paths.

BASE_DIR is resolved from this file's location, so scripts run from any cwd.
External tool paths (LDSC, MAGMA, MTAG, references) come from env vars when set,
otherwise fall back to the historical locations on the original machine. Override
by exporting the corresponding variable before running a script.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LDSC_DATA_DIR = DATA_DIR / "ldsc"
LAVA_DATA_DIR = DATA_DIR / "lava"

REFERENCE_DIR = BASE_DIR / "reference"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
TOOLS_DIR = BASE_DIR / "tools"


# ---------------------------------------------------------------------------
# Phenotypes
# ---------------------------------------------------------------------------

# Canonical order used across analyses, tables, and plots.
PHENOTYPES: list[str] = [
    "POP",
    "BPH",
    "Bladder",
    "Constipation",
    "FemaleProlapse",
    "Incontinence",
]

# Total N = cases + controls (used by LDSC munge, MTAG, MAGMA).
SAMPLE_SIZES: dict[str, int] = {
    "POP": 574377,
    "BPH": 501137,
    "Bladder": 503550,
    "Constipation": 501956,
    "FemaleProlapse": 503074,
    "Incontinence": 430019,
}

# Case / control split (LAVA input.info needs this).
CASES_CONTROLS: dict[str, tuple[int, int]] = {
    "POP": (28086, 546291),
    "BPH": (41137, 460000),
    "Bladder": (3550, 500000),
    "Constipation": (51956, 450000),
    "FemaleProlapse": (23074, 480000),
    "Incontinence": (27714, 402305),
}


def sumstats_path(phenotype: str, build: str = "GRCh38") -> Path:
    """Standardized sumstats path produced by the data-prep stage."""
    return PROCESSED_DIR / f"{phenotype}_{build}.tsv.gz"


def ldsc_sumstats_path(phenotype: str) -> Path:
    """LDSC-formatted sumstats produced by 04_prepare_ldsc."""
    return LDSC_DATA_DIR / f"{phenotype}.sumstats.gz"


# ---------------------------------------------------------------------------
# External tool paths (override via env vars)
# ---------------------------------------------------------------------------

def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default)))


# LDSC (Python 3 fork)
LDSC_PYTHON = _env_path("LDSC_PYTHON", Path(r"D:\miniconda3\envs\ldsc_py311\python.exe"))
LDSC_SCRIPT = _env_path("LDSC_SCRIPT", Path(r"D:\Nproject\gwas\ldsc-python3\ldsc.py"))
MUNGE_SCRIPT = _env_path("MUNGE_SCRIPT", Path(r"D:\Nproject\gwas\ldsc-python3\munge_sumstats.py"))
LDSC_REF_DIR = _env_path("LDSC_REF_DIR", Path(r"D:\Nproject\gwas\gwas_stroke_incontinence\reference\eur_w_ld_chr"))
HM3_SNPLIST = LDSC_REF_DIR / "w_hm3.snplist"

# S-LDSC annotations (downloaded by 12_download_sldsc_data)
SLDSC_ANNOT_DIR = REFERENCE_DIR / "ldsc_annotations"
SLDSC_BASELINE_PREFIX = SLDSC_ANNOT_DIR / "baselineLD."
SLDSC_WEIGHTS_PREFIX = SLDSC_ANNOT_DIR / "1000G_Phase3_weights_hm3_no_MHC" / "weights.hm3_noMHC."
SLDSC_FRQ_PREFIX = SLDSC_ANNOT_DIR / "1000G_Phase3_frq" / "1000G.EUR.QC."

# MTAG
MTAG_DIR = _env_path("MTAG_DIR", BASE_DIR / "mtag")
MTAG_LD_REF = MTAG_DIR / "ld_ref_panel" / "eur_w_ld_chr/"

# MAGMA
MAGMA_BIN = _env_path("MAGMA_BIN", TOOLS_DIR / "magma" / "magma.exe")
MAGMA_REF_DIR = _env_path("MAGMA_REF_DIR", REFERENCE_DIR / "magma")
MAGMA_GENE_LOC = MAGMA_REF_DIR / "NCBI37.3.gene.loc"
MAGMA_G1000_EUR = MAGMA_REF_DIR / "g1000_eur"

# LAVA (R-only analysis; results consumed by Python visualization scripts)
LAVA_REF_PREFIX = _env_path("LAVA_REF_PREFIX", REFERENCE_DIR / "lava_ukb" / "lava-ukb-v1.1")
LAVA_LOCUS_FILE = _env_path(
    "LAVA_LOCUS_FILE",
    BASE_DIR / "LAVA-main" / "support_data" / "blocks_s2500_m25_f1_w200.GRCh37_hg19.locfile",
)

# OpenGWAS — set OPENGWAS_JWT in the environment, never commit it.
OPENGWAS_JWT = os.environ.get("OPENGWAS_JWT", "")


# ---------------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------------

PHENOTYPE_COLORS: dict[str, str] = {
    "POP": "#E64B35",
    "BPH": "#4DBBD5",
    "Bladder": "#00A087",
    "Constipation": "#3C5488",
    "FemaleProlapse": "#F39B7F",
    "Incontinence": "#8491B4",
}

PHENOTYPE_SHORT: dict[str, str] = {
    "POP": "POP",
    "BPH": "BPH",
    "Bladder": "Bladder",
    "Constipation": "Constip.",
    "FemaleProlapse": "F.Prolapse",
    "Incontinence": "Incontin.",
}

# Approximate GRCh38 chromosome lengths for Manhattan x-axis layout.
CHR_LENGTHS_GRCh38: dict[int, int] = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555,
    5: 181538259, 6: 170805979, 7: 159345973, 8: 145138636,
    9: 138394717, 10: 133797422, 11: 135086622, 12: 133275309,
    13: 114364328, 14: 107043718, 15: 101991189, 16: 90338345,
    17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}
