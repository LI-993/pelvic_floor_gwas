#!/usr/bin/env python3
"""Download GTEx v8 eQTL archive (and optionally eQTLGen) for colocalization."""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REFERENCE_DIR

EQTL_DIR = REFERENCE_DIR / "gtex_eqtl"

GTEX_TAR_URL = "https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_Analysis_v8_eQTL.tar"

# Tissues most relevant to pelvic floor.
TISSUES: list[str] = [
    "Bladder", "Prostate", "Muscle_Skeletal",
    "Uterus", "Vagina", "Colon_Sigmoid", "Colon_Transverse",
]


def download(url: str, dest: Path, label: str) -> bool:
    print(f"Downloading {label}\n  URL: {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size / 1024**2:.1f} MB")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  Error: {e}")
        return False


def download_gtex() -> bool:
    """Idempotent: skip when an extracted tissue file is already present."""
    test = EQTL_DIR / "GTEx_Analysis_v8_eQTL" / "Muscle_Skeletal.v8.signif_variant_gene_pairs.txt.gz"
    if test.exists():
        print("GTEx v8 eQTL data already extracted")
        return True

    tar_path = EQTL_DIR / "GTEx_Analysis_v8_eQTL.tar"
    if not tar_path.exists():
        print("Downloading GTEx v8 eQTL archive (~1.3 GB)...")
        if not download(GTEX_TAR_URL, tar_path, "GTEx v8 eQTL"):
            return False

    print("\nExtracting...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(EQTL_DIR)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Extraction error: {e}")
        return False


def main() -> None:
    EQTL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GTEx eQTL Data Setup")
    print("=" * 60)

    if not download_gtex():
        return

    gtex_dir = EQTL_DIR / "GTEx_Analysis_v8_eQTL"
    if gtex_dir.exists():
        print(f"\nAvailable relevant tissues:")
        for tissue in TISSUES:
            f = gtex_dir / f"{tissue}.v8.signif_variant_gene_pairs.txt.gz"
            if f.exists():
                print(f"  {tissue}: {f.stat().st_size / 1024**2:.1f} MB")
            else:
                print(f"  {tissue}: missing")

    print(f"\nFiles location: {EQTL_DIR}")


if __name__ == "__main__":
    main()
