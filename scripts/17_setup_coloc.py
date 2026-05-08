#!/usr/bin/env python3
"""
Download GTEx v8 eQTL data for colocalization analysis.

Relevant tissues for pelvic floor disorders:
- Bladder
- Prostate
- Muscle_Skeletal
- Uterus
- Vagina
- Colon_Sigmoid
- Colon_Transverse
"""

import os
import urllib.request
import gzip
import shutil
import tarfile
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
EQTL_DIR = BASE_DIR / "reference/gtex_eqtl"
EQTL_DIR.mkdir(parents=True, exist_ok=True)

# GTEx v8 eQTL data - full archive (~1.3GB)
# Contains all 49 tissues with signif_variant_gene_pairs.txt.gz
GTEX_TAR_URL = "https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_Analysis_v8_eQTL.tar"

# Relevant tissues for pelvic floor analysis
TISSUES = [
    "Bladder",
    "Prostate",
    "Muscle_Skeletal",
    "Uterus",
    "Vagina",
    "Colon_Sigmoid",
    "Colon_Transverse",
]

# Alternative: eQTLGen for blood (larger sample size, N=31,684)
# Updated URL from eQTLGen website
EQTLGEN_URL = "https://molgenis26.gcc.rug.nl/downloads/eqtlgen/cis-eqtl/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonijtertsFiltered.txt.gz"


def download_file(url, dest_path, description=""):
    """Download file with progress indicator."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  To: {dest_path}")

    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = dest_path.stat().st_size / 1024 / 1024
        print(f"  Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_gtex_tar():
    """Download and extract GTEx v8 eQTL tar archive."""
    tar_path = EQTL_DIR / "GTEx_Analysis_v8_eQTL.tar"

    # Check if we already have the extracted files
    test_file = EQTL_DIR / "GTEx_Analysis_v8_eQTL" / "Muscle_Skeletal.v8.signif_variant_gene_pairs.txt.gz"
    if test_file.exists():
        print("GTEx v8 eQTL data already extracted")
        return True

    # Download if not exists
    if not tar_path.exists():
        print("Downloading GTEx v8 eQTL archive (~1.3GB)...")
        print("This may take a while...")
        if not download_file(GTEX_TAR_URL, tar_path, "GTEx v8 eQTL"):
            return False

    # Extract
    print("\nExtracting tar archive...")
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(EQTL_DIR)
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def download_eqtlgen():
    """Download eQTLGen cis-eQTL data (blood)."""
    dest = EQTL_DIR / "eQTLGen_cis_eQTLs.txt.gz"

    if dest.exists():
        print("eQTLGen data already exists")
        return True

    return download_file(EQTLGEN_URL, dest, "eQTLGen (blood)")


def main():
    print("="*60)
    print("GTEx eQTL Data Download")
    print("="*60)

    # Download GTEx tar archive
    print("\n[Step 1] Downloading GTEx v8 eQTL data...")
    gtex_success = download_gtex_tar()

    # Check which relevant tissues are available
    if gtex_success:
        print("\n[Step 2] Checking available tissues...")
        gtex_dir = EQTL_DIR / "GTEx_Analysis_v8_eQTL"
        if gtex_dir.exists():
            available = []
            for tissue in TISSUES:
                tissue_file = gtex_dir / f"{tissue}.v8.signif_variant_gene_pairs.txt.gz"
                if tissue_file.exists():
                    size_mb = tissue_file.stat().st_size / 1024 / 1024
                    print(f"  {tissue}: {size_mb:.1f} MB")
                    available.append(tissue)
                else:
                    print(f"  {tissue}: NOT FOUND")
            print(f"\nAvailable relevant tissues: {len(available)}/{len(TISSUES)}")

    # Optional: Download eQTLGen
    print("\n[Step 3] eQTLGen data (optional)...")
    print("Skipping eQTLGen download (use GTEx tissues instead)")
    # download_eqtlgen()

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nFiles location: {EQTL_DIR}")


if __name__ == "__main__":
    main()
