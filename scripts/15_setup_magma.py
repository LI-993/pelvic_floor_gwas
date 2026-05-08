#!/usr/bin/env python3
"""Download and setup MAGMA for gene-based analysis."""

import os
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
TOOLS_DIR = BASE_DIR / "tools"
MAGMA_DIR = TOOLS_DIR / "magma"
REF_DIR = BASE_DIR / "reference/magma"

MAGMA_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# MAGMA downloads from https://cncr.nl/research/magma/ (hosted on SURF)
# These are direct download links
DOWNLOADS = {
    # MAGMA Windows binary v1.10 (64-bit, static)
    "magma_win": "https://vu.data.surfsara.nl/index.php/s/TOH4SuvczAKE29d/download",

    # Gene location file (GRCh37/Build 37)
    "gene_loc": "https://vu.data.surfsara.nl/index.php/s/Pj2orwuF2JYyKxq/download",

    # 1000G EUR reference for LD
    "g1000_eur": "https://vu.data.surfsara.nl/index.php/s/VZNByNwpD8qqINe/download",
}

def download_file(url, dest_path):
    """Download file with progress."""
    print(f"Downloading: {url}")
    print(f"To: {dest_path}")

    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  Downloaded: {dest_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def extract_zip(zip_path, dest_dir):
    """Extract zip file."""
    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        print(f"  Extracted to: {dest_dir}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("="*60)
    print("MAGMA Setup")
    print("="*60)

    # Clean up any previous failed downloads
    for old_file in MAGMA_DIR.glob("*.zip"):
        if old_file.stat().st_size < 1000000:  # Less than 1MB probably failed
            old_file.unlink()
            print(f"Removed invalid file: {old_file}")
    for old_file in REF_DIR.glob("*.zip"):
        if old_file.stat().st_size < 1000000:
            old_file.unlink()
            print(f"Removed invalid file: {old_file}")

    # Check if MAGMA already exists
    magma_exe = MAGMA_DIR / "magma.exe"
    if magma_exe.exists():
        print(f"\nMAGMA already installed: {magma_exe}")
    else:
        # Download MAGMA binary
        print("\n[1/3] Downloading MAGMA binary...")
        magma_zip = MAGMA_DIR / "magma_v1.10_win.zip"
        if download_file(DOWNLOADS["magma_win"], magma_zip):
            extract_zip(magma_zip, MAGMA_DIR)

    # Download gene location file
    print("\n[2/3] Downloading gene location file (Build 37)...")
    gene_file = REF_DIR / "NCBI37.3.gene.loc"
    if not gene_file.exists():
        gene_zip = REF_DIR / "NCBI37.3.zip"
        if download_file(DOWNLOADS["gene_loc"], gene_zip):
            extract_zip(gene_zip, REF_DIR)
    else:
        print(f"  Already exists: {gene_file}")

    # Download 1000G EUR reference
    print("\n[3/3] Downloading 1000G EUR reference...")
    g1000_file = REF_DIR / "g1000_eur.bed"
    if not g1000_file.exists():
        g1000_zip = REF_DIR / "g1000_eur.zip"
        if download_file(DOWNLOADS["g1000_eur"], g1000_zip):
            extract_zip(g1000_zip, REF_DIR)
    else:
        print(f"  Already exists: {g1000_file}")

    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)

    # Check MAGMA executable
    magma_exe = MAGMA_DIR / "magma.exe"
    if magma_exe.exists():
        print(f"\nMAGMA executable: {magma_exe}")
    else:
        # Check in subdirectory
        for f in MAGMA_DIR.rglob("magma.exe"):
            print(f"\nMAGMA executable: {f}")
            break

    # List reference files
    print(f"\nReference files in {REF_DIR}:")
    for f in REF_DIR.iterdir():
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
