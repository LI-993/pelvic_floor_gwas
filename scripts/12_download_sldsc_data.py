#!/usr/bin/env python3
"""Download S-LDSC annotation data for partitioned heritability analysis."""

import os
import subprocess
import urllib.request
from pathlib import Path

BASE_DIR = Path("D:/Nproject/gwas/pelvic_floor_gwas")
ANNOT_DIR = BASE_DIR / "reference/ldsc_annotations"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

# URLs for LDSC annotation files (from Broad Institute)
# https://alkesgroup.broadinstitute.org/LDSCORE/

DOWNLOADS = {
    # Baseline model v2.2 (97 annotations) - from Zenodo
    "baselineLD_v2.2": "https://zenodo.org/records/10515792/files/1000G_Phase3_baselineLD_v2.2_ldscores.tgz?download=1",

    # Weights for regression - from Zenodo
    "weights": "https://zenodo.org/records/10515792/files/1000G_Phase3_weights_hm3_no_MHC.tgz?download=1",

    # Frequency files - from Zenodo
    "frq": "https://zenodo.org/records/10515792/files/1000G_Phase3_frq.tgz?download=1",
}

def download_file(url, dest_path):
    """Download file with progress."""
    print(f"Downloading: {url}")
    print(f"To: {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  Done!")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def extract_tgz(filepath, dest_dir):
    """Extract .tgz file."""
    print(f"Extracting: {filepath}")
    cmd = f'tar -xzf "{filepath}" -C "{dest_dir}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Extracted successfully")
        return True
    else:
        print(f"  Error: {result.stderr}")
        return False

def main():
    print("="*60)
    print("S-LDSC Annotation Data Download")
    print("="*60)

    os.chdir(ANNOT_DIR)

    for name, url in DOWNLOADS.items():
        # Handle Zenodo URLs with query parameters
        filename = url.split("/")[-1].split("?")[0]
        filepath = ANNOT_DIR / filename

        if filepath.exists():
            print(f"\n{name}: Already downloaded ({filename})")
        else:
            print(f"\n{name}:")
            if download_file(url, filepath):
                extract_tgz(filepath, ANNOT_DIR)

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)

    # List downloaded contents
    print("\nContents of annotation directory:")
    for item in sorted(ANNOT_DIR.iterdir()):
        if item.is_dir():
            print(f"  [DIR] {item.name}/")
        else:
            size_mb = item.stat().st_size / (1024*1024)
            print(f"  {item.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
