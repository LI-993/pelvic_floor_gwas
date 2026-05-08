#!/usr/bin/env python3
"""Download the MAGMA Windows binary, gene-location file, and 1000G EUR LD reference."""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MAGMA_BIN, MAGMA_REF_DIR, TOOLS_DIR

MAGMA_DIR = TOOLS_DIR / "magma"

# Direct download URLs from cncr.nl (hosted on SURFsara).
DOWNLOADS: dict[str, str] = {
    "magma_win": "https://vu.data.surfsara.nl/index.php/s/TOH4SuvczAKE29d/download",
    "gene_loc": "https://vu.data.surfsara.nl/index.php/s/Pj2orwuF2JYyKxq/download",
    "g1000_eur": "https://vu.data.surfsara.nl/index.php/s/VZNByNwpD8qqINe/download",
}


def download_file(url: str, dest: Path) -> bool:
    print(f"Downloading: {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size / 1024**2:.1f} MB")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  Error: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  Error: {e}")
        return False


def cleanup_failed(dirs: tuple[Path, ...]) -> None:
    for d in dirs:
        if not d.exists():
            continue
        for old in d.glob("*.zip"):
            if old.stat().st_size < 1_000_000:  # < 1 MB likely failed
                old.unlink()
                print(f"Removed invalid file: {old}")


def main() -> None:
    MAGMA_DIR.mkdir(parents=True, exist_ok=True)
    MAGMA_REF_DIR.mkdir(parents=True, exist_ok=True)

    cleanup_failed((MAGMA_DIR, MAGMA_REF_DIR))

    if not MAGMA_BIN.exists():
        print("\n[1/3] MAGMA binary")
        zip_path = MAGMA_DIR / "magma_v1.10_win.zip"
        if download_file(DOWNLOADS["magma_win"], zip_path):
            extract_zip(zip_path, MAGMA_DIR)
    else:
        print(f"\nMAGMA already installed: {MAGMA_BIN}")

    gene_file = MAGMA_REF_DIR / "NCBI37.3.gene.loc"
    if not gene_file.exists():
        print("\n[2/3] Gene-location file (Build 37)")
        zip_path = MAGMA_REF_DIR / "NCBI37.3.zip"
        if download_file(DOWNLOADS["gene_loc"], zip_path):
            extract_zip(zip_path, MAGMA_REF_DIR)
    else:
        print(f"\nGene-loc already exists: {gene_file}")

    g1000_bed = MAGMA_REF_DIR / "g1000_eur.bed"
    if not g1000_bed.exists():
        print("\n[3/3] 1000G EUR reference")
        zip_path = MAGMA_REF_DIR / "g1000_eur.zip"
        if download_file(DOWNLOADS["g1000_eur"], zip_path):
            extract_zip(zip_path, MAGMA_REF_DIR)
    else:
        print(f"\n1000G EUR already exists: {g1000_bed}")

    print(f"\nReference files in {MAGMA_REF_DIR}:")
    for item in MAGMA_REF_DIR.iterdir():
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
