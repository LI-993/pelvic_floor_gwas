#!/usr/bin/env python3
"""Download S-LDSC annotation files (baseline LD v2.2, weights, MAF) from Zenodo."""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SLDSC_ANNOT_DIR

DOWNLOADS: dict[str, str] = {
    "baselineLD_v2.2": "https://zenodo.org/records/10515792/files/1000G_Phase3_baselineLD_v2.2_ldscores.tgz?download=1",
    "weights": "https://zenodo.org/records/10515792/files/1000G_Phase3_weights_hm3_no_MHC.tgz?download=1",
    "frq": "https://zenodo.org/records/10515792/files/1000G_Phase3_frq.tgz?download=1",
}


def download_file(url: str, dest: Path) -> bool:
    print(f"Downloading: {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print("  Done.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  Error: {e}")
        return False


def extract(tgz: Path, dest_dir: Path) -> bool:
    print(f"Extracting {tgz.name}")
    result = subprocess.run(["tar", "-xzf", str(tgz), "-C", str(dest_dir)], capture_output=True, text=True)
    if result.returncode == 0:
        print("  Done.")
        return True
    print(f"  Error: {result.stderr}")
    return False


def main() -> None:
    SLDSC_ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    for label, url in DOWNLOADS.items():
        filename = url.split("/")[-1].split("?")[0]
        path = SLDSC_ANNOT_DIR / filename
        if path.exists():
            print(f"\n{label}: already downloaded ({filename})")
        else:
            print(f"\n{label}:")
            if download_file(url, path):
                extract(path, SLDSC_ANNOT_DIR)

    print(f"\nContents of {SLDSC_ANNOT_DIR}:")
    for item in sorted(SLDSC_ANNOT_DIR.iterdir()):
        if item.is_dir():
            print(f"  [DIR] {item.name}/")
        else:
            print(f"  {item.name} ({item.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
