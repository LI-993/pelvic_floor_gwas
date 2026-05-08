#!/usr/bin/env python3
"""Download UK Biobank reference data for LAVA with retry support."""

import requests
import os
import sys
from pathlib import Path
import time

OUT_DIR = Path(r"D:\Nproject\gwas\pelvic_floor_gwas\reference\lava_ukb")

# Files to download
FILES = {
    "lava-ukb-v1.1_chr1-2.zip": "https://vu.data.surfsara.nl/index.php/s/7NBVIvtPRdu7Qhz/download",
    "lava-ukb-v1.1_chr5-6.zip": "https://vu.data.surfsara.nl/index.php/s/mRz31q0lq7KMcuI/download",
    "lava-ukb-v1.1_chr10-12.zip": "https://vu.data.surfsara.nl/index.php/s/3dU1L2Hap43xuCs/download",
}

def download_file(url, filename, max_retries=10):
    """Download file with resume support and retries."""
    filepath = OUT_DIR / filename

    # Get current file size if exists
    downloaded = filepath.stat().st_size if filepath.exists() else 0

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"  Resuming from {downloaded / 1024 / 1024:.1f} MB")

    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")

            response = requests.get(url, headers=headers, stream=True, timeout=30)

            # Check if server supports resume
            if downloaded > 0 and response.status_code == 200:
                # Server doesn't support resume, start from beginning
                downloaded = 0
                headers = {}
                response = requests.get(url, stream=True, timeout=30)

            total_size = int(response.headers.get('content-length', 0)) + downloaded

            mode = 'ab' if downloaded > 0 and response.status_code == 206 else 'wb'
            if mode == 'wb':
                downloaded = 0

            with open(filepath, mode) as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = 100 * downloaded / total_size if total_size > 0 else 0
                        print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="", flush=True)

            print(f"\n  Completed: {filename}")
            return True

        except Exception as e:
            print(f"\n  Error: {e}")
            downloaded = filepath.stat().st_size if filepath.exists() else 0
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
            time.sleep(5)  # Wait before retry

    print(f"  Failed after {max_retries} attempts")
    return False

def main():
    # Only download file specified in command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if filename in FILES:
            print(f"Downloading {filename}...")
            download_file(FILES[filename], filename)
        else:
            print(f"Unknown file: {filename}")
    else:
        # Download all incomplete files
        for filename, url in FILES.items():
            filepath = OUT_DIR / filename
            expected_sizes = {
                "lava-ukb-v1.1_chr1-2.zip": 3 * 1024**3,  # ~3GB
                "lava-ukb-v1.1_chr5-6.zip": 2 * 1024**3,  # ~2GB
                "lava-ukb-v1.1_chr10-12.zip": 1.5 * 1024**3,  # ~1.5GB
            }

            current_size = filepath.stat().st_size if filepath.exists() else 0
            expected = expected_sizes.get(filename, 0)

            if current_size < expected * 0.9:  # Less than 90% complete
                print(f"\nDownloading {filename}...")
                download_file(url, filename)
            else:
                print(f"{filename}: Already complete ({current_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
