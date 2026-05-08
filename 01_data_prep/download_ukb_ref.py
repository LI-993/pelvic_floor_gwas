#!/usr/bin/env python3
"""Download UK-Biobank LAVA reference panels with resume + retry.

The reference is split across multiple zips on the SURFsara host. Each file is
~1-3 GB; resume support uses HTTP Range when the server allows it, otherwise
restarts from byte 0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REFERENCE_DIR

OUT_DIR = REFERENCE_DIR / "lava_ukb"
CHUNK_BYTES = 1024 * 1024  # 1 MB

FILES: dict[str, str] = {
    "lava-ukb-v1.1_chr1-2.zip": "https://vu.data.surfsara.nl/index.php/s/7NBVIvtPRdu7Qhz/download",
    "lava-ukb-v1.1_chr5-6.zip": "https://vu.data.surfsara.nl/index.php/s/mRz31q0lq7KMcuI/download",
    "lava-ukb-v1.1_chr10-12.zip": "https://vu.data.surfsara.nl/index.php/s/3dU1L2Hap43xuCs/download",
}

# Approximate sizes for "is this download already complete" detection.
EXPECTED_SIZES: dict[str, int] = {
    "lava-ukb-v1.1_chr1-2.zip": 3 * 1024**3,
    "lava-ukb-v1.1_chr5-6.zip": 2 * 1024**3,
    "lava-ukb-v1.1_chr10-12.zip": int(1.5 * 1024**3),
}


def download_file(url: str, filepath: Path, max_retries: int = 10) -> bool:
    """Download with resume on partial-content responses."""
    downloaded = filepath.stat().st_size if filepath.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    if downloaded:
        print(f"  Resuming from {downloaded / 1024**2:.1f} MB")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}...")
            response = requests.get(url, headers=headers, stream=True, timeout=30)

            if downloaded and response.status_code == 200:  # server ignored Range
                downloaded = 0
                headers = {}
                response = requests.get(url, stream=True, timeout=30)

            total = int(response.headers.get("content-length", 0)) + downloaded
            mode = "ab" if downloaded and response.status_code == 206 else "wb"
            if mode == "wb":
                downloaded = 0

            with open(filepath, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = 100 * downloaded / total if total else 0
                    print(
                        f"\r  {downloaded / 1024**2:.1f} / {total / 1024**2:.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

            print(f"\n  Completed: {filepath.name}")
            return True

        except Exception as e:  # noqa: BLE001 — broad except so retry logic kicks in
            print(f"\n  Error: {e}")
            downloaded = filepath.stat().st_size if filepath.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            time.sleep(5)

    print(f"  Failed after {max_retries} attempts")
    return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = sys.argv[1:] or list(FILES)
    for filename in targets:
        if filename not in FILES:
            print(f"Unknown file: {filename}")
            continue

        filepath = OUT_DIR / filename
        size = filepath.stat().st_size if filepath.exists() else 0
        expected = EXPECTED_SIZES.get(filename, 0)
        if size >= expected * 0.9:
            print(f"{filename}: already complete ({size / 1024**2:.1f} MB)")
            continue

        print(f"\nDownloading {filename}...")
        download_file(FILES[filename], filepath)


if __name__ == "__main__":
    main()
