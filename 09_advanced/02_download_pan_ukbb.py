#!/usr/bin/env python3
"""Download pelvic-floor-relevant Pan-UKBB sumstats across ancestries.

Pan-UKBB hosts the same trait across six ancestries (EUR / AFR / EAS / CSA /
MID / AMR) plus a meta-analysis. The script downloads the phenotype manifest,
filters by ICD-10 codes and keyword matches, and writes the manifest subset
plus a download guide. The actual per-trait downloads happen lazily when the
test downloader at the end fires for the first relevant row.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

OUT = DATA_DIR / "pan_ukbb"
S3_BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com"
MANIFEST_URL = f"{S3_BASE}/sumstats_release/phenotype_manifest.tsv.bgz"

TARGET_KEYWORDS = ("prolapse", "incontinence", "prostat", "bladder", "constipation", "urinary", "pelvic")
TARGET_ICD10 = ("N81", "N39", "N40", "N32", "K59", "R32", "N31")
POPULATIONS = ("EUR", "AFR", "EAS", "CSA", "MID", "AMR", "meta")


def curl_download(url: str, dest: Path) -> bool:
    print(f"  Downloading: {url.split('/')[-1]}")
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    Error: {e}")
        return False
    if result.returncode == 0 and dest.exists() and dest.stat().st_size > 100:
        print(f"    {dest.stat().st_size / 1e6:.2f} MB")
        return True
    print("    Failed")
    return False


def fetch_manifest() -> pd.DataFrame | None:
    path = OUT / "phenotype_manifest.tsv.bgz"
    if not path.exists() and not curl_download(MANIFEST_URL, path):
        return None
    df = pd.read_csv(path, sep="\t", compression="gzip")
    print(f"  Manifest rows: {len(df)}")
    return df


def filter_relevant(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        phenocode = str(row.get("phenocode", "")).upper()
        desc = str(row.get("description", "")).lower()
        if any(icd in phenocode for icd in TARGET_ICD10) or any(kw in desc for kw in TARGET_KEYWORDS):
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"  Relevant phenotypes: {len(df)}")
    return df


def write_download_guide(relevant: pd.DataFrame) -> None:
    path = OUT / "download_guide.txt"
    with open(path, "w") as fh:
        fh.write("Pan-UKBB Download Guide\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Base URL: {S3_BASE}/sumstats_release/\n\n")
        fh.write("Relevant phenotypes:\n")
        for _, row in relevant.iterrows():
            fh.write(f"  {row.get('phenocode', 'NA')}: {row.get('description', 'NA')}\n")
            if "aws_path" in row.index:
                fh.write(f"    Path: {row.get('aws_path', 'NA')}\n")
    print(f"  Guide: {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pan-UKBB Multi-Ancestry Sumstats Download")
    print("=" * 60)

    manifest = fetch_manifest()
    if manifest is None:
        return

    relevant = filter_relevant(manifest)
    if relevant.empty:
        return
    relevant.to_csv(OUT / "relevant_phenotypes.csv", index=False)

    cols = [c for c in ["phenocode", "description", "trait_type"] if c in relevant.columns]
    if cols:
        print("\nRelevant phenotypes:")
        print(relevant[cols].head(20).to_string())

    print("\nPer-population data availability:")
    for pop in POPULATIONS:
        col = f"n_cases_{pop}"
        if col in relevant.columns:
            n = (relevant[col].notna() & (relevant[col] > 0)).sum()
            print(f"  {pop}: {n} phenotypes")

    write_download_guide(relevant)

    if "aws_path" in relevant.columns:
        first = relevant.iloc[0]
        path = first.get("aws_path")
        if pd.notna(path):
            url = f"{S3_BASE}/sumstats_release/{path}"
            dest = OUT / Path(path).name
            print(f"\nTest download:\n  {url}")
            curl_download(url, dest)


if __name__ == "__main__":
    main()
