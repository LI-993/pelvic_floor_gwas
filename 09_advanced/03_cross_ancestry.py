#!/usr/bin/env python3
"""Cross-ancestry validation against Pan-UKBB.

Downloads the per-trait Pan-UKBB sumstats files (each contains all ancestries
in beta_/se_/pval_ columns), summarizes GWAS-significant counts per ancestry,
and computes EUR-vs-other effect-size Pearson correlations.
"""

from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from utils.plotting import setup_publication_style

PAN_UKBB_DIR = DATA_DIR / "pan_ukbb"
OUT = RESULTS_DIR / "cross_ancestry"
FIG = FIGURES_DIR / "cross_ancestry"

S3_BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files"

KEY_PHENOTYPES: dict[str, dict] = {
    "N81":   {"name": "Female_genital_prolapse", "file": "icd10-N81-both_sexes.tsv.bgz", "populations": ["EUR", "AFR", "CSA"]},
    "N39":   {"name": "Urinary_disorders",       "file": "icd10-N39-both_sexes.tsv.bgz", "populations": ["EUR", "AFR", "CSA", "EAS", "MID"]},
    "N32":   {"name": "Bladder_disorders",       "file": "icd10-N32-both_sexes.tsv.bgz", "populations": ["EUR", "AFR", "CSA", "MID"]},
    "K59":   {"name": "Constipation",            "file": "icd10-K59-both_sexes.tsv.bgz", "populations": ["EUR", "AFR", "CSA", "EAS", "MID"]},
    "599.4": {"name": "Urinary_incontinence",    "file": "phecode-599.4-both_sexes.tsv.bgz", "populations": ["EUR", "AFR", "CSA"]},
}


def curl_download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"    Already exists: {dest.name}")
        return True
    print(f"    Downloading {dest.name}")
    try:
        r = subprocess.run(["curl", "-L", "-o", str(dest), url], capture_output=True, text=True, timeout=1200)
    except Exception as e:  # noqa: BLE001
        print(f"      Error: {e}")
        return False
    if r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
        print(f"      {dest.stat().st_size / 1e6:.1f} MB")
        return True
    print("      Failed")
    return False


def download_pan_ukbb() -> list[dict]:
    PAN_UKBB_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Downloading Pan-UKBB Multi-Ancestry Data")
    print("=" * 60)

    rows: list[dict] = []
    for phenocode, info in KEY_PHENOTYPES.items():
        url = f"{S3_BASE}/{info['file']}"
        dest = PAN_UKBB_DIR / info["file"]
        print(f"\n  {phenocode}: {info['name']}")
        if curl_download(url, dest):
            rows.append({"phenocode": phenocode, "name": info["name"], "file": str(dest), "populations": info["populations"]})

    if rows:
        pd.DataFrame(rows).to_csv(PAN_UKBB_DIR / "downloaded_files.csv", index=False)
    return rows


def replication_summary(downloaded: list[dict]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Cross-Ancestry Replication Summary")
    print("=" * 60)

    rows: list[dict] = []
    for item in downloaded:
        path = Path(item["file"])
        if not path.exists():
            continue
        print(f"\n  {item['name']}")

        # Identify which columns we need.
        with gzip.open(path, "rt") as f:
            header = f.readline().strip().split("\t")
        usecols = [c for c in ("chr", "pos", "ref", "alt") if c in header]
        for pop in item["populations"]:
            for prefix in ("beta_", "pval_"):
                col = f"{prefix}{pop}"
                if col in header and col not in usecols:
                    usecols.append(col)

        df = pd.read_csv(path, sep="\t", compression="gzip", usecols=usecols)
        for pop in item["populations"]:
            pval_col = f"pval_{pop}"
            if pval_col not in df.columns:
                continue
            n_valid = df[pval_col].notna().sum()
            n_sig = (df[pval_col] < 5e-8).sum()
            rows.append({"phenotype": item["name"], "population": pop, "n_variants": int(n_valid), "n_gwas_sig": int(n_sig)})
            print(f"    {pop}: {n_valid:,} variants | {n_sig} GWAS-significant")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT / "cross_ancestry_summary.csv", index=False)
    return df


def effect_size_correlations(downloaded: list[dict]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Effect-size correlation: EUR vs other ancestries")
    print("=" * 60)

    rows: list[dict] = []
    for item in downloaded:
        path = Path(item["file"])
        if not path.exists():
            continue
        print(f"\n  {item['name']}")
        with gzip.open(path, "rt") as f:
            header = f.readline().strip().split("\t")

        usecols: list[str] = []
        for col in ("chr", "pos", "ref", "alt"):
            if col in header:
                usecols.append(col)
        for pop in item["populations"]:
            for prefix in ("beta_", "pval_", "af_"):
                col = f"{prefix}{pop}"
                if col in header:
                    usecols.append(col)

        df = pd.read_csv(path, sep="\t", compression="gzip", usecols=usecols)
        eur_col = "beta_EUR"
        if eur_col not in df.columns:
            continue
        for pop in item["populations"]:
            if pop == "EUR":
                continue
            pop_col = f"beta_{pop}"
            if pop_col not in df.columns:
                continue
            valid = df[eur_col].notna() & df[pop_col].notna()
            sub = df[valid]
            if len(sub) < 100:
                continue
            corr = sub[eur_col].corr(sub[pop_col])
            rows.append({
                "phenotype": item["name"],
                "comparison": f"EUR_vs_{pop}",
                "n_variants": int(len(sub)),
                "effect_correlation": float(corr),
            })
            print(f"    EUR vs {pop}: r = {corr:.3f} ({len(sub):,} variants)")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT / "effect_size_correlations.csv", index=False)
    return df


def make_plots(replication_df: pd.DataFrame, correlation_df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    if not replication_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        replication_df.pivot(index="phenotype", columns="population", values="n_gwas_sig").plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of GWAS-significant variants (P < 5e-8)")
        ax.set_title("GWAS Signals Across Ancestries")
        ax.legend(title="Population")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIG / "gwas_signals_by_ancestry.png", dpi=300)
        plt.close()
        print("  Saved: gwas_signals_by_ancestry.png")

    if not correlation_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        pivot = correlation_df.pivot(index="phenotype", columns="comparison", values="effect_correlation")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("Effect-size correlation: EUR vs other ancestries")
        plt.tight_layout()
        fig.savefig(FIG / "effect_correlation_heatmap.png", dpi=300)
        plt.close()
        print("  Saved: effect_correlation_heatmap.png")


def main() -> None:
    setup_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pan-UKBB Cross-Ancestry Validation")
    print("=" * 60)

    downloaded = download_pan_ukbb()
    if not downloaded:
        print("No data downloaded.")
        return

    replication = replication_summary(downloaded)
    correlations = effect_size_correlations(downloaded)
    make_plots(replication, correlations)

    print(f"\nResults: {OUT}")
    print(f"Figures: {FIG}")


if __name__ == "__main__":
    main()
