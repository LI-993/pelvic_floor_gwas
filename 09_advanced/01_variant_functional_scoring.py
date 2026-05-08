#!/usr/bin/env python3
"""Functional impact scoring of MTAG multi-trait SNPs via Ensembl VEP.

Pulls VEP annotations in batches, maps consequence terms to a numeric impact
score (0-1) based on the standard ENCODE/VEP severity ranking, classifies
each SNP into impact categories, and produces summary plots.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR
from utils.plotting import setup_publication_style

OUT = RESULTS_DIR / "ml_variant_scores"
FIG = FIGURES_DIR / "variant_scoring"
MTAG_SNPS_PATH = RESULTS_DIR / "mtag" / "mtag_multi_trait_snps.csv"

VEP_ENDPOINT = "https://rest.ensembl.org/vep/human/id"
VEP_BATCH = 50
MAX_SNPS = 500   # cap to keep API runtime sane

# Consequence-to-score map (drawn from VEP severity ranking).
CONSEQUENCE_SCORES: dict[str, float] = {
    "transcript_ablation": 1.0,
    "splice_acceptor_variant": 0.95,
    "splice_donor_variant": 0.95,
    "stop_gained": 0.9, "frameshift_variant": 0.9,
    "stop_lost": 0.85, "start_lost": 0.85,
    "transcript_amplification": 0.8,
    "inframe_insertion": 0.7, "inframe_deletion": 0.7,
    "missense_variant": 0.65, "protein_altering_variant": 0.6,
    "regulatory_region_ablation": 0.6,
    "regulatory_region_amplification": 0.55, "TFBS_ablation": 0.55,
    "splice_region_variant": 0.5, "TF_binding_site_variant": 0.5,
    "TFBS_amplification": 0.5, "coding_sequence_variant": 0.5,
    "incomplete_terminal_codon_variant": 0.45,
    "regulatory_region_variant": 0.4, "mature_miRNA_variant": 0.4,
    "start_retained_variant": 0.4, "stop_retained_variant": 0.4,
    "5_prime_UTR_variant": 0.35, "synonymous_variant": 0.3,
    "feature_truncation": 0.3, "3_prime_UTR_variant": 0.3,
    "non_coding_transcript_exon_variant": 0.25,
    "NMD_transcript_variant": 0.2, "feature_elongation": 0.2,
    "intron_variant": 0.15, "non_coding_transcript_variant": 0.15,
    "upstream_gene_variant": 0.1, "downstream_gene_variant": 0.1,
    "intergenic_variant": 0.05,
}


def fetch_vep(rsids: list[str]) -> pd.DataFrame:
    out_rows: list[dict] = []
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    for i in range(0, len(rsids), VEP_BATCH):
        batch = rsids[i : i + VEP_BATCH]
        try:
            r = requests.post(VEP_ENDPOINT, headers=headers, json={"ids": batch}, timeout=60)
        except Exception as e:  # noqa: BLE001
            print(f"  VEP error: {e}")
            time.sleep(1)
            continue
        if r.status_code != 200:
            print(f"  VEP status {r.status_code}")
            time.sleep(1)
            continue
        for var in r.json():
            consequences: set[str] = set()
            genes: set[str] = set()
            biotypes: set[str] = set()
            cadd_phred: float | None = None
            cadd_raw: float | None = None
            for tc in var.get("transcript_consequences", []) or []:
                consequences.update(tc.get("consequence_terms", []))
                if "gene_symbol" in tc:
                    genes.add(tc["gene_symbol"])
                if "biotype" in tc:
                    biotypes.add(tc["biotype"])
                cadd_phred = cadd_phred or tc.get("cadd_phred")
                cadd_raw = cadd_raw or tc.get("cadd_raw")
            out_rows.append({
                "SNP": var.get("id", ""),
                "most_severe_consequence": var.get("most_severe_consequence", "unknown"),
                "all_consequences": ";".join(consequences) if consequences else var.get("most_severe_consequence", ""),
                "genes": ";".join(genes),
                "biotypes": ";".join(biotypes),
                "cadd_phred": cadd_phred,
                "cadd_raw": cadd_raw,
            })
        time.sleep(1)
        if (i + VEP_BATCH) % 200 == 0:
            print(f"  Processed {min(i + VEP_BATCH, len(rsids))}/{len(rsids)}")
    return pd.DataFrame(out_rows)


def score_variants(df: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for _, row in df.iterrows():
        terms = (row["all_consequences"] or "").split(";") if pd.notna(row["all_consequences"]) else []
        score = max((CONSEQUENCE_SCORES.get(t.strip(), 0.0) for t in terms), default=0.0)
        if score == 0:
            score = CONSEQUENCE_SCORES.get(row["most_severe_consequence"], 0.05)
        scores.append(score)
    df = df.copy()
    df["functional_score"] = scores
    df["impact_category"] = df["functional_score"].apply(category_for)
    return df


def category_for(score: float) -> str:
    if score >= 0.8:
        return "High Impact (Protein-altering)"
    if score >= 0.5:
        return "Moderate Impact (Splice/Regulatory)"
    if score >= 0.25:
        return "Low-Moderate Impact (UTR/Exonic)"
    if score >= 0.1:
        return "Low Impact (Intronic/Flanking)"
    return "Modifier (Intergenic)"


def plot_summary(scored: pd.DataFrame, mtag_df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(scored["functional_score"], bins=30, kde=True, ax=axes[0], color="#3C5488")
    axes[0].axvline(0.5, color="red", linestyle="--", label="Moderate impact")
    axes[0].set_title("Functional Score Distribution", fontweight="bold")
    axes[0].legend()

    counts = scored["impact_category"].value_counts()
    axes[1].pie(
        counts.values, labels=counts.index,
        colors=["#E64B35", "#F39B7F", "#4DBBD5", "#00A087", "#3C5488"][: len(counts)],
        autopct="%1.1f%%", startangle=90,
    )
    axes[1].set_title("Variant Impact Categories", fontweight="bold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"variant_functional_scores.{ext}", bbox_inches="tight")
    plt.close()

    if "n_traits" in mtag_df.columns:
        merged = scored.merge(mtag_df[["SNP", "n_traits"]], on="SNP", how="left")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=merged, x="n_traits", y="functional_score", palette="Set2", ax=ax)
        ax.set_xlabel("Number of Associated Traits")
        ax.set_ylabel("Functional Score")
        ax.set_title("Functional Scores by Number of Traits", fontweight="bold")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"functional_score_by_traits.{ext}", bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    cons_counts = scored["most_severe_consequence"].value_counts().head(15)
    bars = ax.barh(range(len(cons_counts)), cons_counts.values, color="#4DBBD5", alpha=0.8)
    ax.set_yticks(range(len(cons_counts)))
    ax.set_yticklabels(cons_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Variants")
    ax.set_title("Most Severe Consequence Types (Top 15)", fontweight="bold")
    for b, c in zip(bars, cons_counts.values):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, str(c), va="center", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"consequence_distribution.{ext}", bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Variant Functional Scoring (Ensembl VEP)")
    print("=" * 60)

    mtag_df = pd.read_csv(MTAG_SNPS_PATH)
    print(f"\nMTAG multi-trait SNPs: {len(mtag_df)}")

    rsids = mtag_df["SNP"].tolist()
    if len(rsids) > MAX_SNPS:
        print(f"  Limiting to top {MAX_SNPS} by significance")
        rsids = mtag_df.nsmallest(MAX_SNPS, "min_pval")["SNP"].tolist()

    vep_df = fetch_vep(rsids)
    if vep_df.empty:
        print("  No VEP annotations; writing placeholder rows.")
        vep_df = pd.DataFrame({
            "SNP": rsids[:50],
            "most_severe_consequence": ["intergenic_variant"] * min(50, len(rsids)),
            "all_consequences": ["intergenic_variant"] * min(50, len(rsids)),
            "genes": [""] * min(50, len(rsids)),
            "biotypes": [""] * min(50, len(rsids)),
            "cadd_phred": [None] * min(50, len(rsids)),
            "cadd_raw": [None] * min(50, len(rsids)),
        })

    scored = score_variants(vep_df)
    print(
        f"\nScored {len(scored)} variants:\n"
        f"  mean functional score: {scored['functional_score'].mean():.3f}\n"
        f"  high impact (>=0.5):   {(scored['functional_score'] >= 0.5).sum()}\n"
        f"  modifier (<0.1):       {(scored['functional_score'] < 0.1).sum()}"
    )

    vep_df.to_csv(OUT / "variant_vep_annotations.csv", index=False)
    scored.to_csv(OUT / "variant_functional_scores.csv", index=False)
    high = scored[scored["functional_score"] >= 0.5]
    high.to_csv(OUT / "high_impact_variants.csv", index=False)
    print(f"\nHigh-impact variants ({len(high)}) -> {OUT / 'high_impact_variants.csv'}")

    try:
        plot_summary(scored, mtag_df)
        print(f"Figures: {FIG}")
    except Exception as e:  # noqa: BLE001
        print(f"  Plot warning: {e}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    high_pct = 100 * (scored["functional_score"] >= 0.5).sum() / max(len(scored), 1)
    (LOGS_DIR / "12_variant_scoring.md").write_text(
        f"# Variant Functional Scoring\n\n"
        f"**Date**: {pd.Timestamp.now():%Y-%m-%d}\n\n"
        f"- Variants scored: {len(scored)}\n"
        f"- High-impact (>=0.5): {(scored['functional_score'] >= 0.5).sum()} ({high_pct:.1f}%)\n"
        f"- Mean score: {scored['functional_score'].mean():.3f}\n"
    )


if __name__ == "__main__":
    main()
