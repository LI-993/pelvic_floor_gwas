#!/usr/bin/env python3
"""ML-based gene prioritization with external (OMIM/HPO) validation labels.

Builds a gene-level feature matrix from MAGMA, the PPI network, and
drug-target evidence, then trains Random Forest + Gradient Boosting models
against curated OMIM/HPO disease genes (avoiding the circularity of using
GWAS p-values as both feature and label). Outputs a ranked gene table and
top-N enrichment statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPES, RESULTS_DIR
from utils.gene_mapping import get_symbol, load_ncbi_gene_mapping
from utils.plotting import setup_publication_style

OUT_RESULTS = RESULTS_DIR / "gene_prioritization_ml"
OUT_FIGURES = FIGURES_DIR / "gene_prioritization"
MAGMA_DIR = RESULTS_DIR / "magma"
PPI_DIR = RESULTS_DIR / "ppi_network"
DRUG_DIR = RESULTS_DIR / "drug_repurposing"


# Curated OMIM disease genes for pelvic floor disorders (manuscript supplement).
OMIM_PELVIC_GENES: set[str] = {
    "COL3A1", "COL1A1", "COL1A2",          # Ehlers-Danlos
    "FBN1", "FBN2",                          # Marfan
    "FLNA",                                  # filamin A
    "LOXL1",                                 # elastin cross-linking
    "MMP2", "MMP9",                          # matrix metalloproteinases
    "LAMC1", "FBLN5",                        # ECM
    "ESR1", "ESR2", "PGR",                   # hormone receptors
    "CHRM2", "CHRM3", "ADRB3",               # bladder receptors
    "SRD5A1", "SRD5A2", "AR", "CYP17A1",
    "HSD3B1", "HSD3B2",                      # androgen biosynthesis
    "RET", "GDNF", "NRTN", "EDN3", "EDNRB",  # Hirschsprung-related
    "SOX10", "PHOX2B",
    "ELN", "ACTA2", "MYH11", "ACTG2",        # smooth muscle
    "ADAMTS2", "ADAMTS13", "BMP1", "SPARC", "VCAN",
}

# HPO terms HP:0000020/0000139/0011025/0002019 — manually curated subset.
HPO_PELVIC_GENES: set[str] = {
    "ATP2B4", "CFTR", "CLCN2", "DRD1", "DRD2",
    "GNB3", "GNAS", "HTR4", "KCNQ1", "NOS1", "NOS3",
    "NPY", "OPRM1", "SCN5A", "SLC12A2", "SLC26A3", "SLC9A3",
    "TRPV4", "VIP",
    "CHRNA3", "CHRNB4", "P2RX1", "P2RX2", "P2RX3",
    "TACR1", "TACR2", "TRPM8", "TRPV1",
    "CYP1A1", "CYP1B1", "CYP19A1", "COMT", "SULT1A1",
    "HSD17B1", "HSD17B2", "SHBG",
}

EXTERNAL_POSITIVE_GENES: set[str] = OMIM_PELVIC_GENES | HPO_PELVIC_GENES

# Entrez fallbacks for genes whose symbols may not match in the MAGMA gene-loc.
EXTRA_ENTREZ: dict[str, str] = {
    "1277": "COL1A1", "1278": "COL1A2", "1281": "COL3A1",
    "2006": "ELN", "4015": "LOXL1",
    "2099": "ESR1", "2100": "ESR2", "5241": "PGR",
    "6715": "SRD5A1", "6716": "SRD5A2", "367": "AR", "354": "KLK3",
    "5979": "RET", "2668": "GDNF", "1906": "EDN3",
    "4313": "MMP2", "4318": "MMP9", "2192": "FBLN5",
    "59": "ACTA2", "4629": "MYH11",
    "54361": "WNT4", "7490": "WT1",
    "53335": "BCL11A", "3122": "HLA-DRA", "185": "AGTR1",
}
EXTERNAL_POSITIVE_ENTREZ: set[str] = set(EXTRA_ENTREZ)


def load_magma_features() -> pd.DataFrame:
    """Aggregate per-phenotype MAGMA outputs into one row per gene."""
    print("  Loading MAGMA features...")
    entrez_to_symbol, _, _ = load_ncbi_gene_mapping()

    frames: list[pd.DataFrame] = []
    for pheno in PHENOTYPES:
        path = MAGMA_DIR / f"{pheno}_genes.genes.out.txt"
        if path.exists():
            df = pd.read_csv(path, sep=r"\s+", comment="#")
            df["Phenotype"] = pheno
            df["Symbol"] = df["GENE"].astype(str).map(lambda x: get_symbol(x, entrez_to_symbol))
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    sig = combined[combined["P"] < 0.05].copy()
    sig = sig.rename(columns={"GENE": "GeneID", "ZSTAT": "Z", "NSNPS": "nSNPs"})

    rows: dict[str, dict] = {}
    for _, r in sig.iterrows():
        gene = r["Symbol"]
        rec = rows.setdefault(gene, {
            "gene_id": r["GeneID"], "min_p": r["P"], "max_z": r["Z"], "n_snps": r["nSNPs"],
            "phenotypes": set(), "z_values": [], "p_values": [],
        })
        rec["phenotypes"].add(r["Phenotype"])
        rec["z_values"].append(r["Z"])
        rec["p_values"].append(r["P"])
        rec["min_p"] = min(rec["min_p"], r["P"])
        rec["max_z"] = max(rec["max_z"], r["Z"])

    out = pd.DataFrame([
        {
            "Gene": gene,
            "gene_id": rec["gene_id"],
            "min_p": rec["min_p"],
            "max_z": rec["max_z"],
            "mean_z": np.mean(rec["z_values"]),
            "std_z": np.std(rec["z_values"]) if len(rec["z_values"]) > 1 else 0.0,
            "n_snps": rec["n_snps"],
            "n_phenotypes": len(rec["phenotypes"]),
        }
        for gene, rec in rows.items()
    ])
    out["neglog10p"] = -np.log10(out["min_p"].clip(lower=1e-300))
    print(f"  -> features for {len(out)} genes")
    return out


def load_network_features() -> pd.DataFrame:
    print("  Loading network features...")
    metrics_file = PPI_DIR / "network_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    print("    (network metrics not found)")
    return pd.DataFrame(columns=["Gene", "Degree", "Betweenness", "Closeness"])


def load_drug_features() -> pd.DataFrame:
    print("  Loading drug-target features...")
    drug_file = DRUG_DIR / "prioritized_candidates.csv"
    if not drug_file.exists():
        print("    (drug data not found)")
        return pd.DataFrame(columns=["Gene", "n_drug_interactions", "max_drug_priority"])

    drug_df = pd.read_csv(drug_file)
    out = (
        drug_df.groupby("gene_symbol").agg(n_drug_interactions=("drug", "count"), max_drug_priority=("priority_score", "max"))
        .reset_index().rename(columns={"gene_symbol": "Gene"})
    )
    return out


def build_feature_matrix(magma: pd.DataFrame, net: pd.DataFrame, drug: pd.DataFrame) -> pd.DataFrame:
    df = magma.copy()
    if len(net):
        df = df.merge(net, on="Gene", how="left")
    if len(drug):
        df = df.merge(drug, on="Gene", how="left")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    print(f"  feature matrix: {df.shape[0]} genes x {df.shape[1]} columns")
    return df


def assign_external_labels(df: pd.DataFrame) -> pd.Series:
    by_symbol = df["Gene"].isin(EXTERNAL_POSITIVE_GENES)
    by_entrez = df["gene_id"].astype(str).isin(EXTERNAL_POSITIVE_ENTREZ) if "gene_id" in df.columns else pd.Series(False, index=df.index)
    df["is_known_disease_gene"] = (by_symbol | by_entrez).astype(int)

    n_pos = int(df["is_known_disease_gene"].sum())
    print(f"  Known disease genes (OMIM/HPO): {n_pos} / {len(df)}")

    if n_pos < 5:
        print("  Too few external labels — falling back to multi-phenotype pseudo-labels.")
        df["is_known_disease_gene"] = (df["n_phenotypes"] > 1).astype(int)
    return df["is_known_disease_gene"]


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict | None:
    n_pos = int(y.sum())
    if n_pos < 5:
        print(f"  Only {n_pos} positives; skipping supervised training.")
        return None

    X_scaled = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=min(5, n_pos), shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="roc_auc")
    rf.fit(X_scaled, y)
    print(f"  RF CV AUC: {rf_scores.mean():.3f} (+/- {rf_scores.std():.3f})")

    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, min_samples_leaf=5, random_state=42)
    gb_scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring="roc_auc")
    gb.fit(X_scaled, y)
    print(f"  GB CV AUC: {gb_scores.mean():.3f} (+/- {gb_scores.std():.3f})")

    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    gb_proba = gb.predict_proba(X_scaled)[:, 1]
    ensemble = (rf_proba + gb_proba) / 2

    precision, recall, _ = precision_recall_curve(y, ensemble)
    pr_auc = auc(recall, precision)
    print(f"  Ensemble PR-AUC: {pr_auc:.3f}")

    return {
        "RandomForest": {"cv_auc": rf_scores.mean(), "cv_auc_std": rf_scores.std(),
                         "predictions": rf_proba,
                         "feature_importance": dict(zip(feature_names, rf.feature_importances_))},
        "GradientBoosting": {"cv_auc": gb_scores.mean(), "cv_auc_std": gb_scores.std(),
                             "predictions": gb_proba,
                             "feature_importance": dict(zip(feature_names, gb.feature_importances_))},
        "Ensemble": {"predictions": ensemble, "pr_auc": pr_auc},
    }


def make_ranking(features: pd.DataFrame, model_results: dict | None) -> pd.DataFrame:
    keep = ["Gene", "gene_id", "n_phenotypes", "min_p", "max_z", "mean_z"]
    rank = features[keep].copy()
    rank["GWAS_score"] = -np.log10(rank["min_p"].clip(lower=1e-300))
    gmin, gmax = rank["GWAS_score"].min(), rank["GWAS_score"].max()
    rank["GWAS_score_norm"] = (rank["GWAS_score"] - gmin) / (gmax - gmin) if gmax > gmin else 0.0
    rank["is_known_disease_gene"] = rank["Gene"].isin(EXTERNAL_POSITIVE_GENES).astype(int)

    if model_results:
        rank["RF_score"] = model_results["RandomForest"]["predictions"]
        rank["GB_score"] = model_results["GradientBoosting"]["predictions"]
        rank["Ensemble_score"] = model_results["Ensemble"]["predictions"]
        rank["Final_score"] = (
            0.35 * rank["Ensemble_score"]
            + 0.35 * rank["GWAS_score_norm"]
            + 0.20 * rank["n_phenotypes"] / max(rank["n_phenotypes"].max(), 1)
            + 0.10 * (rank["mean_z"] - rank["mean_z"].min())
                / max(rank["mean_z"].max() - rank["mean_z"].min(), 1e-9)
        )
    else:
        rank[["RF_score", "GB_score", "Ensemble_score"]] = np.nan
        rank["Final_score"] = (
            0.5 * rank["GWAS_score_norm"]
            + 0.3 * rank["n_phenotypes"] / max(rank["n_phenotypes"].max(), 1)
            + 0.2 * (rank["mean_z"] - rank["mean_z"].min())
                / max(rank["mean_z"].max() - rank["mean_z"].min(), 1e-9)
        )

    rank = rank.sort_values("Final_score", ascending=False)
    rank["Rank"] = range(1, len(rank) + 1)
    return rank


def validate_ranking(ranking: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = len(ranking)
    total_known = ranking["is_known_disease_gene"].sum()
    for top_n in (10, 20, 50, 100):
        if top_n > total:
            continue
        top = set(ranking.head(top_n)["Gene"])
        n_known = len(top & EXTERNAL_POSITIVE_GENES)
        expected = top_n * total_known / total if total else 0
        out[f"top_{top_n}"] = {
            "n_known": n_known,
            "expected": expected,
            "enrichment": n_known / expected if expected else 0,
        }
        print(f"  Top {top_n}: {n_known} known (expected {expected:.1f}, {out[f'top_{top_n}']['enrichment']:.2f}x)")
    return out


def plot_summary(features: pd.DataFrame, model_results: dict | None, ranking: pd.DataFrame, validation: dict) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    if model_results:
        importance = model_results["RandomForest"]["feature_importance"]
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:12]
        labels, values = zip(*sorted_imp)
        ax1.barh(np.arange(len(labels)), values, color="#3C5488", alpha=0.8)
        ax1.set_yticks(np.arange(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.invert_yaxis()
        ax1.set_xlabel("Importance")
        ax1.set_title("Random Forest Feature Importance", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No supervised model", ha="center", va="center")
        ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    keys = [k.replace("top_", "") for k in validation]
    vals = [v["enrichment"] for v in validation.values()]
    bars = ax2.bar(keys, vals, color="#E64B35", alpha=0.8)
    ax2.axhline(1, color="gray", linestyle="--", label="Expected")
    ax2.set_xlabel("Top N")
    ax2.set_ylabel("Enrichment")
    ax2.set_title("Enrichment of Known Disease Genes", fontweight="bold")
    ax2.legend()
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{v:.2f}x", ha="center", fontsize=9)

    ax3 = fig.add_subplot(2, 2, 3)
    top_genes = ranking.head(25)
    colors = ["#E64B35" if k else "#4DBBD5" for k in top_genes["is_known_disease_gene"]]
    ax3.barh(np.arange(len(top_genes)), top_genes["Final_score"], color=colors, alpha=0.8)
    ax3.set_yticks(np.arange(len(top_genes)))
    ax3.set_yticklabels(top_genes["Gene"])
    ax3.invert_yaxis()
    ax3.set_xlabel("Prioritization Score")
    ax3.set_title("Top 25 Prioritized Genes\n(Red = OMIM/HPO known)", fontweight="bold")

    ax4 = fig.add_subplot(2, 2, 4)
    if model_results:
        sizes = [80 if k else 30 for k in ranking["is_known_disease_gene"]]
        scatter_colors = ["#E64B35" if k else "#CCCCCC" for k in ranking["is_known_disease_gene"]]
        ax4.scatter(ranking["GWAS_score_norm"], ranking["Ensemble_score"], c=scatter_colors, s=sizes, alpha=0.6)
        for _, row in ranking[ranking["is_known_disease_gene"] == 1].head(5).iterrows():
            ax4.annotate(row["Gene"], (row["GWAS_score_norm"], row["Ensemble_score"]), fontsize=8, fontweight="bold")
        ax4.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax4.set_xlabel("GWAS Score (normalized)")
        ax4.set_ylabel("ML Ensemble Score")
        ax4.set_title("GWAS vs ML Scores", fontweight="bold")
    else:
        ax4.axis("off")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"ml_prioritization.{ext}", bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_publication_style()
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ML Gene Prioritization (external OMIM/HPO labels)")
    print("=" * 60)

    print("\n[1] Loading features...")
    magma_df = load_magma_features()
    if magma_df.empty:
        print("Error: no MAGMA results found.")
        return

    network_df = load_network_features()
    drug_df = load_drug_features()

    print("\n[2] Feature matrix...")
    features = build_feature_matrix(magma_df, network_df, drug_df)
    features.to_csv(OUT_RESULTS / "feature_matrix.csv", index=False)

    print("\n[3] External labels...")
    labels = assign_external_labels(features)

    print("\n[4] Training...")
    feature_cols = [
        c for c in features.columns
        if c not in {"Gene", "gene_id", "phenotypes", "min_p", "is_known_disease_gene"}
    ]
    model_results = train_models(features[feature_cols].values, labels.values, feature_cols)

    print("\n[5] Ranking...")
    ranking = make_ranking(features, model_results)
    ranking.to_csv(OUT_RESULTS / "final_gene_ranking.csv", index=False)

    if model_results:
        perf = pd.DataFrame([
            {"Model": name, "CV_AUC": rec.get("cv_auc"), "CV_AUC_std": rec.get("cv_auc_std")}
            for name, rec in model_results.items() if "cv_auc" in rec
        ])
        perf.to_csv(OUT_RESULTS / "model_performance.csv", index=False)

    print("\n[6] External validation...")
    validation = validate_ranking(ranking)
    pd.DataFrame([{"TopN": k.replace("top_", ""), **v} for k, v in validation.items()]).to_csv(
        OUT_RESULTS / "validation_results.csv", index=False
    )

    print("\nTop 15:")
    for _, row in ranking.head(15).iterrows():
        marker = "*" if row["is_known_disease_gene"] else " "
        print(f"  {row['Rank']:2d}. {marker}{row['Gene']:12s} (score={row['Final_score']:.3f}, n_pheno={row['n_phenotypes']})")
    print("  (* = OMIM/HPO known)")

    print("\n[7] Plot...")
    plot_summary(features, model_results, ranking, validation)
    print(f"\nResults: {OUT_RESULTS}\nFigures: {OUT_FIGURES}")


if __name__ == "__main__":
    main()
