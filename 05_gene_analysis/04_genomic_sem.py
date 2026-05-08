#!/usr/bin/env python3
"""Latent factor analysis on the LDSC genetic-correlation matrix.

A pure-Python alternative to the GenomicSEM R package: runs eigenvalue
decomposition + scree plot, EFA with varimax rotation, hierarchical clustering
on (1 - |rg|) distances, and writes loadings + cluster assignments. Results
inform the dimensionality argument used in the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA, FactorAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPE_COLORS, PHENOTYPES, RESULTS_DIR
from utils.plotting import setup_publication_style

OUT_RESULTS = RESULTS_DIR / "genomic_sem"
OUT_FIGURES = FIGURES_DIR / "genomic_sem"
LOADING_THRESHOLD = 0.4


def load_rg_matrix() -> pd.DataFrame:
    rg_df = pd.read_csv(RESULTS_DIR / "ldsc" / "genetic_correlation_summary.tsv", sep="\t")
    n = len(PHENOTYPES)
    matrix = pd.DataFrame(np.eye(n), index=PHENOTYPES, columns=PHENOTYPES)
    for _, row in rg_df.iterrows():
        a, b = row["phenotype1"], row["phenotype2"]
        matrix.loc[a, b] = matrix.loc[b, a] = row["rg"]
    return matrix


def determine_n_factors(rg_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    eigenvalues = np.real(np.linalg.eigvals(rg_matrix.values))
    eigenvalues = np.sort(eigenvalues)[::-1]
    var_explained = eigenvalues / eigenvalues.sum()

    kaiser = int(np.sum(eigenvalues > 1))
    eighty_pct = int(np.argmax(np.cumsum(var_explained) >= 0.8) + 1)
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Kaiser (λ > 1): {kaiser} factors | 80% variance: {eighty_pct} factors")
    return eigenvalues, var_explained, max(kaiser, 2)


def run_factor_analysis(rg_matrix: pd.DataFrame, n_factors: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=42)
    fa.fit(rg_matrix.values)

    loadings = pd.DataFrame(
        fa.components_.T,
        index=rg_matrix.index,
        columns=[f"Factor{i + 1}" for i in range(n_factors)],
    )
    communalities = pd.Series((loadings.values ** 2).sum(axis=1), index=rg_matrix.index, name="Communality")
    factor_variance = pd.Series((loadings.values ** 2).sum(axis=0), index=loadings.columns, name="Variance")
    return loadings, communalities, factor_variance


def hierarchical_cluster(rg_matrix: pd.DataFrame, n_clusters: int = 2) -> tuple[np.ndarray, pd.Series]:
    distance = 1 - np.abs(rg_matrix.values)
    link = linkage(distance, method="ward")
    labels = pd.Series(fcluster(link, n_clusters, criterion="maxclust"), index=rg_matrix.index, name="Cluster")
    return link, labels


def name_factor(high_loading_phenos: list[str]) -> str:
    if "POP" in high_loading_phenos and "FemaleProlapse" in high_loading_phenos:
        return "Female Pelvic Floor Factor"
    if "BPH" in high_loading_phenos:
        return "Prostate/Urinary Factor"
    if "Constipation" in high_loading_phenos:
        return "Bowel Function Factor"
    return f"Factor ({', '.join(high_loading_phenos[:2])})"


def interpret_factors(loadings: pd.DataFrame, threshold: float = LOADING_THRESHOLD) -> dict:
    interpretations: dict[str, dict] = {}
    for factor in loadings.columns:
        high = loadings[factor][loadings[factor].abs() > threshold].sort_values(ascending=False)
        interpretations[factor] = {
            "name": name_factor(high.index.tolist()),
            "loadings": high.to_dict(),
        }
        print(f"  {factor}: {interpretations[factor]['name']}")
        for pheno, load in high.items():
            print(f"    - {pheno}: {load:.3f}")
    return interpretations


def plot_scree(eigenvalues: np.ndarray, var_explained: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(1, len(eigenvalues) + 1)

    axes[0].plot(x, eigenvalues, "bo-", linewidth=2, markersize=10)
    axes[0].axhline(1, color="red", linestyle="--", label="Kaiser criterion (λ=1)")
    axes[0].set_xlabel("Factor", fontsize=12)
    axes[0].set_ylabel("Eigenvalue", fontsize=12)
    axes[0].set_title("Scree Plot", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].legend()

    cumvar = np.cumsum(var_explained)
    axes[1].bar(x, var_explained * 100, alpha=0.7, color="#4DBBD5", label="Individual")
    axes[1].plot(x, cumvar * 100, "ro-", linewidth=2, label="Cumulative")
    axes[1].axhline(80, color="gray", linestyle="--", label="80% threshold")
    axes[1].set_xlabel("Factor", fontsize=12)
    axes[1].set_ylabel("Variance Explained (%)", fontsize=12)
    axes[1].set_title("Variance Explained", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].legend()

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"scree_plot.{ext}", bbox_inches="tight")
    plt.close()


def plot_loadings(loadings: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Factor Loadings Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Factors")
    ax.set_ylabel("Phenotypes")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"factor_loadings_heatmap.{ext}", bbox_inches="tight")
    plt.close()

    n_factors = len(loadings.columns)
    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 6))
    if n_factors == 1:
        axes = [axes]
    for ax, factor in zip(axes, loadings.columns):
        data = loadings[factor].sort_values()
        colors = ["#E64B35" if v > 0 else "#4DBBD5" for v in data.values]
        ax.barh(data.index, data.values, color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.axvline(LOADING_THRESHOLD, color="red", linestyle="--", alpha=0.5)
        ax.axvline(-LOADING_THRESHOLD, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Loading")
        ax.set_title(factor, fontweight="bold")
        ax.set_xlim(-1, 1)

    plt.suptitle("Factor Loadings by Phenotype", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"factor_loadings_bar.{ext}", bbox_inches="tight")
    plt.close()


def plot_dendrogram(link: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(link, labels=labels, leaf_font_size=12, ax=ax)
    ax.set_title("Hierarchical Clustering of Phenotypes\n(Distance = 1 - |rg|)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"phenotype_dendrogram.{ext}", bbox_inches="tight")
    plt.close()


def plot_factor_diagram(loadings: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    factor_y, pheno_y = 0.8, 0.2
    factor_x = np.linspace(0.2, 0.8, len(loadings.columns))
    pheno_x = np.linspace(0.1, 0.9, len(loadings.index))

    for i, factor in enumerate(loadings.columns):
        ax.add_patch(plt.Circle((factor_x[i], factor_y), 0.08, color="#3C5488", alpha=0.8))
        ax.text(factor_x[i], factor_y, factor, ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    for i, pheno in enumerate(loadings.index):
        color = PHENOTYPE_COLORS.get(pheno, "#888888")
        ax.add_patch(plt.Rectangle((pheno_x[i] - 0.05, pheno_y - 0.05), 0.1, 0.1, color=color, alpha=0.8))
        ax.text(pheno_x[i], pheno_y - 0.12, pheno, ha="center", va="top", fontsize=9, rotation=45)

    for i, factor in enumerate(loadings.columns):
        for j, pheno in enumerate(loadings.index):
            load = loadings.loc[pheno, factor]
            if abs(load) > 0.3:
                color = "#E64B35" if load > 0 else "#4DBBD5"
                ax.plot(
                    [factor_x[i], pheno_x[j]],
                    [factor_y - 0.08, pheno_y + 0.05],
                    color=color, linewidth=abs(load) * 3, alpha=0.6,
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Factor Structure Model\n(Line width ∝ |loading|, red = positive, blue = negative)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIGURES / f"factor_structure.{ext}", bbox_inches="tight")
    plt.close()


def save_results(
    eigenvalues: np.ndarray,
    var_explained: np.ndarray,
    loadings: pd.DataFrame,
    communalities: pd.Series,
    factor_variance: pd.Series,
    cluster_labels: pd.Series,
    n_factors: int,
) -> None:
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    loadings.to_csv(OUT_RESULTS / "factor_loadings.csv")
    communalities.to_frame().to_csv(OUT_RESULTS / "communalities.csv")
    cluster_labels.to_frame().to_csv(OUT_RESULTS / "cluster_assignments.csv")

    eig_df = pd.DataFrame({
        "Factor": range(1, len(eigenvalues) + 1),
        "Eigenvalue": eigenvalues,
        "Variance_Explained": var_explained,
        "Cumulative_Variance": np.cumsum(var_explained),
    })
    eig_df.to_csv(OUT_RESULTS / "eigenvalues.csv", index=False)

    summary = OUT_RESULTS / "model_summary.txt"
    summary.write_text(
        f"Number of factors: {n_factors}\n"
        f"Total variance explained: {var_explained[:n_factors].sum() * 100:.1f}%\n\n"
        + "\n".join(f"  {f}: {v:.3f}" for f, v in factor_variance[:n_factors].items())
    )


def main() -> None:
    setup_publication_style()
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Genomic SEM / Factor Analysis")
    print("=" * 60)

    rg = load_rg_matrix()
    print(f"\nMatrix: {rg.shape}, phenotypes: {list(rg.index)}")

    print("\n[1] Eigenvalue / variance analysis...")
    eigenvalues, var_explained, n_factors = determine_n_factors(rg)

    print(f"\n[2] Factor analysis ({n_factors} factors, varimax rotation)...")
    loadings, communalities, factor_variance = run_factor_analysis(rg, n_factors)

    print("\n[3] Hierarchical clustering...")
    link, cluster_labels = hierarchical_cluster(rg)

    print("\n[4] Factor interpretation:")
    interpret_factors(loadings)

    print("\n[5] Saving results...")
    save_results(eigenvalues, var_explained, loadings, communalities, factor_variance, cluster_labels, n_factors)

    print("\n[6] Plots...")
    plot_scree(eigenvalues, var_explained)
    plot_loadings(loadings)
    plot_dendrogram(link, list(rg.index))
    plot_factor_diagram(loadings)

    # PCA reported only as a sanity check.
    pca = PCA().fit(rg.values)
    print(f"\nPCA explained-variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}")
    print(f"\nResults: {OUT_RESULTS}\nFigures: {OUT_FIGURES}")


if __name__ == "__main__":
    main()
