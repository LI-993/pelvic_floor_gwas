#!/usr/bin/env python3
"""Mendelian-randomization visualizations.

Builds the directed causal-effect network, IVW forest plot, IVW/WM/Egger
method comparison, exposure-outcome heatmap with significance stars, and
the MR-Egger intercept (horizontal pleiotropy) check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, PHENOTYPE_COLORS, PHENOTYPE_SHORT, RESULTS_DIR
from utils.plotting import setup_publication_style, significance_marker

KEY_PAIRS_FOR_COMPARISON: list[tuple[str, str]] = [
    ("BPH", "Incontinence"),
    ("Incontinence", "BPH"),
    ("POP", "Incontinence"),
    ("POP", "FemaleProlapse"),
    ("FemaleProlapse", "POP"),
    ("FemaleProlapse", "Incontinence"),
]


def load_mr() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "mr" / "mr_bidirectional_results.csv")


def plot_causal_network(mr_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.DiGraph()
    G.add_nodes_from(set(mr_df["exposure"]).union(mr_df["outcome"]))

    sig = mr_df[mr_df["ivw_p"] < 0.05]
    for _, row in sig.iterrows():
        G.add_edge(row["exposure"], row["outcome"], beta=row["ivw_beta"], p=row["ivw_p"])

    pos = nx.circular_layout(G)
    node_colors = [PHENOTYPE_COLORS.get(n, "#888888") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: PHENOTYPE_SHORT.get(n, n) for n in G.nodes()}, font_size=11, font_weight="bold", ax=ax)

    for u, v, d in G.edges(data=True):
        beta, p_val = d["beta"], d["p"]
        color = "#E64B35" if beta > 0 else "#4DBBD5"
        width = 1 + abs(beta) * 5
        alpha = min(1.0, 0.4 - np.log10(p_val) * 0.1)

        sx, sy = pos[u]
        ex, ey = pos[v]
        dx, dy = ex - sx, ey - sy
        shrink = 0.15
        arrow = FancyArrowPatch(
            (sx + dx * shrink, sy + dy * shrink),
            (ex - dx * shrink, ey - dy * shrink),
            connectionstyle="arc3,rad=0.1",
            arrowstyle=ArrowStyle("->", head_length=10, head_width=6),
            color=color, alpha=alpha, linewidth=width, mutation_scale=15,
        )
        ax.add_patch(arrow)
        mx, my = (sx + ex) / 2 + 0.05, (sy + ey) / 2 + 0.05
        ax.text(mx, my, f"β={beta:.2f}", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.legend(handles=[
        Line2D([0], [0], color="#E64B35", linewidth=3, label="Positive effect (β>0)"),
        Line2D([0], [0], color="#4DBBD5", linewidth=3, label="Negative effect (β<0)"),
    ], loc="upper left", fontsize=10)
    ax.set_title("Mendelian Randomization Causal Network\n(IVW p<0.05)", fontsize=14, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mr_causal_network.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: mr_causal_network.png/pdf")


def plot_forest(mr_df: pd.DataFrame, out_dir: Path) -> None:
    sig = mr_df[mr_df["ivw_p"] < 0.05].copy()
    if sig.empty:
        print("  Forest plot: no significant results")
        return

    sig["label"] = sig["exposure"] + " → " + sig["outcome"]
    sig = sig.sort_values("ivw_beta")
    sig["ci_lower"] = sig["ivw_beta"] - 1.96 * sig["ivw_se"]
    sig["ci_upper"] = sig["ivw_beta"] + 1.96 * sig["ivw_se"]

    fig, ax = plt.subplots(figsize=(12, max(6, len(sig) * 0.5)))
    y = np.arange(len(sig))
    ax.errorbar(
        sig["ivw_beta"], y,
        xerr=[sig["ivw_beta"] - sig["ci_lower"], sig["ci_upper"] - sig["ivw_beta"]],
        fmt="o", markersize=8, capsize=4, capthick=1.5, color="black", ecolor="gray", elinewidth=1.5,
    )
    for i, beta in enumerate(sig["ivw_beta"]):
        ax.scatter(beta, i, c=("#E64B35" if beta > 0 else "#4DBBD5"), s=100, zorder=5, edgecolors="white", linewidths=1)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sig["label"])
    ax.set_xlabel("Causal Effect (β) with 95% CI", fontsize=12)
    ax.set_title("Mendelian Randomization Forest Plot (IVW, p<0.05)", fontsize=14, fontweight="bold")

    for i, r in enumerate(sig.itertuples()):
        ax.text(ax.get_xlim()[1] * 1.02, i, f"p={r.ivw_p:.2e} (n={r.n_snps})", va="center", fontsize=9)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 1.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mr_forest_plot.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: mr_forest_plot.png/pdf")


def plot_method_comparison(mr_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for exp, outcome in KEY_PAIRS_FOR_COMPARISON:
        match = mr_df[(mr_df["exposure"] == exp) & (mr_df["outcome"] == outcome)]
        if len(match):
            r = match.iloc[0]
            rows.append({
                "Pair": f"{exp} → {outcome}",
                "IVW": r["ivw_beta"], "IVW_SE": r["ivw_se"],
                "WM": r["wm_beta"], "WM_SE": r["wm_se"],
                "Egger": r.get("egger_beta", np.nan), "Egger_SE": r.get("egger_se", np.nan),
            })
    if not rows:
        print("  Method comparison: no data")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["IVW"], width, yerr=df["IVW_SE"], label="IVW", color="#E64B35", alpha=0.8, capsize=3)
    ax.bar(x, df["WM"], width, yerr=df["WM_SE"], label="Weighted Median", color="#4DBBD5", alpha=0.8, capsize=3)
    ax.bar(x + width, df["Egger"], width, yerr=df["Egger_SE"], label="MR-Egger", color="#00A087", alpha=0.8, capsize=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Pair"], rotation=45, ha="right")
    ax.set_xlabel("Causal Relationship", fontsize=12)
    ax.set_ylabel("Causal Effect (β)", fontsize=12)
    ax.set_title("Comparison of MR Methods", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mr_method_comparison.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: mr_method_comparison.png/pdf")


def plot_heatmap(mr_df: pd.DataFrame, out_dir: Path) -> None:
    phenos = sorted(set(mr_df["exposure"]).union(mr_df["outcome"]))
    beta = pd.DataFrame(np.nan, index=phenos, columns=phenos)
    pval = pd.DataFrame(1.0, index=phenos, columns=phenos)
    for _, r in mr_df.iterrows():
        beta.loc[r["exposure"], r["outcome"]] = r["ivw_beta"]
        pval.loc[r["exposure"], r["outcome"]] = r["ivw_p"]
    for p in phenos:
        beta.loc[p, p] = 0

    annot = beta.copy().astype(str)
    for i in phenos:
        for j in phenos:
            b = beta.loc[i, j]
            if pd.isna(b):
                annot.loc[i, j] = ""
            elif i == j:
                annot.loc[i, j] = "-"
            else:
                annot.loc[i, j] = f"{b:.2f}{significance_marker(pval.loc[i, j])}"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        beta, mask=beta.isna(), annot=annot, fmt="", cmap="RdBu_r", center=0,
        linewidths=0.5, cbar_kws={"label": "Causal Effect (β)"},
        xticklabels=[PHENOTYPE_SHORT.get(p, p) for p in phenos],
        yticklabels=[PHENOTYPE_SHORT.get(p, p) for p in phenos], ax=ax,
    )
    ax.set_xlabel("Outcome", fontsize=12)
    ax.set_ylabel("Exposure", fontsize=12)
    ax.set_title("Bidirectional MR Causal Effects\n(*** p<0.001, ** p<0.01, * p<0.05)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mr_causal_heatmap.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: mr_causal_heatmap.png/pdf")


def plot_pleiotropy(mr_df: pd.DataFrame, out_dir: Path) -> None:
    sig = mr_df[mr_df["ivw_p"] < 0.1].copy()
    if sig.empty or "egger_intercept" not in sig.columns:
        print("  Pleiotropy: no data")
        return

    sig = sig.dropna(subset=["egger_intercept", "egger_intercept_p"])
    sig["label"] = sig["exposure"] + " → " + sig["outcome"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#E64B35" if p < 0.05 else "#00A087" for p in sig["egger_intercept_p"]]
    ax.barh(np.arange(len(sig)), sig["egger_intercept"], color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(sig)))
    ax.set_yticklabels(sig["label"])
    ax.set_xlabel("MR-Egger Intercept", fontsize=12)
    ax.set_title("Horizontal Pleiotropy Check\nRed = significant pleiotropy (p<0.05)", fontsize=14, fontweight="bold")

    for i, r in enumerate(sig.itertuples()):
        ax.text(ax.get_xlim()[1] * 0.95, i, f"p={r.egger_intercept_p:.3f}", va="center", ha="right", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mr_pleiotropy_check.{ext}", bbox_inches="tight")
    plt.close()
    print("  Saved: mr_pleiotropy_check.png/pdf")


def main() -> None:
    setup_publication_style()
    out_dir = FIGURES_DIR / "mr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MR Visualizations")
    print("=" * 60)

    mr_df = load_mr()
    print(f"\nLoaded {len(mr_df)} MR tests; significant (p<0.05): {(mr_df['ivw_p'] < 0.05).sum()}")

    plot_causal_network(mr_df, out_dir)
    plot_forest(mr_df, out_dir)
    plot_method_comparison(mr_df, out_dir)
    plot_heatmap(mr_df, out_dir)
    plot_pleiotropy(mr_df, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
