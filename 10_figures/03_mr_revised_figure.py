#!/usr/bin/env python3
"""Mendelian Randomization manuscript figure (revised palette).

Two-panel layout:
  A: Forest plot of significant IVW results with WM/Egger overlays. Excludes
     POP↔FemaleProlapse (positive control).
  B: Clean directed network of significant causal effects, with edge color =
     direction of effect and edge width = -log10(p).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, RESULTS_DIR

OUT = FIGURES_DIR / "revised"

NODE_COLORS = {
    "POP":        "#7B68AE",
    "BPH":        "#E8A838",
    "Bladder":    "#2D8E4E",
    "Constip.":   "#C47A6E",
    "F.Prolapse": "#5B9BD5",
    "Incontin.":  "#8B6BAE",
}

NAME_MAP = {
    "POP": "POP", "BPH": "BPH",
    "FemaleProlapse": "F.Prolapse",
    "Incontinence": "Incontin.",
    "Constipation": "Constip.",
    "Bladder": "Bladder",
}

NODE_POS = {
    "POP":        (0.15, 0.80),
    "F.Prolapse": (0.15, 0.20),
    "BPH":        (0.85, 0.80),
    "Incontin.":  (0.85, 0.20),
    "Constip.":   (0.50, -0.10),
}

METHODS = [
    ("ivw_beta", "ivw_se", "IVW", "#2D5F8A", "o", 8),
    ("wm_beta", "wm_se", "Weighted Median", "#E8A838", "s", 7),
    ("egger_beta", "egger_se", "MR-Egger", "#C47A6E", "D", 7),
]


def load_mr() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "mr" / "mr_bidirectional_results.csv")
    mask = ~(
        ((df["exposure"] == "POP") & (df["outcome"] == "FemaleProlapse"))
        | ((df["exposure"] == "FemaleProlapse") & (df["outcome"] == "POP"))
    )
    return df[mask].copy()


def panel_forest(ax, df_sig: pd.DataFrame) -> None:
    y_pos = np.arange(len(df_sig))
    bar_height = 0.25

    for j, (beta_col, se_col, method_name, color, marker, ms) in enumerate(METHODS):
        offset = (j - 1) * bar_height
        ax.errorbar(
            df_sig[beta_col].values, y_pos + offset,
            xerr=1.96 * df_sig[se_col].values,
            fmt=marker, color=color, markersize=ms, capsize=3,
            linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
            label=method_name, zorder=3,
        )

    for i, row in df_sig.iterrows():
        idx = df_sig.index.get_loc(i)
        p = row["ivw_p"]
        n = int(row["n_snps"])
        sig_text = f"P={p:.1e}  (n={n})" if p < 1e-3 else f"P={p:.3f}  (n={n})"
        ax.text(max(row["ivw_beta"] + 1.96 * row["ivw_se"] + 0.01, 0.42), idx,
                sig_text, va="center", fontsize=8, color="#333333")

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sig["label"].values, fontsize=10)
    ax.set_xlabel("Causal Effect (β) with 95% CI", fontsize=11)
    ax.set_title("A  Mendelian Randomization Forest Plot", fontsize=13, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.15, 0.65)


def panel_network(ax, df_sig: pd.DataFrame) -> None:
    for node, (x, y) in NODE_POS.items():
        ax.add_patch(plt.Circle((x, y), 0.09, color=NODE_COLORS[node], alpha=0.85, zorder=5))
        ax.text(x, y, node, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=6,
                path_effects=[pe.withStroke(linewidth=1, foreground="#333333")])

    edges = []
    for _, row in df_sig.iterrows():
        edges.append({
            "from": NAME_MAP[row["exposure"]],
            "to": NAME_MAP[row["outcome"]],
            "beta": row["ivw_beta"],
            "p": row["ivw_p"],
        })

    for edge in edges:
        if edge["from"] not in NODE_POS or edge["to"] not in NODE_POS:
            continue
        x1, y1 = NODE_POS[edge["from"]]
        x2, y2 = NODE_POS[edge["to"]]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / dist, dy / dist
        sx, sy = x1 + ux * 0.10, y1 + uy * 0.10
        ex, ey = x2 - ux * 0.10, y2 - uy * 0.10
        reverse = any(e["from"] == edge["to"] and e["to"] == edge["from"] for e in edges)

        if edge["p"] < 1e-3:
            lw, alpha = 2.5, 0.9
        elif edge["p"] < 1e-2:
            lw, alpha = 2.0, 0.8
        else:
            lw, alpha = 1.5, 0.65

        edge_color = "#D94444" if edge["beta"] > 0 else "#4488CC"
        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            connectionstyle="arc3,rad=0.2" if reverse else "arc3,rad=0",
            arrowstyle="->", mutation_scale=15,
            color=edge_color, linewidth=lw, alpha=alpha, zorder=3,
        )
        ax.add_patch(arrow)

        mid_x, mid_y = (sx + ex) / 2, (sy + ey) / 2
        offset_x, offset_y = -uy * 0.05, ux * 0.05
        if reverse:
            offset_x *= 1.8
            offset_y *= 1.8
        sig_star = "***" if edge["p"] < 1e-3 else ("**" if edge["p"] < 1e-2 else "*")
        ax.text(mid_x + offset_x, mid_y + offset_y, f"β={edge['beta']:.2f}{sig_star}",
                ha="center", va="center", fontsize=7.5, color=edge_color, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    pos_patch = mpatches.Patch(color="#D94444", label="Positive effect (risk)")
    neg_patch = mpatches.Patch(color="#4488CC", label="Negative effect (protective)")
    ax.text(0.02, -0.05, "*** P<0.001  ** P<0.01  * P<0.05", fontsize=8, color="#555555", transform=ax.transAxes)
    ax.legend(handles=[pos_patch, neg_patch], loc="upper left", fontsize=8, framealpha=0.9, bbox_to_anchor=(-0.02, 1.02))
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.35, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("B  Causal Network (IVW, P<0.05)", fontsize=13, fontweight="bold", loc="left")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== MR Results Summary ===")
    header = (
        f"{'Exposure':<18} {'Outcome':<18} {'N_SNPs':>6} {'IVW_β':>8} {'IVW_P':>12} "
        f"{'WM_β':>8} {'WM_P':>12} {'Egger_β':>8} {'Egger_P':>12} {'Egger_int_P':>12}"
    )
    print(header)
    print("-" * 130)
    for _, r in df.sort_values("ivw_p").iterrows():
        sig = "***" if r["ivw_p"] < 1e-3 else ("**" if r["ivw_p"] < 1e-2 else ("*" if r["ivw_p"] < 5e-2 else ""))
        print(
            f"{r['exposure']:<18} {r['outcome']:<18} {int(r['n_snps']):>6} "
            f"{r['ivw_beta']:>8.3f} {r['ivw_p']:>12.2e}{sig:<3} "
            f"{r['wm_beta']:>8.3f} {r['wm_p']:>12.2e} "
            f"{r.get('egger_beta', float('nan')):>8.3f} {r.get('egger_p', float('nan')):>12.2e} "
            f"{r.get('egger_intercept_p', float('nan')):>12.3f}"
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_mr()
    df_sig = df[df["ivw_p"] < 0.05].copy()
    df_sig["label"] = df_sig.apply(lambda r: f"{NAME_MAP[r['exposure']]} → {NAME_MAP[r['outcome']]}", axis=1)
    df_sig = df_sig.sort_values("ivw_beta").reset_index(drop=True)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    panel_forest(fig.add_subplot(gs[0]), df_sig)
    panel_network(fig.add_subplot(gs[1]), df_sig)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig_MR_revised.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {OUT / 'Fig_MR_revised.png'} and .pdf")

    print_summary(df)


if __name__ == "__main__":
    main()
