#!/usr/bin/env python3
"""Manuscript Fig 1 (study-design overview) and graphical abstract.

Pure matplotlib layout — no external data needed; the boxes summarize the
project pipeline. Outputs go to figures/revised/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR

OUT = FIGURES_DIR / "revised"

# Color palette.
C: dict[str, str] = {
    "bg": "#FAFAFA",
    "box_data": "#4E79A7",
    "box_method": "#F28E2B",
    "box_result": "#59A14F",
    "box_valid": "#E15759",
    "box_app": "#B07AA1",
    "text": "#333333",
    "arrow": "#888888",
    "highlight": "#EDC948",
    "light_bg": "#F0F4F8",
}


def rounded_box(
    ax,
    xy,
    width,
    height,
    text: str,
    color: str,
    *,
    fontsize: int = 9,
    text_color: str = "white",
    alpha: float = 0.9,
    bold: bool = False,
    subtext: str | None = None,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x + width / 2, y + height * 0.62, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=3)
        ax.text(x + width / 2, y + height * 0.3, subtext, ha="center", va="center",
                fontsize=fontsize - 2, color=text_color, fontstyle="italic", zorder=3, alpha=0.9)
    else:
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=3)


def arrow(ax, start, end, color: str = "#888888") -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, connectionstyle="arc3,rad=0"),
        zorder=1,
    )


def make_fig1() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7, 8.7, "Study Design Overview", ha="center", va="center", fontsize=16, fontweight="bold", color=C["text"])

    ax.text(0.3, 8.15, "DATA SOURCES", fontsize=10, fontweight="bold", color=C["box_data"], va="center")
    rounded_box(ax, (0.3, 7.2), 2.8, 0.8, "FinnGen R12", C["box_data"], fontsize=10, bold=True,
                subtext="BPH · Bladder · Constipation\nFemaleProlapse")
    rounded_box(ax, (3.5, 7.2), 2.8, 0.8, "GWAS Catalog", C["box_data"], fontsize=10, bold=True,
                subtext="POP (GCST90102470)\nIncontinence")
    rounded_box(ax, (6.7, 7.2), 2.8, 0.8, "Pan-UKBB", C["box_data"], fontsize=10, bold=True,
                subtext="EUR · AFR · CSA\n(Cross-ancestry)")
    rounded_box(ax, (9.9, 7.2), 2.8, 0.8, "Reference Data", C["box_data"], fontsize=10, bold=True,
                subtext="STRING v11.5 · GTEx v8\nOMIM/HPO · DGIdb")

    ax.text(0.3, 6.65, "GENETIC ARCHITECTURE", fontsize=10, fontweight="bold", color=C["box_method"], va="center")
    y2 = 5.6
    methods = [
        (0.3, "LDSC", "h² · rg"),
        (2.6, "Genomic SEM", "EFA → CFA"),
        (4.9, "MTAG", "Power boost"),
        (7.2, "S-LDSC", "Tissue enrichment"),
        (9.5, "LAVA", "Local rg"),
    ]
    for x, name, sub in methods:
        rounded_box(ax, (x, y2), 2.0, 0.8, name, C["box_method"], fontsize=10, bold=True, subtext=sub)
    rounded_box(ax, (11.8, y2), 1.5, 0.8, "MR", C["box_method"], fontsize=10, bold=True, subtext="Bidirectional")

    for x_start in (1.7, 4.9):
        arrow(ax, (x_start, 7.2), (x_start, 6.45))
    arrow(ax, (8.1, 7.2), (8.2, 6.45))

    ax.text(0.3, 5.1, "GENE DISCOVERY & PRIORITIZATION", fontsize=10, fontweight="bold", color=C["box_result"], va="center")
    y3 = 4.0
    rounded_box(ax, (0.3, y3), 2.5, 0.8, "MAGMA", C["box_result"], fontsize=10, bold=True, subtext="267 genes (P < 2.5e-6)")
    rounded_box(ax, (3.2, y3), 3.0, 0.8, "GNN Ensemble", C["box_result"], fontsize=10, bold=True,
                subtext="GAT + GraphSAGE + GCN\n+ Gradient Boosting")
    rounded_box(ax, (6.6, y3), 2.5, 0.8, "eQTL Integration", C["box_result"], fontsize=10, bold=True,
                subtext="GTEx v8 · Functional\nevidence scoring")
    rounded_box(ax, (9.5, y3), 2.5, 0.8, "Drug Repurposing", C["box_result"], fontsize=10, bold=True,
                subtext="DGIdb v5.0\nDrug-gene network")

    arrow(ax, (5.9, 5.6), (4.7, 4.85))
    arrow(ax, (1.55, 5.6), (1.55, 4.85))
    arrow(ax, (2.8, 4.4), (3.2, 4.4))
    arrow(ax, (6.2, 4.4), (6.6, 4.4))
    arrow(ax, (9.1, 4.4), (9.5, 4.4))

    ax.text(0.3, 3.5, "VALIDATION", fontsize=10, fontweight="bold", color=C["box_valid"], va="center")
    y4 = 2.4
    rounded_box(ax, (0.3, y4), 3.0, 0.8, "Cross-ancestry", C["box_valid"], fontsize=10, bold=True,
                subtext="EUR → AFR (n=106)\nEUR → CSA (n=181)")
    rounded_box(ax, (3.7, y4), 3.0, 0.8, "Independent Gene Sets", C["box_valid"], fontsize=10, bold=True,
                subtext="Drug targets · Pre-GWAS lit.\nPermutation P < 0.01")
    rounded_box(ax, (7.1, y4), 3.0, 0.8, "Sensitivity Analysis", C["box_valid"], fontsize=10, bold=True,
                subtext="−FemaleProlapse · −Bladder\nModel robustness")
    rounded_box(ax, (10.5, y4), 2.8, 0.8, "MR Pleiotropy", C["box_valid"], fontsize=10, bold=True,
                subtext="Egger intercept\nAll P > 0.05")

    arrow(ax, (4.7, 4.0), (5.2, 3.25))
    arrow(ax, (8.1, 7.2), (1.8, 3.25))

    ax.text(0.3, 1.9, "KEY FINDINGS", fontsize=10, fontweight="bold", color=C["box_app"], va="center")
    findings = [
        ("2-Factor\nGenetic Structure", 0.3),
        ("267 MAGMA\nGenes", 3.0),
        ("WNT4 / WT1\nTop Candidates", 5.7),
        ("BPH → UI\nCausal Path", 8.4),
        ("Drug\nCandidates", 11.1),
    ]
    y5 = 0.8
    for text, x in findings:
        rounded_box(ax, (x, y5), 2.3, 0.8, text, C["box_app"], fontsize=9, bold=True)

    arrow(ax, (1.8, 2.4), (1.45, 1.65))
    arrow(ax, (5.2, 2.4), (6.85, 1.65))

    plt.tight_layout(pad=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig1_study_design.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> Fig1_study_design.png/pdf")


def make_graphical_abstract() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6, 6.6, "Shared Genetic Architecture of Pelvic Floor Disorders",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C["text"])
    ax.text(6, 6.25, "Multi-trait GWAS · Genomic SEM · Graph Neural Network",
            ha="center", va="center", fontsize=10, color="#666666", fontstyle="italic")

    rounded_box(ax, (0.3, 4.5), 3.2, 1.4, "6 PFD Phenotypes", C["box_data"], fontsize=11, bold=True,
                subtext="POP · BPH · Bladder\nConstipation · FemaleProlapse\nIncontinence")

    cx, cy = 5.1, 4.8
    rounded_box(ax, (cx, cy), 2.2, 0.9, "Factor 1\nFemalePelvic", "#E15759", fontsize=9, bold=True)
    rounded_box(ax, (cx, cy - 1.2), 2.2, 0.9, "Factor 2\nUrinary", "#4E79A7", fontsize=9, bold=True)
    ax.annotate("", xy=(cx + 1.1, cy), xytext=(cx + 1.1, cy - 0.25),
                arrowprops=dict(arrowstyle="<->", color=C["highlight"], lw=2))
    ax.text(cx + 1.7, cy - 0.15, "rg", fontsize=8, color=C["highlight"], fontweight="bold")

    arrow(ax, (3.5, 5.2), (cx, 5.2))
    rounded_box(ax, (8.2, 4.5), 3.2, 1.4, "Top Candidates", C["box_result"], fontsize=11, bold=True,
                subtext="WNT4 · WT1 · FGFR2\nESR1 · SMAD3 · LOXL1\n267 genes prioritized")
    arrow(ax, (7.3, 5.2), (8.2, 5.2))

    rounded_box(ax, (0.3, 2.2), 3.2, 1.6, "GNN Ensemble", C["box_method"], fontsize=11, bold=True,
                subtext="PPI network + GWAS features\nGAT + GraphSAGE + GCN\n3 independent validations")
    rounded_box(ax, (4.2, 2.2), 3.2, 1.6, "Cross-ancestry\nValidation", C["box_valid"], fontsize=11, bold=True,
                subtext="EUR → AFR: 76.4% concordance\nWNT4 locus replicated\nacross ancestries")
    rounded_box(ax, (8.2, 2.2), 3.2, 1.6, "Therapeutic\nImplications", C["box_app"], fontsize=11, bold=True,
                subtext="Drug repurposing candidates\nBPH → Incontinence causal path\nTGF-β / ECM therapeutic axis")

    arrow(ax, (1.9, 4.5), (1.9, 3.85))
    arrow(ax, (6.2, 3.6), (5.8, 3.85))
    arrow(ax, (9.8, 4.5), (9.8, 3.85))

    ax.plot([0.3, 11.4], [1.8, 1.8], color="#CCCCCC", lw=1.5, zorder=0)
    findings_text = [
        ("h² = 0.003–0.034", 1.5),
        ("14/15 rg sig.", 4.0),
        ("CFI = 0.948", 6.0),
        ("MR: BPH→UI β=0.17", 8.5),
        ("P_perm < 0.01", 11.0),
    ]
    for text, x in findings_text:
        ax.text(x, 1.45, text, ha="center", va="center", fontsize=8, color="#555555", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5", edgecolor="#DDDDDD", alpha=0.8))

    plt.tight_layout(pad=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Graphical_Abstract.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> Graphical_Abstract.png/pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating Fig 1 and Graphical Abstract...")
    make_fig1()
    make_graphical_abstract()
    print("Done")


if __name__ == "__main__":
    main()
