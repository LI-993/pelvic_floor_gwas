"""LDSC log-file parsers.

LDSC writes free-form text logs; the regexes below extract the fields the
downstream code consumes (heritability, genetic correlation, intercepts,
partitioned-h2 enrichment).
"""

from __future__ import annotations

import re
from pathlib import Path


_H2_P1 = re.compile(
    r"Heritability of phenotype 1.*?Total Observed scale h2: ([\d.\-eE]+) \(([\d.\-eE]+)\)",
    re.DOTALL,
)
_H2_P2 = re.compile(
    r"Heritability of phenotype 2/2.*?Total Observed scale h2: ([\d.\-eE]+) \(([\d.\-eE]+)\)",
    re.DOTALL,
)
_RG = re.compile(r"Genetic Correlation: ([\d.\-eE]+) \(([\d.\-eE]+)\)")
_Z = re.compile(r"Z-score: ([\d.\-eE]+)")
_P = re.compile(r"P: ([\d.\-eE]+)")
_INTERCEPT = re.compile(r"Intercept:\s+([\d.\-eE]+)\s*\(([\d.\-eE]+)\)?")
_INTERCEPTS_VALUE_ONLY = re.compile(r"Intercept:\s+([\d.\-eE]+)")
_LAMBDA_GC = re.compile(r"Lambda GC: ([\d.\-eE]+)")
_MEAN_CHI2 = re.compile(r"Mean Chi\^2: ([\d.\-eE]+)")
_TOTAL_H2 = re.compile(r"Total Observed scale h2: ([\d.\-eE]+) \(([\d.\-eE]+)\)")
_CATEGORIES = re.compile(r"Categories: (.+)")
_ENRICHMENT = re.compile(r"Enrichment: (.+)")


def parse_rg_log(log_path: Path) -> dict:
    """Parse LDSC `--rg` log into a dict of floats."""
    content = Path(log_path).read_text()
    out: dict[str, float] = {}

    if m := _H2_P1.search(content):
        out["h2_p1"], out["h2_p1_se"] = float(m.group(1)), float(m.group(2))
    if m := _H2_P2.search(content):
        out["h2_p2"], out["h2_p2_se"] = float(m.group(1)), float(m.group(2))
    if m := _RG.search(content):
        out["rg"], out["rg_se"] = float(m.group(1)), float(m.group(2))
    if m := _Z.search(content):
        out["z"] = float(m.group(1))
    if m := _P.search(content):
        out["p"] = float(m.group(1))

    # Three intercepts in an rg log: h2_p1, h2_p2, gcov. Capture in order.
    intercepts = _INTERCEPTS_VALUE_ONLY.findall(content)
    if len(intercepts) >= 3:
        out["gcov_int"] = float(intercepts[2])

    return out


def parse_h2_log(log_path: Path) -> dict:
    """Parse LDSC `--h2` (or partitioned S-LDSC) log."""
    content = Path(log_path).read_text()
    out: dict = {}

    if m := _TOTAL_H2.search(content):
        out["h2"], out["h2_se"] = float(m.group(1)), float(m.group(2))
    if m := _LAMBDA_GC.search(content):
        out["lambda_gc"] = float(m.group(1))
    if m := _MEAN_CHI2.search(content):
        out["mean_chi2"] = float(m.group(1))
    if m := _INTERCEPT.search(content):
        out["intercept"], out["intercept_se"] = float(m.group(1)), float(m.group(2))
    if m := _CATEGORIES.search(content):
        out["categories"] = m.group(1).split()
    if m := _ENRICHMENT.search(content):
        out["enrichment"] = [float(x) for x in m.group(1).split()]

    return out
