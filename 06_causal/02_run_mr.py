#!/usr/bin/env python3
"""Bidirectional Mendelian randomization between pelvic-floor phenotypes.

Implements IVW, weighted median, and MR-Egger from scratch using the
standardized sumstats. Instrument selection uses the GWAS-significance
threshold; the LD clumping is the simple distance-based version from
utils.clumping (proper LD-aware clumping needs PLINK + a reference).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PHENOTYPES, RESULTS_DIR
from utils.clumping import simple_clump
from utils.io import load_sumstats

P_THRESHOLD = 5e-8
WINDOW_BP = 10_000_000   # 10 Mb for instrument clumping
MIN_SNPS_IVW = 3
MIN_SNPS_EGGER = 5

# Pairs tested in the manuscript (forward + reverse run automatically).
PAIRS: list[tuple[str, str]] = [
    ("POP", "FemaleProlapse"),
    ("POP", "Incontinence"),
    ("FemaleProlapse", "Incontinence"),
    ("BPH", "POP"),
    ("BPH", "Incontinence"),
    ("Constipation", "POP"),
    ("Constipation", "FemaleProlapse"),
]


def harmonize(exposure: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame | None:
    merged = exposure.merge(outcome, on="SNP", suffixes=("_exp", "_out"))
    if merged.empty:
        return None

    rows = []
    for _, r in merged.iterrows():
        if r["A1_exp"] == r["A1_out"] and r["A2_exp"] == r["A2_out"]:
            beta_out = r["BETA_out"]
        elif r["A1_exp"] == r["A2_out"] and r["A2_exp"] == r["A1_out"]:
            beta_out = -r["BETA_out"]
        else:
            continue
        rows.append({
            "SNP": r["SNP"],
            "beta_exp": r["BETA_exp"], "se_exp": r["SE_exp"],
            "beta_out": beta_out, "se_out": r["SE_out"],
        })
    return pd.DataFrame(rows) if rows else None


def ivw(beta_exp, se_exp, beta_out, se_out) -> tuple[float, float, float]:
    ratio = beta_out / beta_exp
    ratio_se = np.abs(se_out / beta_exp)
    weights = 1 / ratio_se**2
    beta = np.sum(weights * ratio) / np.sum(weights)
    se = np.sqrt(1 / np.sum(weights))
    p = 2 * stats.norm.sf(abs(beta / se))
    return beta, se, p


def weighted_median(beta_exp, se_exp, beta_out, se_out) -> tuple[float, float, float]:
    ratio = beta_out / beta_exp
    ratio_se = np.abs(se_out / beta_exp)
    weights = 1 / ratio_se**2
    order = np.argsort(ratio)
    cum = np.cumsum(weights[order]) / weights.sum()
    median_idx = np.searchsorted(cum, 0.5)
    beta = ratio[order][median_idx]
    se = np.std(ratio) / np.sqrt(len(ratio))  # bootstrap-free approximation
    p = 2 * stats.norm.sf(abs(beta / se))
    return beta, se, p


def mr_egger(beta_exp, se_exp, beta_out, se_out) -> tuple[float, float, float, float, float]:
    weights = 1 / se_out**2
    X = np.column_stack([np.ones(len(beta_exp)), beta_exp])
    W = np.diag(weights)
    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    coef = XtWX_inv @ X.T @ W @ beta_out
    intercept, slope = coef
    residuals = beta_out - X @ coef
    mse = np.sum(weights * residuals**2) / (len(beta_out) - 2)
    se_coef = np.sqrt(np.diag(XtWX_inv) * mse)
    se_intercept, se_slope = se_coef
    p_slope = 2 * stats.norm.sf(abs(slope / se_slope))
    p_intercept = 2 * stats.norm.sf(abs(intercept / se_intercept))
    return slope, se_slope, p_slope, intercept, p_intercept


def run_pair(exposure: str, outcome: str, exp_df: pd.DataFrame, out_df: pd.DataFrame) -> dict | None:
    print(f"\n{'=' * 60}\nMR: {exposure} -> {outcome}\n{'=' * 60}")
    instruments = exp_df[exp_df["P"] < P_THRESHOLD]
    print(f"  Significant instruments: {len(instruments)}")
    if instruments.empty:
        return None

    clumped = simple_clump(instruments, window=WINDOW_BP)
    print(f"  After clumping: {len(clumped)}")
    if len(clumped) < MIN_SNPS_IVW:
        return None

    harm = harmonize(clumped, out_df)
    if harm is None or len(harm) < MIN_SNPS_IVW:
        print("  Insufficient harmonized SNPs")
        return None
    print(f"  Harmonized SNPs: {len(harm)}")

    beta_exp = harm["beta_exp"].values
    se_exp = harm["se_exp"].values
    beta_out = harm["beta_out"].values
    se_out = harm["se_out"].values

    out: dict = {"exposure": exposure, "outcome": outcome, "n_snps": len(harm)}

    out["ivw_beta"], out["ivw_se"], out["ivw_p"] = ivw(beta_exp, se_exp, beta_out, se_out)
    print(f"  IVW: beta={out['ivw_beta']:.4f}, SE={out['ivw_se']:.4f}, P={out['ivw_p']:.2e}")

    out["wm_beta"], out["wm_se"], out["wm_p"] = weighted_median(beta_exp, se_exp, beta_out, se_out)
    print(f"  Weighted Median: beta={out['wm_beta']:.4f}, P={out['wm_p']:.2e}")

    if len(harm) >= MIN_SNPS_EGGER:
        out["egger_beta"], out["egger_se"], out["egger_p"], out["egger_intercept"], out["egger_intercept_p"] = mr_egger(
            beta_exp, se_exp, beta_out, se_out,
        )
        print(f"  MR-Egger: beta={out['egger_beta']:.4f}, P={out['egger_p']:.2e}")
        print(f"  Egger intercept: {out['egger_intercept']:.4f}, P={out['egger_intercept_p']:.2e}")

    return out


def main() -> None:
    out_dir = RESULTS_DIR / "mr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bidirectional Mendelian Randomization")
    print("=" * 60)

    print("\nLoading sumstats...")
    sumstats = {p: load_sumstats(p) for p in PHENOTYPES}

    results = []
    for exp, outcome in PAIRS:
        for direction in ((exp, outcome), (outcome, exp)):
            r = run_pair(*direction, sumstats[direction[0]], sumstats[direction[1]])
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results)
        out_path = out_dir / "mr_bidirectional_results.csv"
        df.to_csv(out_path, index=False)

        print("\n" + "=" * 60)
        print("Significant relationships (IVW p<0.05):")
        sig = df[df["ivw_p"] < 0.05]
        if sig.empty:
            print("  none")
        else:
            for _, r in sig.iterrows():
                arrow = "+" if r["ivw_beta"] > 0 else "-"
                print(f"  {r['exposure']} -> {r['outcome']}: beta={r['ivw_beta']:.3f} ({arrow}), p={r['ivw_p']:.2e}")
        print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
