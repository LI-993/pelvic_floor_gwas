#!/usr/bin/env python3
"""Parse S-LDSC log files into a heritability summary + per-annotation enrichment table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PHENOTYPES, RESULTS_DIR
from utils.ldsc import parse_h2_log


def main() -> None:
    results_dir = RESULTS_DIR / "sldsc"
    print("S-LDSC Results Summary")
    print("=" * 60)

    parsed: dict[str, dict] = {}
    for p in PHENOTYPES:
        log = results_dir / f"{p}_baselineLD.log"
        if not log.exists():
            print(f"  {p}: log missing ({log})")
            continue
        parsed[p] = parse_h2_log(log)
        rec = parsed[p]
        print(
            f"\n{p}:"
            f"\n  h2 = {rec.get('h2', float('nan')):.4f} (SE: {rec.get('h2_se', float('nan')):.4f})"
            f"\n  Lambda GC = {rec.get('lambda_gc', float('nan')):.4f}"
            f"\n  Mean Chi^2 = {rec.get('mean_chi2', float('nan')):.4f}"
            f"\n  Intercept = {rec.get('intercept', float('nan')):.4f}"
        )

    h2_rows = [{
        "Phenotype": p,
        "h2": rec.get("h2"),
        "h2_SE": rec.get("h2_se"),
        "Lambda_GC": rec.get("lambda_gc"),
        "Mean_Chi2": rec.get("mean_chi2"),
        "Intercept": rec.get("intercept"),
        "Intercept_SE": rec.get("intercept_se"),
    } for p, rec in parsed.items()]
    h2_df = pd.DataFrame(h2_rows)
    h2_df.to_csv(results_dir / "sldsc_h2_summary.csv", index=False)
    print(f"\nh2 summary: {results_dir / 'sldsc_h2_summary.csv'}")

    # Per-annotation enrichment, indexed off the POP categories.
    if "POP" in parsed and "categories" in parsed["POP"]:
        clean_cats = [c.replace("L2_0", "") for c in parsed["POP"]["categories"]]
        enrich = {"Category": clean_cats}
        for p in PHENOTYPES:
            if p in parsed and "enrichment" in parsed[p]:
                vals = parsed[p]["enrichment"]
                if len(vals) < len(clean_cats):
                    vals = vals + [None] * (len(clean_cats) - len(vals))
                enrich[p] = vals[: len(clean_cats)]
        enrich_df = pd.DataFrame(enrich)
        enrich_df.to_csv(results_dir / "sldsc_enrichment.csv", index=False)
        print(f"Enrichment: {results_dir / 'sldsc_enrichment.csv'}")

        print("\nTop enriched categories (>50x):")
        for p in PHENOTYPES:
            if p in parsed and "enrichment" in parsed[p]:
                top = sorted(
                    [(i, e) for i, e in enumerate(parsed[p]["enrichment"]) if e > 50],
                    key=lambda x: x[1],
                    reverse=True,
                )
                if top:
                    print(f"\n{p}:")
                    for i, val in top[:5]:
                        if i < len(clean_cats):
                            print(f"  {clean_cats[i]}: {val:.1f}")


if __name__ == "__main__":
    main()
