"""
Descriptive-only per-component temperature calibration report
(beta_cls, beta_bbox, beta_obj). NOT a formal hypothesis in the final
thesis draft.

This is the renamed successor to h4b_beta_onesample_ttest.py (already
converted to descriptive-only in an earlier revision of this repo; content
unchanged here, only renamed to drop the misleading H4b label). Ch.4's
Preliminary Note states explicitly: "temperature calibration deviation
analysis... formerly... H4b... in the earlier draft, [is] exclusively
maintained as descriptive context." Ch.1 Sec.1.5 confirms the same.
Chapter 3 Sec.3.5 still labels this "the statistical test for RQ4b and
H4b" with a formal Bonferroni-adjusted one-sample t-test decision -- that
description is stale (flagged in STATISTICS_UPDATE_NOTES.md).

h4b_beta_onesample_ttest.py is left in the repo unchanged for now;
recommend deleting it once this file is verified.

WHY THIS REMAINS DESCRIPTIVE
------------------------------
Condition 6 (M6) learns exactly ONE beta_cls, ONE beta_bbox, and ONE
beta_obj value from a single optimisation run. There is no natural sample
of independent replicate beta values to support a one-sample t-test or an
ANOVA, and the N=6 ingredient models are not independent replicates of
beta (beta is a property of the merged model, optimised once). Unless you
rerun the beta optimisation under multiple independently resampled
calibration subsets, this stays descriptive.

Input file
----------
  results/phase3_soup_results.json
    Key: condition_6 -> "beta_values": {"cls": ..., "bbox": ..., "obj": ...}
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
BETA_NULL = 1.0
N_BOOT = 10_000
RNG_SEED = 42


def bootstrap_ci_mean(values: np.ndarray, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(N_BOOT)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_bootstrap_beta_replicates(component: str, replicate_values: np.ndarray) -> dict:
    """Valid ONLY for genuine independent beta-optimisation replicates."""
    ci_lo, ci_hi = bootstrap_ci_mean(replicate_values)
    deviates_from_one = not (ci_lo <= BETA_NULL <= ci_hi)
    return {
        "component": component, "n_replicates": int(len(replicate_values)),
        "mean": float(replicate_values.mean()),
        "std": float(replicate_values.std(ddof=1)) if len(replicate_values) > 1 else None,
        "ci_95_lower": ci_lo, "ci_95_upper": ci_hi, "deviates_from_1_0": deviates_from_one,
        "note": "Valid only if these are independent beta-optimisation replicates.",
    }


def main() -> None:
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    beta_data = soup["condition_6"].get("beta_values")
    if beta_data is None:
        raise KeyError("'beta_values' not found in condition_6.")

    components = ["cls", "bbox", "obj"]
    report = {}

    print("\n" + "=" * 70)
    print("DESCRIPTIVE per-component temperature calibration (NOT a formal hypothesis)")
    print("=" * 70)

    for comp in components:
        raw_value = beta_data[comp]

        if isinstance(raw_value, list) and len(raw_value) >= 2:
            replicate_values = np.array(raw_value, dtype=float)
            print(f"\n  Component '{comp}': {len(replicate_values)} independent replicates detected.")
            report[comp] = run_bootstrap_beta_replicates(comp, replicate_values)
            print(f"    Mean beta_{comp} = {report[comp]['mean']:.6f}")
            print(f"    95% bootstrap CI = [{report[comp]['ci_95_lower']:.6f}, {report[comp]['ci_95_upper']:.6f}]")
        else:
            single_value = float(raw_value[0]) if isinstance(raw_value, list) else float(raw_value)
            print(f"\n  Component '{comp}': single learned value, no independent replicates available.")
            print(f"    beta_{comp} = {single_value:.6f} (descriptive only)")
            report[comp] = {
                "component": comp, "n_replicates": 1, "value": single_value,
                "note": "Single learned scalar from one optimisation run; no formal hypothesis test performed.",
            }

    out_path = RESULTS_DIR / "descriptive_beta_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "status": "descriptive_only",
                "rationale": (
                    "Per Ch.1 Sec.1.5 and Ch.4 Preliminary Note, this analysis (formerly "
                    "labeled H4b) is retained exclusively as descriptive context and is not "
                    "a formal hypothesis in the final thesis."
                ),
                "beta_report": report,
            },
            f, indent=2, default=float,
        )
    print(f"\nDescriptive report saved -> {out_path}")


if __name__ == "__main__":
    main()
