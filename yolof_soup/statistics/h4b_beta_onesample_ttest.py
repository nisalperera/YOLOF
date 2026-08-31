"""
RQ4b / H4b -- Per-component temperature scalars beta_cls, beta_bbox,
              beta_obj: descriptive calibration report (NOT a formal
              hypothesis test).

STATISTICAL METHOD (updated -- important change)
-------------------------------------------------
The previous version ran one-sample t-tests against beta = 1.0 and a
one-way ANOVA for heterogeneity across components, treating either the
six ingredient-model-conditioned coefficients, or a single learned scalar
repeated artificially, as if they were independent replicates.

This is not defensible as currently designed:
  - Condition 6 (M6) learns exactly ONE beta_cls, ONE beta_bbox, and ONE
    beta_obj value from a single optimisation run over the calibration
    split. There is no natural sample of independent replicate beta
    values to feed a one-sample t-test or an ANOVA.
  - Treating the N=6 ingredient models as if each contributed an
    independent beta observation is incorrect, because beta is a
    property of the MERGED model, optimised once, not a property of any
    individual ingredient model.

Unless you deliberately rerun the beta optimisation step multiple times
with independently resampled calibration subsets or independent random
seeds (producing genuine replicate beta_cls / beta_bbox / beta_obj
values), this section should report the three learned scalars
descriptively and qualitatively, without a formal p-value.

If you DO have genuine independent replicates (e.g., beta re-optimised
under K >= 5 independently resampled calibration subsets with different
seeds), see the `run_bootstrap_beta_replicates` function below, which
applies a bootstrap CI to those genuine replicates -- this is valid
because the replicates themselves (not COCO categories) are the
resampling unit.

Input file
----------
  results/phase3_soup_results.json
    Key: condition_6 -> "beta_values": {"cls": ..., "bbox": ..., "obj": ...}
    Each value is normally a single float (one optimisation run).
    OPTIONAL: if you have genuine independent replicates, each value may
    instead be a list of floats, one per independent optimisation run.
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
    """
    Valid ONLY if replicate_values are genuine independent replicates of
    the beta-optimisation procedure (e.g., different random seeds /
    different resampled calibration subsets) -- NOT ingredient models,
    and NOT COCO categories.
    """
    ci_lo, ci_hi = bootstrap_ci_mean(replicate_values)
    deviates_from_one = not (ci_lo <= BETA_NULL <= ci_hi)
    return {
        "component": component,
        "n_replicates": int(len(replicate_values)),
        "mean": float(replicate_values.mean()),
        "std": float(replicate_values.std(ddof=1)) if len(replicate_values) > 1 else None,
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        "deviates_from_1_0": deviates_from_one,
        "note": "Valid only if these are independent beta-optimisation replicates.",
    }


def main() -> None:
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    beta_data = soup["condition_6"].get("beta_values")
    if beta_data is None:
        raise KeyError("'beta_values' not found in condition_6. Ensure soup_construction.py saves beta_cls, beta_bbox, beta_obj.")

    components = ["cls", "bbox", "obj"]
    report = {}

    print("\n" + "=" * 70)
    print("H4b -- Learned temperature scalars (DESCRIPTIVE REPORT, not a formal test)")
    print("=" * 70)

    for comp in components:
        raw_value = beta_data[comp]

        if isinstance(raw_value, list) and len(raw_value) >= 2:
            replicate_values = np.array(raw_value, dtype=float)
            print(f"\n  Component '{comp}': {len(replicate_values)} independent replicates detected.")
            report[comp] = run_bootstrap_beta_replicates(comp, replicate_values)
            print(f"    Mean beta_{comp} = {report[comp]['mean']:.6f}")
            print(f"    95% bootstrap CI = [{report[comp]['ci_95_lower']:.6f}, {report[comp]['ci_95_upper']:.6f}]")
            print(f"    Deviates from 1.0: {report[comp]['deviates_from_1_0']}")
        else:
            single_value = float(raw_value[0]) if isinstance(raw_value, list) else float(raw_value)
            print(f"\n  Component '{comp}': single learned value, no independent replicates available.")
            print(f"    beta_{comp} = {single_value:.6f} (descriptive only; deviation from 1.0 not tested)")
            report[comp] = {
                "component": comp,
                "n_replicates": 1,
                "value": single_value,
                "note": (
                    "Single learned scalar from one optimisation run. No formal "
                    "hypothesis test performed. To test deviation from 1.0 "
                    "formally, rerun beta optimisation under >= 5 independently "
                    "resampled calibration subsets and populate this field as a list."
                ),
            }

    out_path = RESULTS_DIR / "h4b_results.json"
    with open(out_path, "w") as f:
        json.dump({"family": "RQ4b / H4b", "type": "descriptive_report", "beta_report": report}, f, indent=2, default=float)
    print(f"\nDescriptive report saved -> {out_path}")
    print(
        "\nReminder: this file no longer performs a formal H04b hypothesis "
        "test unless genuine independent beta replicates were supplied. "
        "Update Chapter 3/4 wording accordingly if you rely on this output."
    )


if __name__ == "__main__":
    main()
