"""
RQ3 / H3 (renamed from earlier RQ2/H2 label) -- Coefficient-learning
strategy comparison across Conditions 2-6.

STATISTICAL METHOD (updated)
-----------------------------
The previous version ran a one-way repeated-measures ANOVA with
Greenhouse-Geisser correction and Bonferroni-corrected pairwise t-tests
over 80 per-class AP values, treating each COCO category as a "subject".
That is invalid: categories are not independent repeated measures of the
same underlying construct, they are correlated components of one mAP
estimate.

Test 1 (updated): rather than an omnibus ANOVA over classes, this script
now runs a predefined, limited family of pairwise PAIRED IMAGE-LEVEL
BOOTSTRAP contrasts among Conditions 2-6, with Holm-Bonferroni correction
across the family. This directly answers "is condition X practically and
statistically better than condition Y" without the class-independence
assumption.

Test 2 (updated): Condition 3 (Dirichlet) vs Condition 4 (Fisher) is one
of the planned pairwise contrasts in Test 1 and is reported there; it is
no longer a separate ad hoc per-class paired t-test.

Test 3 (UNCHANGED IN PRINCIPLE): the two-way comparison of learned
coefficient magnitudes (cls / bbox / obj) across the Dirichlet vs Fisher
strategies uses N = 6 ingredient models as the sampling unit. That is a
legitimate independent-ish sampling unit for this specific question (it is
not COCO categories), so a repeated-measures / mixed design over the 6
ingredient models remains defensible. It is retained here using paired
tests over the 6-model coefficient vectors, NOT over classes.

Decision rule for Test 1 (pre-specified, revised):
  H03 rejected for a specific contrast if, after Holm-Bonferroni adjustment
  across the planned contrast family:
    - Holm-adjusted one-sided p-value < 0.05, AND
    - 95% bootstrap CI lower bound > 0, AND
    - observed |ΔmAP| >= 0.5 pp.

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h3_predictions.json
      {
        "condition_2": "results/predictions/condition_2.json",
        "condition_3": "results/predictions/condition_3.json",
        "condition_4": "results/predictions/condition_4.json",
        "condition_5": "results/predictions/condition_5.json",
        "condition_6": "results/predictions/condition_6.json"
      }
  results/phase3_soup_results.json (optional; for Test 3 coefficient magnitudes only)
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
from scipy import stats

from yolof_soup.statistics.bootstrap_core import (
    apply_decision_rule,
    holm_adjust,
    load_bootstrap_manifest_inputs,
    load_json,
    run_comparison,
    save_result_artifacts,
)

RESULTS_DIR = pathlib.Path("results")
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth_heldout.json"
BOOTSTRAP_IMAGE_IDS_FILE = RESULTS_DIR / "bootstrap_manifest" / "bootstrap_image_ids.npy"
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h3_predictions.json"
SOUP_FILE_FOR_COEFFICIENTS = RESULTS_DIR / "phase3_soup_results.json"

ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.5
WORKERS = 4

CONDITION_LABELS = {
    "condition_2": "Condition2 (component uniform)",
    "condition_3": "Condition3 (Dirichlet)",
    "condition_4": "Condition4 (Fisher-weighted)",
    "condition_5": "M5 (shared learned α+β)",
    "condition_6": "M6 (tri-component learned α+β)",
}

# Planned pairwise contrast family for Test 1. Kept deliberately small to
# limit multiple-testing burden. M6 vs M5 is intentionally EXCLUDED here
# because it is the dedicated confirmatory RQ3(H3 in Ch.1 numbering)/H4a
# test and is run in h4a_paired_ttest_M6_vs_M5.py to avoid double-counting
# the same contrast in two Holm-corrected families.
PLANNED_CONTRASTS = [
    ("condition_2", "condition_3"),
    ("condition_2", "condition_4"),
    ("condition_2", "condition_5"),
    ("condition_2", "condition_6"),
    ("condition_3", "condition_4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3/H3 planned pairwise bootstrap contrasts.")
    parser.add_argument("--ground-truth", type=pathlib.Path, default=GROUND_TRUTH_FILE)
    parser.add_argument("--bootstrap-image-ids", type=pathlib.Path, default=BOOTSTRAP_IMAGE_IDS_FILE)
    parser.add_argument("--predictions-config", type=pathlib.Path, default=PREDICTIONS_CONFIG_FILE)
    parser.add_argument("--coefficients-file", type=pathlib.Path, default=SOUP_FILE_FOR_COEFFICIENTS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--practical-threshold", type=float, default=PRACTICAL_THRESHOLD)
    return parser.parse_args()


def run_test1_pairwise_bootstrap(
    predictions_config: dict,
    ground_truth: dict,
    ground_truth_path: pathlib.Path,
    bootstrap_image_ids: np.ndarray,
    workers: int,
) -> list[dict]:
    results = []
    for key_a, key_b in PLANNED_CONTRASTS:
        result = run_comparison(
            name=f"H3-Test1: {CONDITION_LABELS[key_a]} vs {CONDITION_LABELS[key_b]}",
            model_a_name=CONDITION_LABELS[key_a],
            model_a_predictions_path=predictions_config[key_a],
            model_b_name=CONDITION_LABELS[key_b],
            model_b_predictions_path=predictions_config[key_b],
            ground_truth=ground_truth,
            ground_truth_path=ground_truth_path,
            bootstrap_image_ids=bootstrap_image_ids,
            workers=workers,
        )
        results.append(result)
    return results


def run_test3_coefficient_magnitude_analysis(coefficients_file: pathlib.Path) -> dict | None:
    """
    Legitimate paired analysis: N = 6 ingredient models are the sampling
    unit, not COCO categories. Retained largely as before.
    """
    if not coefficients_file.is_file():
        print(f"NOTE: {coefficients_file} not found; Test 3 skipped.")
        return None

    soup = load_json(coefficients_file)
    coef3 = soup.get("condition_3", {}).get("coefficients")
    coef4 = soup.get("condition_4", {}).get("coefficients")

    if not coef3 or not coef4:
        print("NOTE: 'coefficients' key missing in condition_3/condition_4; Test 3 skipped.")
        return None

    components = ["cls", "bbox", "obj"]
    strategy_d = np.array([coef3[c] for c in components])  # shape (3, 6)
    strategy_f = np.array([coef4[c] for c in components])  # shape (3, 6)

    if strategy_d.shape[1] < 2 or strategy_f.shape[1] < 2:
        print("NOTE: fewer than 2 ingredient models available; Test 3 reported descriptively only.")
        return {
            "note": "insufficient replicates for inferential test",
            "means": {c: {"dirichlet": float(np.mean(coef3[c])), "fisher": float(np.mean(coef4[c]))} for c in components},
        }

    strategy_d_mean_per_model = strategy_d.mean(axis=0)  # mean over components, per ingredient model
    strategy_f_mean_per_model = strategy_f.mean(axis=0)

    t_stat, p_val = stats.ttest_rel(strategy_f_mean_per_model, strategy_d_mean_per_model)

    per_component = {}
    for i, comp in enumerate(components):
        t_c, p_c = stats.ttest_rel(strategy_f[i], strategy_d[i])
        per_component[comp] = {"t": float(t_c), "p": float(p_c)}

    print("\nTest 3 -- Coefficient magnitude, Dirichlet vs Fisher (unit = N=6 ingredient models)")
    print(f"  Overall strategy effect: t = {t_stat:.4f}, p = {p_val:.4f}")
    for comp, values in per_component.items():
        print(f"  Component {comp}: t = {values['t']:.4f}, p = {values['p']:.4f}")

    return {
        "sampling_unit": "N=6 ingredient models (legitimate; not COCO categories)",
        "overall_strategy_effect": {"t": float(t_stat), "p": float(p_val)},
        "per_component": per_component,
    }


def main() -> None:
    args = parse_args()

    ground_truth, bootstrap_image_ids = load_bootstrap_manifest_inputs(
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids_path=args.bootstrap_image_ids,
    )

    predictions_config = load_json(args.predictions_config)
    required_keys = {"condition_2", "condition_3", "condition_4", "condition_5", "condition_6"}
    missing = required_keys - set(predictions_config)
    if missing:
        raise ValueError(f"{args.predictions_config} is missing keys: {sorted(missing)}")

    print("=" * 70)
    print("TEST 1 -- Planned pairwise image-level bootstrap contrasts (Conditions 2-6)")
    print("=" * 70)

    test1_results = run_test1_pairwise_bootstrap(
        predictions_config=predictions_config,
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    raw_p_values = [r["bootstrap"]["p_one_sided_model_b_greater"] for r in test1_results]
    holm_p_values = holm_adjust(raw_p_values)

    any_significant_contrast = False
    max_abs_diff = 0.0

    for result, p_holm in zip(test1_results, holm_p_values):
        apply_decision_rule(result, p_holm, alpha=args.alpha, practical_threshold_ap_points=args.practical_threshold)
        any_significant_contrast = any_significant_contrast or result["decision"]["support_directional_superiority"]
        max_abs_diff = max(max_abs_diff, abs(result["observed_difference_ap_points"]))

        print(f"\n  {result['name']}")
        print(f"    ΔAP = {result['observed_difference_ap_points']:+.4f} pp, 95% CI [{result['bootstrap']['ci_95_percentile_lower']:+.4f}, {result['bootstrap']['ci_95_percentile_upper']:+.4f}]")
        print(f"    Holm p = {result['bootstrap']['p_one_sided_holm_adjusted']:.6f}, decision = {'SUPPORTED' if result['decision']['support_directional_superiority'] else 'NOT SUPPORTED'}")

    decision_h3 = (
        "REJECT H03 (at least one strategy differs practically and statistically)"
        if any_significant_contrast
        else "FAIL TO REJECT H03 (strategy-level differences are small / not significant)"
    )
    print(f"\nOverall Test 1 decision: {decision_h3} (max |ΔAP| observed = {max_abs_diff:.4f} pp)")

    print("\n" + "=" * 70)
    print("TEST 3 -- Coefficient magnitude analysis (N=6 ingredient models, legitimate unit)")
    print("=" * 70)
    test3_result = run_test3_coefficient_magnitude_analysis(args.coefficients_file)

    output_dir = RESULTS_DIR / "h3_rq3_bootstrap"
    for index, result in enumerate(test1_results, start=1):
        save_result_artifacts(result, output_dir, file_stem=f"contrast_{index}")

    output = {
        "method": "Test 1: paired image-level bootstrap over planned Condition 2-6 contrasts. Test 3: paired t-tests over N=6 ingredient-model coefficient magnitudes.",
        "family": "RQ3 / H3",
        "test1_pairwise_contrasts": test1_results,
        "test1_overall_decision": decision_h3,
        "test3_coefficient_magnitude": test3_result,
        "multiple_testing": {"method": "Holm-Bonferroni", "alpha": args.alpha, "n_comparisons": len(test1_results)},
    }

    out_path = RESULTS_DIR / "h3_rq3_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
