"""
RQ2 / H2 (FINAL NUMBERING, Ch.1 Sec.1.5 / Ch.4 Preliminary Note) -- Strategy
equivalence across coefficient-learning strategies (Conditions 2, 3, 4,
M5, M6).

This is the renamed and relabeled successor to h3_rm_anova_conditions2to5.py.
In the FINAL thesis draft, this analysis is RQ2/H2, not RQ3/H3 -- Ch.4's
Preliminary Note states "strategy equivalence across Conditions 2 to 6
addresses H2 and RQ2" [Ch.4]. h3_rm_anova_conditions2to5.py is left in the
repo unchanged for now; recommend deleting it once this file is verified.

Ha2 (verbatim from the final thesis): "No coefficient strategy within this
family outperforms the others by >= 0.5pp; observed differences are small
and not statistically significant."

Note the directionality of H02/Ha2: the NULL is "differences are small";
the study's own alternative hypothesis Ha2 IS the null of "no practical
difference." H02 is rejected -- i.e. a practically/statistically important
strategy difference IS found -- only if at least one planned contrast
clears BOTH the statistical and the practical bar.

STATISTICAL METHOD
-------------------
Chapter 3 Sec.3.5 still describes this as a repeated-measures ANOVA with
Greenhouse-Geisser correction and Tukey HSD post-hoc over per-class AP
values (mislabeled there as "RQ3/H3"). That design treats 80 COCO
categories as independent repeated measures, which is invalid (see
bootstrap_core.py docstring and STATISTICS_UPDATE_NOTES.md). This script
replaces it with a small planned family of paired image-level bootstrap
contrasts, Holm-adjusted, plus a legitimate secondary analysis of learned
coefficient magnitudes across the N=6 ingredient models (a valid sampling
unit, unrelated to the per-class problem).

M6 vs M5 is intentionally EXCLUDED from this family -- that specific
contrast is the dedicated H3/RQ3 test in h3_rq3_m6_vs_m5.py, to avoid
double-counting it in two Holm-corrected families.

Planned contrasts (Test 1):
  Condition 2 vs Condition 3 (Dirichlet)
  Condition 2 vs Condition 4 (Fisher)
  Condition 2 vs Condition 5 (M5)
  Condition 2 vs Condition 6 (M6)
  Condition 3 vs Condition 4 (Dirichlet vs Fisher)

Decision rule (revised, pre-specified): H02 rejected if, after Holm
correction across the 5 planned contrasts, AT LEAST ONE contrast has:
  - Holm-adjusted one-sided p-value < 0.05, AND
  - 95% bootstrap CI lower bound > 0, AND
  - observed |ΔmAP| >= 0.5 pp.

Test 3 (coefficient magnitude, N=6 ingredient models -- legitimate unit,
unchanged in principle from the earlier draft) is retained as supplementary
evidence for strategy differences at the coefficient level.

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h2_predictions.json
      {
        "condition_2": "results/predictions/condition_2.json",
        "condition_3": "results/predictions/condition_3.json",
        "condition_4": "results/predictions/condition_4.json",
        "condition_5": "results/predictions/condition_5.json",
        "condition_6": "results/predictions/condition_6.json"
      }
  results/phase3_soup_results.json (optional; Test 3 coefficient magnitudes only)
"""

from __future__ import annotations

import argparse
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
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h2_predictions.json"
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

PLANNED_CONTRASTS = [
    ("condition_2", "condition_3"),
    ("condition_2", "condition_4"),
    ("condition_2", "condition_5"),
    ("condition_2", "condition_6"),
    ("condition_3", "condition_4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H2/RQ2 (final numbering): strategy equivalence.")
    parser.add_argument("--ground-truth", type=pathlib.Path, default=GROUND_TRUTH_FILE)
    parser.add_argument("--bootstrap-image-ids", type=pathlib.Path, default=BOOTSTRAP_IMAGE_IDS_FILE)
    parser.add_argument("--predictions-config", type=pathlib.Path, default=PREDICTIONS_CONFIG_FILE)
    parser.add_argument("--coefficients-file", type=pathlib.Path, default=SOUP_FILE_FOR_COEFFICIENTS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--practical-threshold", type=float, default=PRACTICAL_THRESHOLD)
    return parser.parse_args()


def run_test1_pairwise_bootstrap(predictions_config, ground_truth, ground_truth_path, bootstrap_image_ids, workers):
    results = []
    for key_a, key_b in PLANNED_CONTRASTS:
        result = run_comparison(
            name=f"H2-Test1 (RQ2, final numbering): {CONDITION_LABELS[key_a]} vs {CONDITION_LABELS[key_b]}",
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


def run_test3_coefficient_magnitude_analysis(coefficients_file: pathlib.Path):
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
    strategy_d = np.array([coef3[c] for c in components])
    strategy_f = np.array([coef4[c] for c in components])

    if strategy_d.shape[1] < 2 or strategy_f.shape[1] < 2:
        print("NOTE: fewer than 2 ingredient models available; Test 3 reported descriptively only.")
        return {
            "note": "insufficient replicates for inferential test",
            "means": {c: {"dirichlet": float(np.mean(coef3[c])), "fisher": float(np.mean(coef4[c]))} for c in components},
        }

    strategy_d_mean_per_model = strategy_d.mean(axis=0)
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
    print("TEST 1 -- Planned pairwise image-level bootstrap contrasts (Conditions 2-6, excl. M6 vs M5)")
    print("=" * 70)

    test1_results = run_test1_pairwise_bootstrap(predictions_config, ground_truth, args.ground_truth, bootstrap_image_ids, args.workers)
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

    decision_h2 = (
        "REJECT H02 (at least one strategy differs practically and statistically; Ha2 not fully supported)"
        if any_significant_contrast
        else "FAIL TO REJECT H02 (strategy-level differences are small / not significant; consistent with Ha2)"
    )
    print(f"\nOverall Test 1 decision: {decision_h2} (max |ΔAP| observed = {max_abs_diff:.4f} pp)")

    print("\n" + "=" * 70)
    print("TEST 3 -- Coefficient magnitude analysis (N=6 ingredient models, legitimate unit)")
    print("=" * 70)
    test3_result = run_test3_coefficient_magnitude_analysis(args.coefficients_file)

    output_dir = RESULTS_DIR / "h2_rq2_bootstrap"
    for index, result in enumerate(test1_results, start=1):
        save_result_artifacts(result, output_dir, file_stem=f"contrast_{index}")

    output = {
        "family": "RQ2 / H2 (final numbering)",
        "ha2_verbatim": "No coefficient strategy within this family outperforms the others by >= 0.5pp; observed differences are small and not statistically significant.",
        "test1_pairwise_contrasts": test1_results,
        "test1_overall_decision": decision_h2,
        "test3_coefficient_magnitude": test3_result,
        "multiple_testing": {"method": "Holm-Bonferroni", "alpha": args.alpha, "n_comparisons": len(test1_results)},
    }

    out_path = RESULTS_DIR / "h2_rq2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
