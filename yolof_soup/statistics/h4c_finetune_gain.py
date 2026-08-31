"""
RQ4c / H4c -- Decoder-only post-merge fine-tuning gain; merge quality vs
              fine-tuning gain (D1 vs D2 vs C3); full pipeline superiority.

STATISTICAL METHOD (updated)
-----------------------------
The previous version ran paired t-tests on 80 per-class AP gain arrays
(Test 4), then treated those same per-class gain arrays as if they were
independent samples for a Kruskal-Wallis test and Mann-Whitney U
comparisons across pairs (Test 5). Both steps treat COCO categories as
independent observations, and Test 5 additionally treats three
NON-independent gain vectors (all derived from overlapping category sets)
as if they were independent groups.

This version:
  Test 4 (updated): paired image-level bootstrap for each fine-tuning pair
    Pair A: Condition 2 (pre-finetune) vs D1 (post-finetune)
    Pair B: best-of-{3,4,5} (pre-finetune) vs D2 (post-finetune)
    Pair C: Condition 6 / M6 (pre-finetune) vs C3 (post-finetune)
  Decision: H04c supported for a pair if 95% bootstrap CI lower bound > 0
  for the pre -> post gain.

  Test 5 (updated): rather than Kruskal-Wallis/Mann-Whitney over per-class
  gain arrays (invalid, non-independent groups), Test 5 now reports:
    - C3 vs D2 as its own paired image-level bootstrap comparison
      (both are fine-tuned outputs, both are held-out predictions, so this
      is a valid additional pairwise contrast).
    - C3 vs published YOLOF baseline (37.7 AP) reported DESCRIPTIVELY ONLY,
      because the baseline is a single external scalar, not a matched
      prediction set on this held-out subset; see note in output.

Decision rule (Test 4, pre-specified, revised):
  H04c supported for a pair if 95% bootstrap CI lower bound > 0 for the
  pre -> post finetuning gain (Holm-adjusted across the three pairs).

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h4c_predictions.json
      {
        "condition_2": "results/predictions/condition_2.json",
        "condition_6": "results/predictions/condition_6.json",
        "best_condition_3to5": "results/predictions/condition_3.json",
        "D1": "results/predictions/D1.json",
        "D2": "results/predictions/D2.json",
        "C3": "results/predictions/C3.json"
      }

Note: "best_condition_3to5" should be preselected on the calibration
split, consistent with the selection caveat in h1_rq1_component_vs_global.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from yolof_soup.statistics.bootstrap_core import (
    apply_decision_rule,
    evaluate_map50_95,
    holm_adjust,
    load_bootstrap_manifest_inputs,
    load_json,
    run_comparison,
    save_result_artifacts,
)

RESULTS_DIR = pathlib.Path("results")
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth_heldout.json"
BOOTSTRAP_IMAGE_IDS_FILE = RESULTS_DIR / "bootstrap_manifest" / "bootstrap_image_ids.npy"
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h4c_predictions.json"
BASELINE_MAP = 37.7  # Chen et al., 2021, published YOLOF-R50 COCO val2017 AP50:95

ALPHA = 0.05
WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H4c: fine-tuning gain, paired image-level bootstrap.")
    parser.add_argument("--ground-truth", type=pathlib.Path, default=GROUND_TRUTH_FILE)
    parser.add_argument("--bootstrap-image-ids", type=pathlib.Path, default=BOOTSTRAP_IMAGE_IDS_FILE)
    parser.add_argument("--predictions-config", type=pathlib.Path, default=PREDICTIONS_CONFIG_FILE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ground_truth, bootstrap_image_ids = load_bootstrap_manifest_inputs(
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids_path=args.bootstrap_image_ids,
    )

    predictions_config = load_json(args.predictions_config)
    required_keys = {"condition_2", "condition_6", "best_condition_3to5", "D1", "D2", "C3"}
    missing = required_keys - set(predictions_config)
    if missing:
        raise ValueError(f"{args.predictions_config} is missing keys: {sorted(missing)}")

    print("=" * 70)
    print("TEST 4 -- Post-merge fine-tuning gain (paired image-level bootstrap)")
    print("=" * 70)

    pair_definitions = [
        ("Pair A: Condition 2 -> D1", "condition_2", "D1"),
        ("Pair B: best_condition_3to5 -> D2", "best_condition_3to5", "D2"),
        ("Pair C: Condition 6 (M6) -> C3", "condition_6", "C3"),
    ]

    test4_results = []
    for name, pre_key, post_key in pair_definitions:
        result = run_comparison(
            name=f"H4c-Test4 {name}",
            model_a_name=pre_key,
            model_a_predictions_path=predictions_config[pre_key],
            model_b_name=post_key,
            model_b_predictions_path=predictions_config[post_key],
            ground_truth=ground_truth,
            ground_truth_path=args.ground_truth,
            bootstrap_image_ids=bootstrap_image_ids,
            workers=args.workers,
        )
        test4_results.append(result)

    test4_p_values = [r["bootstrap"]["p_one_sided_model_b_greater"] for r in test4_results]
    test4_holm = holm_adjust(test4_p_values)

    for result, p_holm in zip(test4_results, test4_holm):
        apply_decision_rule(result, p_holm, alpha=args.alpha, practical_threshold_ap_points=0.0)
        print(f"\n  {result['name']}")
        print(f"    Gain = {result['observed_difference_ap_points']:+.4f} pp")
        print(f"    95% CI = [{result['bootstrap']['ci_95_percentile_lower']:+.4f}, {result['bootstrap']['ci_95_percentile_upper']:+.4f}]")
        print(f"    Holm p = {result['bootstrap']['p_one_sided_holm_adjusted']:.6f}")
        supported = result["bootstrap"]["ci_95_percentile_lower"] > 0.0
        print(f"    Decision: {'H04c SUPPORTED for this pair' if supported else 'H04c NOT SUPPORTED for this pair'}")

    print("\n" + "=" * 70)
    print("TEST 5 -- C3 vs D2 (paired image-level bootstrap); C3 vs published baseline (descriptive)")
    print("=" * 70)

    c3_vs_d2 = run_comparison(
        name="H4c-Test5: D2 vs C3",
        model_a_name="D2",
        model_a_predictions_path=predictions_config["D2"],
        model_b_name="C3",
        model_b_predictions_path=predictions_config["C3"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )
    apply_decision_rule(c3_vs_d2, p_holm_adjusted=c3_vs_d2["bootstrap"]["p_one_sided_model_b_greater"], alpha=args.alpha, practical_threshold_ap_points=0.5)

    print(f"\n  D2 vs C3: Δ = {c3_vs_d2['observed_difference_ap_points']:+.4f} pp, 95% CI [{c3_vs_d2['bootstrap']['ci_95_percentile_lower']:+.4f}, {c3_vs_d2['bootstrap']['ci_95_percentile_upper']:+.4f}]")

    c3_observed_ap = evaluate_map50_95(args.ground_truth, predictions_config["C3"])
    baseline_delta = c3_observed_ap - BASELINE_MAP

    print(f"\n  C3 observed AP50:95 on held-out set: {c3_observed_ap:.4f}")
    print(f"  Published YOLOF-R50 baseline (Chen et al., 2021): {BASELINE_MAP:.1f}")
    print(f"  Descriptive Δ (C3 - baseline): {baseline_delta:+.4f} pp")
    print(
        "  NOTE: this Δ is DESCRIPTIVE ONLY. The published baseline is a single "
        "external scalar, not a matched prediction set on this held-out subset, "
        "so no bootstrap CI or p-value is computed for it. A rigorous comparison "
        "would require running the original public YOLOF-R50 checkpoint on "
        "exactly this held-out subset and saving its predictions."
    )

    output_dir = RESULTS_DIR / "h4c_bootstrap"
    for index, result in enumerate(test4_results, start=1):
        save_result_artifacts(result, output_dir, file_stem=f"test4_pair_{index}")
    save_result_artifacts(c3_vs_d2, output_dir, file_stem="test5_c3_vs_d2")

    output = {
        "family": "RQ4c / H4c",
        "test4_finetuning_gain": test4_results,
        "test5_c3_vs_d2": c3_vs_d2,
        "test5_c3_vs_published_baseline_DESCRIPTIVE_ONLY": {
            "c3_observed_ap50_95": c3_observed_ap,
            "published_baseline_ap50_95": BASELINE_MAP,
            "delta_ap_points": baseline_delta,
            "note": "Descriptive only; no matched prediction set for the published baseline.",
        },
        "multiple_testing": {"method": "Holm-Bonferroni", "alpha": args.alpha, "n_comparisons_test4": len(test4_results)},
    }

    out_path = RESULTS_DIR / "h4c_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
