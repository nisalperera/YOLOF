"""
RQ4 / H4 (FINAL NUMBERING, Ch.1 Sec.1.5 / Ch.4 Preliminary Note) -- Full
pipeline superiority.

This rewrites h4c_finetune_gain.py. In the final draft, RQ4/H4 is
specifically: "Does the full merging + fine-tuning pipeline (M6 static ->
C3 decoder fine-tune) achieve higher mAP50:95 than both the best single
ingredient model and the published YOLOF-ResNet-50 baseline, with a
practically meaningful margin?" Ha4: "C3 exceeds both the best single
fine-tuned model and the YOLOF baseline by >= 0.5pp at headline and
per-class levels." h4c_finetune_gain.py is left in the repo unchanged for
now; recommend deleting it once this file is verified.

CONFIRMATORY TEST (this is the formal H4/RQ4 decision):
  C3 vs best individual ingredient model -- paired image-level bootstrap.
  Decision: reject H04 if one-sided p < 0.05, 95% CI lower bound > 0, and
  observed Δ >= 0.5 pp.

DESCRIPTIVE-ONLY evidence (per-class levels + published baseline; NOT part
of the formal decision, consistent with treating COCO categories as
non-independent and the published baseline as an unmatched external
scalar -- see bootstrap_core.py and STATISTICS_UPDATE_NOTES.md):
  - C3 vs published YOLOF-R50 baseline (37.7 AP50:95, Chen et al., 2021).
  - Per-class AP delta table, C3 minus best individual ingredient.
  - Supporting fine-tuning-gain evidence: Condition2->D1, best(3-5)->D2,
    M6->C3 (each a valid paired image-level bootstrap comparison in its
    own right, but not the formal H4 decision).
  - D2 vs C3 (which merge-quality produces the larger post-finetune gain).

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h4_predictions.json
      {
        "condition_2": "results/predictions/condition_2.json",
        "condition_6": "results/predictions/condition_6.json",
        "best_condition_3to5": "results/predictions/condition_3.json",
        "best_ingredient": "results/predictions/L1.json",
        "D1": "results/predictions/D1.json",
        "D2": "results/predictions/D2.json",
        "C3": "results/predictions/C3.json"
      }
  results/phase3_soup_results.json (optional; descriptive per-class table only)
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
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h4_predictions.json"
BASELINE_MAP = 37.7  # Chen et al., 2021, published YOLOF-R50 COCO val2017 AP50:95

ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.5
WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H4/RQ4 (final numbering): full pipeline superiority.")
    parser.add_argument("--ground-truth", type=pathlib.Path, default=GROUND_TRUTH_FILE)
    parser.add_argument("--bootstrap-image-ids", type=pathlib.Path, default=BOOTSTRAP_IMAGE_IDS_FILE)
    parser.add_argument("--predictions-config", type=pathlib.Path, default=PREDICTIONS_CONFIG_FILE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--practical-threshold", type=float, default=PRACTICAL_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ground_truth, bootstrap_image_ids = load_bootstrap_manifest_inputs(
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids_path=args.bootstrap_image_ids,
    )

    predictions_config = load_json(args.predictions_config)
    required_keys = {"condition_2", "condition_6", "best_condition_3to5", "best_ingredient", "D1", "D2", "C3"}
    missing = required_keys - set(predictions_config)
    if missing:
        raise ValueError(f"{args.predictions_config} is missing keys: {sorted(missing)}")

    print("=" * 70)
    print("CONFIRMATORY H4/RQ4 TEST -- C3 vs best individual ingredient (headline)")
    print("=" * 70)

    confirmatory_result = run_comparison(
        name="H4/RQ4 (final numbering): best individual ingredient vs C3",
        model_a_name="best_ingredient",
        model_a_predictions_path=predictions_config["best_ingredient"],
        model_b_name="C3",
        model_b_predictions_path=predictions_config["C3"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )
    apply_decision_rule(confirmatory_result, p_holm_adjusted=confirmatory_result["bootstrap"]["p_one_sided_model_b_greater"], alpha=args.alpha, practical_threshold_ap_points=args.practical_threshold)

    print(f"\n  ΔAP (C3 - best_ingredient) = {confirmatory_result['observed_difference_ap_points']:+.4f} pp")
    print(f"  95% CI = [{confirmatory_result['bootstrap']['ci_95_percentile_lower']:+.4f}, {confirmatory_result['bootstrap']['ci_95_percentile_upper']:+.4f}]")
    print(f"  p = {confirmatory_result['bootstrap']['p_one_sided_model_b_greater']:.6f}")

    decision_h4 = "REJECT H04" if confirmatory_result["decision"]["support_directional_superiority"] else "FAIL TO REJECT H04"
    print(f"  Decision (headline, confirmatory): {decision_h4}")

    print("\n" + "=" * 70)
    print("DESCRIPTIVE -- C3 vs published YOLOF-R50 baseline (not part of the formal decision)")
    print("=" * 70)

    c3_observed_ap = evaluate_map50_95(args.ground_truth, predictions_config["C3"])
    baseline_delta = c3_observed_ap - BASELINE_MAP
    print(f"  C3 observed AP50:95: {c3_observed_ap:.4f}")
    print(f"  Published baseline: {BASELINE_MAP:.1f}")
    print(f"  Descriptive Δ (C3 - baseline): {baseline_delta:+.4f} pp")
    print(
        "  NOTE: descriptive only. The published baseline is a single external "
        "scalar, not a matched prediction set on this held-out subset, so no "
        "bootstrap CI/p-value is computed for it."
    )

    print("\n" + "=" * 70)
    print("SUPPORTING DESCRIPTIVE EVIDENCE -- fine-tuning gain pairs and D2 vs C3")
    print("=" * 70)

    pair_definitions = [
        ("Condition 2 -> D1", "condition_2", "D1"),
        ("best_condition_3to5 -> D2", "best_condition_3to5", "D2"),
        ("Condition 6 (M6) -> C3", "condition_6", "C3"),
    ]

    finetune_gain_results = []
    for name, pre_key, post_key in pair_definitions:
        result = run_comparison(
            name=f"H4-supporting: {name}",
            model_a_name=pre_key,
            model_a_predictions_path=predictions_config[pre_key],
            model_b_name=post_key,
            model_b_predictions_path=predictions_config[post_key],
            ground_truth=ground_truth,
            ground_truth_path=args.ground_truth,
            bootstrap_image_ids=bootstrap_image_ids,
            workers=args.workers,
        )
        finetune_gain_results.append(result)

    supporting_p_values = [r["bootstrap"]["p_one_sided_model_b_greater"] for r in finetune_gain_results]
    supporting_holm = holm_adjust(supporting_p_values)
    for result, p_holm in zip(finetune_gain_results, supporting_holm):
        apply_decision_rule(result, p_holm, alpha=args.alpha, practical_threshold_ap_points=0.0)
        print(f"\n  {result['name']}")
        print(f"    Gain = {result['observed_difference_ap_points']:+.4f} pp, 95% CI [{result['bootstrap']['ci_95_percentile_lower']:+.4f}, {result['bootstrap']['ci_95_percentile_upper']:+.4f}]")

    d2_vs_c3 = run_comparison(
        name="H4-supporting: D2 vs C3",
        model_a_name="D2",
        model_a_predictions_path=predictions_config["D2"],
        model_b_name="C3",
        model_b_predictions_path=predictions_config["C3"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )
    apply_decision_rule(d2_vs_c3, p_holm_adjusted=d2_vs_c3["bootstrap"]["p_one_sided_model_b_greater"], alpha=args.alpha, practical_threshold_ap_points=0.5)
    print(f"\n  D2 vs C3: Δ = {d2_vs_c3['observed_difference_ap_points']:+.4f} pp, 95% CI [{d2_vs_c3['bootstrap']['ci_95_percentile_lower']:+.4f}, {d2_vs_c3['bootstrap']['ci_95_percentile_upper']:+.4f}]")

    print("\n" + "=" * 70)
    print("DESCRIPTIVE -- per-class AP delta table (C3 minus best individual ingredient)")
    print("=" * 70)
    descriptive_per_class = None
    soup_file = RESULTS_DIR / "phase3_soup_results.json"
    if soup_file.is_file():
        try:
            soup_data = load_json(soup_file)
            c3_entry = soup_data.get("C3") or soup_data.get("c3")
            best_entry = soup_data.get("best_ingredient")
            if c3_entry and best_entry and "per_class_ap" in c3_entry and "per_class_ap" in best_entry:
                c3_by_class = {row[0]: row[1] for row in c3_entry["per_class_ap"]}
                best_by_class = {row[0]: row[1] for row in best_entry["per_class_ap"]}
                descriptive_per_class = {
                    name: c3_by_class[name] - best_by_class[name]
                    for name in c3_by_class
                    if name in best_by_class
                }
                print("  Per-class deltas computed (reported descriptively, not as independent test units).")
            else:
                print("  Required per_class_ap fields not found; skipping descriptive per-class table.")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  Could not build descriptive per-class table: {exc}")
    else:
        print(f"  {soup_file} not found; skipping descriptive per-class table.")

    output_dir = RESULTS_DIR / "h4_rq4_bootstrap"
    save_result_artifacts(confirmatory_result, output_dir, file_stem="confirmatory_c3_vs_best_ingredient")
    for index, result in enumerate(finetune_gain_results, start=1):
        save_result_artifacts(result, output_dir, file_stem=f"supporting_finetune_pair_{index}")
    save_result_artifacts(d2_vs_c3, output_dir, file_stem="supporting_d2_vs_c3")

    output = {
        "family": "RQ4 / H4 (final numbering)",
        "ha4_verbatim": "C3 exceeds both the best single fine-tuned model and the YOLOF baseline by >= 0.5pp at headline and per-class levels, demonstrating pipeline-level superiority.",
        "confirmatory_test_c3_vs_best_ingredient": confirmatory_result,
        "confirmatory_decision": decision_h4,
        "descriptive_c3_vs_published_baseline": {
            "c3_observed_ap50_95": c3_observed_ap,
            "published_baseline_ap50_95": BASELINE_MAP,
            "delta_ap_points": baseline_delta,
            "note": "Descriptive only; no matched prediction set for the published baseline.",
        },
        "descriptive_per_class_delta_c3_minus_best_ingredient": descriptive_per_class,
        "supporting_finetune_gain_pairs": finetune_gain_results,
        "supporting_d2_vs_c3": d2_vs_c3,
        "multiple_testing_confirmatory": {"method": "none (single confirmatory contrast)", "alpha": args.alpha},
        "multiple_testing_supporting": {"method": "Holm-Bonferroni", "alpha": args.alpha, "n_comparisons": len(finetune_gain_results)},
    }

    out_path = RESULTS_DIR / "h4_rq4_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
