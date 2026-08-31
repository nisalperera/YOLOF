"""
RQ4a / H4a (granularity contrast) -- M6 (tri-component learned α+β,
            three independent pairs) vs M5 (shared α+β pair).

STATISTICAL METHOD (updated)
-----------------------------
The previous version ran a paired t-test, Cohen's d, a class-resampled
10,000-replicate bootstrap, and a Wilcoxon test over 80 per-class AP
values. That treats COCO categories as independent observations, which
is not valid for the reasons documented in bootstrap_core.py and in the
other scripts in this package.

This version runs a single PAIRED IMAGE-LEVEL BOOTSTRAP comparison of M6
vs M5 on the held-out COCO evaluation images, reusing each model's already
computed COCO-format predictions and the shared bootstrap draws generated
by generate_coco_bootstrap_manifest.py.

Decision rule (pre-specified): reject H04a if
  - one-sided bootstrap p-value < 0.05, AND
  - 95% bootstrap CI lower bound > 0.

(No separate multiple-testing correction is needed here because this is a
single confirmatory contrast; if you also run this test as part of a
larger family alongside H1/H3/H4c in run_all_stats.py, apply Holm
correction across the FULL set of confirmatory contrasts there instead of
here, to avoid double correction.)

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h4a_predictions.json
      {
        "condition_5": "results/predictions/condition_5.json",
        "condition_6": "results/predictions/condition_6.json"
      }
"""

from __future__ import annotations

import argparse
import json
import pathlib

from yolof_soup.statistics.bootstrap_core import (
    apply_decision_rule,
    load_bootstrap_manifest_inputs,
    load_json,
    run_comparison,
    save_result_artifacts,
)

RESULTS_DIR = pathlib.Path("results")
GROUND_TRUTH_FILE = RESULTS_DIR / "ground_truth_heldout.json"
BOOTSTRAP_IMAGE_IDS_FILE = RESULTS_DIR / "bootstrap_manifest" / "bootstrap_image_ids.npy"
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h4a_predictions.json"

ALPHA = 0.05
WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H4a: M6 vs M5 paired image-level bootstrap.")
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
    for key in ("condition_5", "condition_6"):
        if key not in predictions_config:
            raise ValueError(f"{args.predictions_config} is missing key: '{key}'")

    result = run_comparison(
        name="H4a: M6 (tri-component learned) vs M5 (shared learned)",
        model_a_name="M5",
        model_a_predictions_path=predictions_config["condition_5"],
        model_b_name="M6",
        model_b_predictions_path=predictions_config["condition_6"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    # Single confirmatory contrast: no Holm correction needed on its own,
    # but note in run_all_stats.py if this is folded into a larger family.
    apply_decision_rule(result, p_holm_adjusted=result["bootstrap"]["p_one_sided_model_b_greater"], alpha=args.alpha, practical_threshold_ap_points=0.0)

    print("\n" + "=" * 70)
    print("H4a -- Paired image-level bootstrap: M6 vs M5")
    print("=" * 70)
    print(f"  Observed AP50:95 M5: {result['model_a']['observed_ap50_95']:.4f}")
    print(f"  Observed AP50:95 M6: {result['model_b']['observed_ap50_95']:.4f}")
    print(f"  Mean Δ (M6 - M5): {result['observed_difference_ap_points']:+.4f} pp")
    print(f"  95% bootstrap CI: [{result['bootstrap']['ci_95_percentile_lower']:+.4f}, {result['bootstrap']['ci_95_percentile_upper']:+.4f}]")
    print(f"  One-sided p: {result['bootstrap']['p_one_sided_model_b_greater']:.6f}")

    decision = (
        "REJECT H04a"
        if result["bootstrap"]["p_one_sided_model_b_greater"] < args.alpha
        and result["bootstrap"]["ci_95_percentile_lower"] > 0.0
        else "FAIL TO REJECT H04a"
    )
    print(f"  Decision: {decision}")

    output_dir = RESULTS_DIR / "h4a_bootstrap"
    save_result_artifacts(result, output_dir, file_stem="m6_vs_m5")

    out_path = RESULTS_DIR / "h4a_results.json"
    with open(out_path, "w") as f:
        json.dump({"family": "RQ4a / H4a", "comparison": result, "decision_label": decision}, f, indent=2, default=float)
    print(f"  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
