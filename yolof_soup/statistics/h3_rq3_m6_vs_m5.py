"""
RQ3 / H3 (FINAL NUMBERING, Ch.1 Sec.1.5 / Ch.4 Preliminary Note) -- M6 vs M5.

This is the renamed successor to h4a_paired_ttest_M6_vs_M5.py. Content is
unchanged from that file; only the RQ/H label is corrected to match the
final thesis numbering -- Ch.4's Preliminary Note states "M6 versus M5 is
addressed in H3 and RQ3" [Ch.4]. h4a_paired_ttest_M6_vs_M5.py is left in
the repo unchanged for now; recommend deleting it once this file is
verified.

Ha3 (verbatim from the final thesis): "M6 yields statistically and
practically higher mAP than M5, demonstrating the combined synergistic
effect of decoupling both the mixing coefficients and temperature scalar."

STATISTICAL METHOD
-------------------
Paired image-level bootstrap. Chapter 3 Sec.3.5 describes this as a
"directional paired t-test" over per-class AP values (mislabeled there as
"RQ4a/H4a"); that per-class design is invalid for the reasons documented
in bootstrap_core.py.

Decision rule (pre-specified): reject H03 if
  - one-sided bootstrap p-value < 0.05, AND
  - 95% bootstrap CI lower bound > 0.

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h3_predictions.json
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
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h3_predictions.json"

ALPHA = 0.05
WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H3/RQ3 (final numbering): M6 vs M5.")
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
        name="H3/RQ3 (final numbering): M6 (tri-component learned) vs M5 (shared learned)",
        model_a_name="M5",
        model_a_predictions_path=predictions_config["condition_5"],
        model_b_name="M6",
        model_b_predictions_path=predictions_config["condition_6"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    apply_decision_rule(result, p_holm_adjusted=result["bootstrap"]["p_one_sided_model_b_greater"], alpha=args.alpha, practical_threshold_ap_points=0.0)

    print("\n" + "=" * 70)
    print("H3/RQ3 (final numbering) -- Paired image-level bootstrap: M6 vs M5")
    print("=" * 70)
    print(f"  Observed AP50:95 M5: {result['model_a']['observed_ap50_95']:.4f}")
    print(f"  Observed AP50:95 M6: {result['model_b']['observed_ap50_95']:.4f}")
    print(f"  Mean Δ (M6 - M5): {result['observed_difference_ap_points']:+.4f} pp")
    print(f"  95% bootstrap CI: [{result['bootstrap']['ci_95_percentile_lower']:+.4f}, {result['bootstrap']['ci_95_percentile_upper']:+.4f}]")
    print(f"  One-sided p: {result['bootstrap']['p_one_sided_model_b_greater']:.6f}")

    decision = (
        "REJECT H03"
        if result["bootstrap"]["p_one_sided_model_b_greater"] < args.alpha
        and result["bootstrap"]["ci_95_percentile_lower"] > 0.0
        else "FAIL TO REJECT H03"
    )
    print(f"  Decision: {decision}")

    output_dir = RESULTS_DIR / "h3_rq3_bootstrap"
    save_result_artifacts(result, output_dir, file_stem="m6_vs_m5")

    out_path = RESULTS_DIR / "h3_rq3_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "family": "RQ3 / H3 (final numbering)",
                "ha3_verbatim": "M6 yields statistically and practically higher mAP than M5, demonstrating the combined synergistic effect of decoupling both the mixing coefficients and temperature scalar.",
                "comparison": result,
                "decision_label": decision,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
