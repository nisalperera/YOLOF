"""
RQ1 / H1 (FINAL NUMBERING, Ch.1 Sec.1.5 / Ch.4 Preliminary Note) -- Static
learning effect.

Ha1 (verbatim from the final thesis): "Learned merging strategies produce
statistically and practically higher mAP than global uniform soup and the
best single fine-tuned YOLOF-R50 model."

This supersedes h1_rq1_component_vs_global.py (kept in the repo for now,
not deleted -- recommend removing it once this file is verified). The
previous version ran three comparisons (A: C1 vs C2, B: C2 vs best-learned,
C: best-ingredient vs best-learned soup). Comparison B tested a strategy
question that, in the final renumbering, belongs to H2/RQ2
(strategy equivalence), not H1/RQ1. This file is trimmed to exactly the
two comparisons Ha1 specifies.

STATISTICAL METHOD
-------------------
Paired image-level bootstrap (see bootstrap_core.py). The resampling unit
is one held-out COCO evaluation image, not a category or per-class AP
value -- see STATISTICS_UPDATE_NOTES.md for the rationale. Chapter 3
Sec.3.5 still describes a one-sample/paired t-test design for this test;
that description is stale and should be rewritten to describe this
bootstrap procedure (flagged in STATISTICS_UPDATE_NOTES.md).

Comparisons:
  H1-A : Condition 1 (global uniform) vs Condition 6 (M6)
  H1-B : best individual ingredient vs Condition 6 (M6)

Decision rule (Ch.1 Sec.1.5 / Ch.3 Sec.3.5, practical threshold 0.5pp):
  H01 rejected for a comparison if, after Holm-Bonferroni adjustment across
  the two comparisons:
    - Holm-adjusted one-sided p-value < 0.05, AND
    - 95% bootstrap CI lower bound > 0, AND
    - observed AP50:95 gain >= 0.5 pp.

SELECTION CAVEAT
-----------------
If 'best_ingredient' is chosen by maximising score on these SAME held-out
predictions, H1-B is exploratory (post-selection inference), not
confirmatory. Prefer selecting the best ingredient on the 953-image
calibration split.

Input files
-----------
  results/ground_truth_heldout.json
  results/bootstrap_manifest/bootstrap_image_ids.npy
  configs/h1_predictions.json
      {
        "condition_1": "results/predictions/condition_1.json",
        "condition_6": "results/predictions/condition_6.json",
        "best_ingredient": "results/predictions/L1.json"
      }
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime

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
PREDICTIONS_CONFIG_FILE = pathlib.Path("configs") / "h1_predictions.json"

ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.5
WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H1/RQ1 (final numbering): static learning effect.")
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
    required_keys = {"condition_1", "condition_6", "best_ingredient"}
    missing = required_keys - set(predictions_config)
    if missing:
        raise ValueError(f"{args.predictions_config} is missing keys: {sorted(missing)}")

    print(
        "NOTE: if 'best_ingredient' was selected by maximising score on these "
        "same held-out predictions, H1-B is exploratory (post-selection "
        "inference), not confirmatory. Prefer selecting it on the "
        "953-image calibration split instead."
    )

    result_a = run_comparison(
        name="H1-A (RQ1, final numbering): Condition 1 (global uniform) vs M6",
        model_a_name="Condition 1",
        model_a_predictions_path=predictions_config["condition_1"],
        model_b_name="M6",
        model_b_predictions_path=predictions_config["condition_6"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    result_b = run_comparison(
        name="H1-B (RQ1, final numbering): best individual ingredient vs M6",
        model_a_name="best_ingredient",
        model_a_predictions_path=predictions_config["best_ingredient"],
        model_b_name="M6",
        model_b_predictions_path=predictions_config["condition_6"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    results = [result_a, result_b]
    raw_p_values = [r["bootstrap"]["p_one_sided_model_b_greater"] for r in results]
    holm_p_values = holm_adjust(raw_p_values)

    for result, p_holm in zip(results, holm_p_values):
        apply_decision_rule(result, p_holm, alpha=args.alpha, practical_threshold_ap_points=args.practical_threshold)

    output_dir = RESULTS_DIR / "h1_rq1_bootstrap"
    for index, result in enumerate(results, start=1):
        save_result_artifacts(result, output_dir, file_stem=f"comparison_{index}")

    for result in results:
        print(f"\n{'='*70}")
        print(f"  {result['name']}")
        print(f"{'='*70}")
        print(f"  Observed AP50:95 A ({result['model_a']['name']}): {result['model_a']['observed_ap50_95']:.4f}")
        print(f"  Observed AP50:95 B ({result['model_b']['name']}): {result['model_b']['observed_ap50_95']:.4f}")
        print(f"  Observed ΔAP (B - A): {result['observed_difference_ap_points']:+.4f} pp")
        print(f"  95% bootstrap CI: [{result['bootstrap']['ci_95_percentile_lower']:+.4f}, {result['bootstrap']['ci_95_percentile_upper']:+.4f}]")
        print(f"  Holm-adjusted p: {result['bootstrap']['p_one_sided_holm_adjusted']:.6f}")
        print(f"  Decision: {'REJECT H01' if result['decision']['support_directional_superiority'] else 'FAIL TO REJECT H01'}")

    output = {
        "method": "Paired image-level bootstrap (image is the resampling unit)",
        "family": "RQ1 / H1 (final numbering)",
        "ha1_verbatim": "Learned merging strategies produce statistically and practically higher mAP than global uniform soup and the best single fine-tuned YOLOF-R50 model.",
        "comparisons": results,
        "multiple_testing": {"method": "Holm-Bonferroni", "alpha": args.alpha, "n_comparisons": len(results)},
    }

    out_path = RESULTS_DIR / f"h1_rq1_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    stable_path = RESULTS_DIR / "h1_rq1_results.json"
    with open(stable_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved -> {out_path}")
    print(f"Stable copy for orchestration -> {stable_path}")


if __name__ == "__main__":
    main()
