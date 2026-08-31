"""
RQ1 / H1 -- Component-specific decoder averaging vs. global uniform soup
            and vs. best individual ingredient model.

STATISTICAL METHOD (updated)
-----------------------------
Previous versions of this script ran a paired t-test, Wilcoxon signed-rank
test, and a class-resampled bootstrap over 80 (or 70) per-class AP values.
That treats each COCO category as an independent observation, which is
incorrect: category APs are correlated components of one aggregate mAP
estimate, and objects within the same image share detector failure modes.

This version instead runs a PAIRED IMAGE-LEVEL BOOTSTRAP over the held-out
COCO evaluation images, reusing each model's already-computed COCO-format
predictions. See yolof_soup/statistics/bootstrap_core.py for the resampling
engine and generate_coco_bootstrap_manifest.py for the shared bootstrap
draws (must be generated once, with a fixed seed, and reused for every
H1/H3/H4a/H4c comparison in this thesis).

Comparisons (Section 3.5.1):
  Comparison A : Condition 1 (global uniform) vs Condition 2 (component uniform)
                 -> isolates IV1 (partition structure)
  Comparison B : Condition 2 vs best of {Condition 3, 4, 5, 6}
                 -> isolates IV2 (coefficient learning strategy)
  Comparison C : best individual ingredient vs best learned soup
                 -> practical value; NOTE selection caveat below

Decision rule (pre-specified in Ch. 3, Section 3.5.1):
  H01 rejected for a comparison if, after Holm-Bonferroni adjustment across
  the three comparisons:
    - the Holm-adjusted one-sided p-value < 0.05, AND
    - the 95% bootstrap CI lower bound > 0, AND
    - the observed AP50:95 gain >= 0.5 pp.

SELECTION CAVEAT (important)
-----------------------------
If "best learned condition" / "best individual ingredient" are selected by
maximising score on the SAME held-out set used for this test, Comparisons B
and C are exploratory (post-selection inference), not confirmatory. Prefer
selecting the best learned condition / best ingredient on the 953-image
calibration split, and only evaluate the fixed, named winner on the
held-out set. This script prints a warning when it detects that selection
was performed on the held-out predictions supplied to it.

Input files
-----------
  results/ground_truth_heldout.json
      COCO-format ground truth for the ~4,047-image held-out val2017 subset.
  results/bootstrap_manifest/bootstrap_image_ids.npy
      Generated once via generate_coco_bootstrap_manifest.py, seed=42.
  configs/h1_predictions.json
      Maps condition/model names to their saved COCO-format prediction
      JSON files (already run; this script does not perform inference).
      Example:
        {
          "condition_1": "results/predictions/condition_1.json",
          "condition_2": "results/predictions/condition_2.json",
          "condition_3": "results/predictions/condition_3.json",
          "condition_4": "results/predictions/condition_4.json",
          "condition_5": "results/predictions/condition_5.json",
          "condition_6": "results/predictions/condition_6.json",
          "best_ingredient": "results/predictions/L1.json"
        }
      "best_ingredient" should be preselected on the calibration split, not
      derived from these same predictions, whenever possible.

Per-class AP/AR tables (if present in results/phase3_soup_results.json)
are still loaded and written to the output JSON, but purely as descriptive
diagnostics -- they are not used for any p-value or confidence interval in
this script.
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
SOUP_DESCRIPTIVE_FILE = RESULTS_DIR / "phase3_soup_results.json"  # optional, descriptive only

ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.5
WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ1/H1 paired image-level bootstrap analysis.")
    parser.add_argument("--ground-truth", type=pathlib.Path, default=GROUND_TRUTH_FILE)
    parser.add_argument("--bootstrap-image-ids", type=pathlib.Path, default=BOOTSTRAP_IMAGE_IDS_FILE)
    parser.add_argument("--predictions-config", type=pathlib.Path, default=PREDICTIONS_CONFIG_FILE)
    parser.add_argument("--descriptive-soup-file", type=pathlib.Path, default=SOUP_DESCRIPTIVE_FILE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--practical-threshold", type=float, default=PRACTICAL_THRESHOLD)
    return parser.parse_args()


def load_descriptive_per_class_ap(soup_file: pathlib.Path, key: str) -> dict | None:
    """Best-effort descriptive-only per-class AP/AR extraction. Never used for inference."""
    if not soup_file.is_file():
        return None
    try:
        data = load_json(soup_file)
        entry = data.get(key)
        if entry is None or "per_class_ap" not in entry:
            return None
        return {"per_class_ap": entry["per_class_ap"], "map50_95": entry.get("map50_95")}
    except (json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    args = parse_args()

    ground_truth, bootstrap_image_ids = load_bootstrap_manifest_inputs(
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids_path=args.bootstrap_image_ids,
    )

    predictions_config = load_json(args.predictions_config)
    required_keys = {"condition_1", "condition_2", "condition_3", "condition_4", "condition_5", "condition_6", "best_ingredient"}
    missing = required_keys - set(predictions_config)
    if missing:
        raise ValueError(f"{args.predictions_config} is missing keys: {sorted(missing)}")

    # Comparison A: Condition 1 vs Condition 2 (isolates IV1)
    result_a = run_comparison(
        name="H1-A: Condition 1 (global uniform) vs Condition 2 (component uniform)",
        model_a_name="Condition 1",
        model_a_predictions_path=predictions_config["condition_1"],
        model_b_name="Condition 2",
        model_b_predictions_path=predictions_config["condition_2"],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    # Comparison B: Condition 2 vs best of {3,4,5,6} -- selection caveat applies
    learned_condition_keys = ["condition_3", "condition_4", "condition_5", "condition_6"]
    best_learned_key = None
    best_learned_score = float("-inf")
    for key in learned_condition_keys:
        # Full-set AP is recomputed by run_comparison; here we just need a
        # provisional ranking, so we reuse the same evaluate step lazily by
        # running a throwaway 1-vs-1 against condition_2 and reading model_b AP.
        probe = run_comparison(
            name=f"probe:{key}",
            model_a_name="Condition 2",
            model_a_predictions_path=predictions_config["condition_2"],
            model_b_name=key,
            model_b_predictions_path=predictions_config[key],
            ground_truth=ground_truth,
            ground_truth_path=args.ground_truth,
            bootstrap_image_ids=bootstrap_image_ids[:1],  # cheap probe, 1 replicate
            workers=1,
        )
        score = probe["model_b"]["observed_ap50_95"]
        if score > best_learned_score:
            best_learned_score = score
            best_learned_key = key

    print(
        f"NOTE: best_learned_key='{best_learned_key}' was selected by maximising "
        "observed AP50:95 on the SAME held-out predictions used for testing. "
        "Comparison B is exploratory unless this selection was made on the "
        "953-image calibration split instead."
    )

    result_b = run_comparison(
        name=f"H1-B: Condition 2 vs {best_learned_key} (best learned strategy)",
        model_a_name="Condition 2",
        model_a_predictions_path=predictions_config["condition_2"],
        model_b_name=best_learned_key,
        model_b_predictions_path=predictions_config[best_learned_key],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    # Comparison C: best individual ingredient vs best learned soup
    print(
        f"NOTE: 'best_ingredient'='{predictions_config['best_ingredient']}' should be "
        "preselected on the calibration split. If it was chosen from the held-out "
        "predictions supplied here, treat Comparison C as exploratory as well."
    )

    result_c = run_comparison(
        name=f"H1-C: best_ingredient vs {best_learned_key} (best learned soup)",
        model_a_name="best_ingredient",
        model_a_predictions_path=predictions_config["best_ingredient"],
        model_b_name=best_learned_key,
        model_b_predictions_path=predictions_config[best_learned_key],
        ground_truth=ground_truth,
        ground_truth_path=args.ground_truth,
        bootstrap_image_ids=bootstrap_image_ids,
        workers=args.workers,
    )

    results = [result_a, result_b, result_c]
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
        print(f"  One-sided p: {result['bootstrap']['p_one_sided_model_b_greater']:.6f}")
        print(f"  Holm-adjusted p: {result['bootstrap']['p_one_sided_holm_adjusted']:.6f}")
        print(f"  Decision: {'REJECT H01' if result['decision']['support_directional_superiority'] else 'FAIL TO REJECT H01'}")

    descriptive = {
        "condition_1": load_descriptive_per_class_ap(args.descriptive_soup_file, "condition_1"),
        "condition_2": load_descriptive_per_class_ap(args.descriptive_soup_file, "condition_2"),
        best_learned_key: load_descriptive_per_class_ap(args.descriptive_soup_file, best_learned_key),
    }

    output = {
        "method": "Paired image-level bootstrap (image is the resampling unit)",
        "family": "RQ1 / H1",
        "comparisons": results,
        "descriptive_per_class_ap_ar": descriptive,
        "multiple_testing": {"method": "Holm-Bonferroni", "alpha": args.alpha, "n_comparisons": len(results)},
    }

    out_path = RESULTS_DIR / f"h1_rq1_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
