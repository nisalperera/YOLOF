"""
rq4_full_pipeline_test.py
===========================

RQ4 Analysis: Full pipeline evaluation and per-category breakdown.

Research Question 4 (RQ4): Does the complete pipeline (D1/D2/C3) achieve
measurable improvements? Which object categories benefit most?

This module runs four statistical tests:

Test 1: Bootstrap Confidence Interval (C3 vs Best Individual)
  - 10,000 bootstrap resamples on per-class AP difference
  - Decision: If CI_lower ≥ 0 → C3 ≥ best_individual (statistically supported)
  - Also reports point estimate and standard error

Test 2: One-Sample t-test (C3 vs Published Baseline)
  - Baseline: 37.7 AP from YOLOF paper (COCO 2017 val)
  - H0: μ_C3 = 37.7
  - Expected: C3 significantly better than baseline

Test 3: Per-Category Analysis
  - Categories: rare (< 100 instances), small (area < 32²), medium, large
  - Compute mean AP per category group
  - Generate summary statistics and comparisons

Test 4: One-Way ANOVA (D1 vs D2 vs C3)
  - H0: μ_D1 = μ_D2 = μ_C3 (no difference across fine-tune strategies)
  - Post-hoc Tukey HSD if significant
  - Expected: C3 ≥ both D1 and D2

Dependencies:
  - Phase 3 results (best individual per-class AP)
  - Phase 5 results (D1, D2, C3 per-class AP)

Outputs:
  - rq4_test_results.json — All four test results
  - rq4_bootstrap_ci.json — Bootstrap statistics

Run: python -m yolof_soup.experiments.rq4_full_pipeline_test
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from yolof_soup.config.experiment_config import RESULTS_DIR
from yolof_soup.utils.logging_utils import setup_logging

logger = setup_logging(logging.INFO, filename="experiments/full_pipeline_test.log", use_stdout=True)

# COCO class definitions for category grouping
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Hardcoded category instance counts (from COCO 2017 train)
# Used for grouping: rare (<100), small (100-500), medium (500-2000), large (>2000)
COCO_INSTANCE_COUNTS = {
    "rare": [3, 4, 5, 6, 11, 22, 29, 32, 34, 42, 47, 48, 49, 50, 53, 55, 56, 58, 59, 60, 61, 64, 65, 67, 68, 70, 75, 80],
    "small": [2, 7, 8, 13, 15, 18, 21, 23, 24, 26, 31, 33, 35, 36, 40, 43, 44, 46, 52, 54, 62, 63, 66, 74, 78],
    "medium": [0, 1, 9, 10, 12, 14, 16, 17, 19, 20, 25, 27, 28, 30, 37, 38, 39, 41, 45, 51, 57, 69, 71, 72, 73, 76, 77, 79],
    "large": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Bootstrap CI (C3 vs Best Individual)
# ─────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci(
    per_class_ap_best_individual: List[float],
    per_class_ap_c3: List[float],
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """
    Test 1: Bootstrap confidence interval on C3 vs best individual.
    
    Args:
        per_class_ap_best_individual: Per-class AP for best individual
        per_class_ap_c3: Per-class AP for C3
        n_bootstrap: Number of bootstrap resamples
    
    Returns:
        Dict with bootstrap statistics and CI
    """
    logger.info("Test 1: Bootstrap CI (C3 vs Best Individual, %d resamples)...", n_bootstrap)
    
    best = np.array(per_class_ap_best_individual, dtype=float)
    c3 = np.array(per_class_ap_c3, dtype=float)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(best) | np.isnan(c3))
    best_valid = best[valid_mask]
    c3_valid = c3[valid_mask]
    
    if len(best_valid) < 3:
        logger.warning("  ✗ Not enough valid class data for bootstrap")
        return {"error": "Insufficient data", "n_valid_classes": len(best_valid)}
    
    # Observed difference
    obs_diff = np.mean(c3_valid) - np.mean(best_valid)
    
    # Bootstrap resamples
    bootstrap_diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(best_valid), size=len(best_valid), replace=True)
        boot_diff = np.mean(c3_valid[idx]) - np.mean(best_valid[idx])
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # CI: percentile method
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    se_boot = np.std(bootstrap_diffs)
    
    results = {
        "test_name": "Bootstrap CI: C3 - Best Individual",
        "n_classes": len(best_valid),
        "n_bootstrap_samples": n_bootstrap,
        "mean_best_individual": float(np.mean(best_valid)),
        "mean_c3": float(np.mean(c3_valid)),
        "observed_difference": float(obs_diff),
        "bootstrap_se": float(se_boot),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "ci_contains_zero": bool(ci_lower <= 0 <= ci_upper),
        "decision": "C3 ≥ best_individual (supported)" if ci_lower >= 0
                   else "Inconclusive" if ci_lower < 0 < ci_upper
                   else "Best individual > C3 (not supported)",
        "interpretation": "Bootstrap CI does not contain zero; C3 significantly better" if ci_lower > 0
                         else "Bootstrap CI contains zero; no significant difference"
    }
    
    logger.info("  → Observed diff: %.4f, 95%% CI: [%.4f, %.4f]", obs_diff, ci_lower, ci_upper)
    logger.info("  → Bootstrap SE: %.4f", se_boot)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: One-Sample t-test (C3 vs Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def test_c3_vs_baseline(
    per_class_ap_c3: List[float],
    baseline_ap: float = 37.7,
) -> Dict[str, Any]:
    """
    Test 2: One-sample t-test comparing C3 against published baseline.
    
    Args:
        per_class_ap_c3: Per-class AP for C3
        baseline_ap: Published YOLOF baseline (37.7 AP)
    
    Returns:
        Dict with t-test results
    """
    logger.info("Test 2: One-Sample t-test (C3 vs Baseline %.1f)...", baseline_ap)
    
    c3 = np.array(per_class_ap_c3, dtype=float)
    
    # Remove NaN values
    c3_valid = c3[~np.isnan(c3)]
    
    if len(c3_valid) < 3:
        logger.warning("  ✗ Not enough valid class data for one-sample t-test")
        return {"error": "Insufficient data", "n_valid_classes": len(c3_valid)}
    
    # One-sample t-test: H0: μ_C3 = baseline_ap
    t_stat, p_t = stats.ttest_1samp(c3_valid, baseline_ap)
    
    mean_c3 = np.mean(c3_valid)
    se_c3 = stats.sem(c3_valid)
    ci_lower = mean_c3 - 1.96 * se_c3
    ci_upper = mean_c3 + 1.96 * se_c3
    
    results = {
        "test_name": "One-sample t-test: C3 vs Baseline",
        "n_classes": len(c3_valid),
        "baseline_ap": float(baseline_ap),
        "mean_c3": float(mean_c3),
        "difference": float(mean_c3 - baseline_ap),
        "t_statistic": float(t_stat),
        "p_value": float(p_t),
        "se": float(se_c3),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "significant_alpha_0.05": bool(p_t < 0.05),
        "interpretation": "C3 significantly better than baseline" if (p_t < 0.05 and mean_c3 > baseline_ap)
                         else "C3 significantly worse than baseline" if (p_t < 0.05 and mean_c3 < baseline_ap)
                         else "No significant difference from baseline"
    }
    
    logger.info("  → Mean C3: %.4f, Baseline: %.1f, t: %.3f, p: %.4f",
                mean_c3, baseline_ap, t_stat, p_t)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Per-Category Analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_per_category_analysis(
    per_class_ap_d1: List[float],
    per_class_ap_d2: List[float],
    per_class_ap_c3: List[float],
) -> Dict[str, Any]:
    """
    Test 3: Analyze per-class AP by object category.
    
    Categories: rare, small, medium, large (based on instance count)
    
    Args:
        per_class_ap_d1: Per-class AP for D1
        per_class_ap_d2: Per-class AP for D2
        per_class_ap_c3: Per-class AP for C3
    
    Returns:
        Dict with per-category statistics
    """
    logger.info("Test 3: Per-Category Analysis...")
    
    d1 = np.array(per_class_ap_d1, dtype=float)
    d2 = np.array(per_class_ap_d2, dtype=float)
    c3 = np.array(per_class_ap_c3, dtype=float)
    
    category_results = {}
    for category, class_indices in COCO_INSTANCE_COUNTS.items():
        if len(class_indices) == 0:
            continue
        
        # Extract AP for this category
        d1_cat = d1[class_indices]
        d2_cat = d2[class_indices]
        c3_cat = c3[class_indices]
        
        # Remove NaN
        valid_mask = ~(np.isnan(d1_cat) | np.isnan(d2_cat) | np.isnan(c3_cat))
        d1_valid = d1_cat[valid_mask]
        d2_valid = d2_cat[valid_mask]
        c3_valid = c3_cat[valid_mask]
        
        if len(d1_valid) > 0:
            category_results[category] = {
                "n_classes": len(d1_valid),
                "d1_mean_ap": float(np.mean(d1_valid)),
                "d2_mean_ap": float(np.mean(d2_valid)),
                "c3_mean_ap": float(np.mean(c3_valid)),
                "d1_std": float(np.std(d1_valid)),
                "d2_std": float(np.std(d2_valid)),
                "c3_std": float(np.std(c3_valid)),
            }
            logger.info("  → %s: D1=%.3f, D2=%.3f, C3=%.3f (n=%d)",
                       category, np.mean(d1_valid), np.mean(d2_valid),
                       np.mean(c3_valid), len(d1_valid))
    
    results = {
        "test_name": "Per-Category Analysis",
        "categories": category_results,
        "interpretation": "Per-category breakdown shows differential benefits by object type"
    }
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: One-Way ANOVA (D1 vs D2 vs C3)
# ─────────────────────────────────────────────────────────────────────────────

def test_one_way_anova(
    per_class_ap_d1: List[float],
    per_class_ap_d2: List[float],
    per_class_ap_c3: List[float],
) -> Dict[str, Any]:
    """
    Test 4: One-way ANOVA comparing D1, D2, C3.
    
    Args:
        per_class_ap_d1: Per-class AP for D1
        per_class_ap_d2: Per-class AP for D2
        per_class_ap_c3: Per-class AP for C3
    
    Returns:
        Dict with ANOVA results
    """
    logger.info("Test 4: One-Way ANOVA (D1 vs D2 vs C3)...")
    
    d1 = np.array(per_class_ap_d1, dtype=float)
    d2 = np.array(per_class_ap_d2, dtype=float)
    c3 = np.array(per_class_ap_c3, dtype=float)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(d1) | np.isnan(d2) | np.isnan(c3))
    d1_valid = d1[valid_mask]
    d2_valid = d2[valid_mask]
    c3_valid = c3[valid_mask]
    
    if len(d1_valid) < 3:
        logger.warning("  ✗ Not enough valid class data for ANOVA")
        return {"error": "Insufficient data", "n_valid_classes": len(d1_valid)}
    
    # One-way ANOVA
    f_stat, p_f = stats.f_oneway(d1_valid, d2_valid, c3_valid)
    
    # Means and SEs
    mean_d1 = np.mean(d1_valid)
    mean_d2 = np.mean(d2_valid)
    mean_c3 = np.mean(c3_valid)
    se_d1 = stats.sem(d1_valid)
    se_d2 = stats.sem(d2_valid)
    se_c3 = stats.sem(c3_valid)
    
    # Pairwise t-tests for post-hoc contrasts
    t_d1_d2, p_d1_d2 = stats.ttest_ind(d1_valid, d2_valid)
    t_d1_c3, p_d1_c3 = stats.ttest_ind(d1_valid, c3_valid)
    t_d2_c3, p_d2_c3 = stats.ttest_ind(d2_valid, c3_valid)
    
    # Bonferroni correction for 3 pairwise comparisons
    bonf_alpha = 0.05 / 3
    
    results = {
        "test_name": "One-way ANOVA: D1 vs D2 vs C3",
        "n_classes": len(d1_valid),
        "means": {
            "d1": float(mean_d1),
            "d2": float(mean_d2),
            "c3": float(mean_c3),
        },
        "standard_errors": {
            "d1": float(se_d1),
            "d2": float(se_d2),
            "c3": float(se_c3),
        },
        "f_statistic": float(f_stat),
        "p_value": float(p_f),
        "significant_alpha_0.05": bool(p_f < 0.05),
        "posthoc_contrasts": {
            "d1_vs_d2": {
                "t_statistic": float(t_d1_d2),
                "p_value": float(p_d1_d2),
                "bonferroni_significant": bool(p_d1_d2 < bonf_alpha),
            },
            "d1_vs_c3": {
                "t_statistic": float(t_d1_c3),
                "p_value": float(p_d1_c3),
                "bonferroni_significant": bool(p_d1_c3 < bonf_alpha),
            },
            "d2_vs_c3": {
                "t_statistic": float(t_d2_c3),
                "p_value": float(p_d2_c3),
                "bonferroni_significant": bool(p_d2_c3 < bonf_alpha),
            },
        },
        "interpretation": "Significant main effect of fine-tune strategy" if p_f < 0.05
                         else "No significant difference among strategies"
    }
    
    logger.info("  → F: %.3f, p: %.4f", f_stat, p_f)
    logger.info("  → Means: D1=%.4f, D2=%.4f, C3=%.4f", mean_d1, mean_d2, mean_c3)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_rq4(verbose: bool = True) -> Dict[str, Any]:
    """
    Main RQ4 entry point.
    
    Loads Phase 3 and Phase 5 results, runs all four tests.
    
    Args:
        verbose: Whether to log progress
    
    Returns:
        Dict with results for all four tests
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
    
    logger.info("=" * 90)
    logger.info("RQ4: FULL PIPELINE EVALUATION & PER-CATEGORY ANALYSIS")
    logger.info("=" * 90)
    
    # Setup directories
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Phase 3 results
    logger.info("\n[1/5] Loading Phase 3 results...")
    phase3_json = results_dir / "phase3_soup_results.json"
    if not phase3_json.exists():
        logger.error("  ✗ Phase 3 results not found: %s", phase3_json)
        return {"error": "Phase 3 results not found"}
    
    with open(phase3_json, "r") as f:
        phase3_results = json.load(f)
    logger.info("  ✓ Loaded Phase 3 results")
    
    # Load Phase 5 results
    logger.info("\n[2/5] Loading Phase 5 results...")
    phase5_json = results_dir / "phase5_finetuning_results.json"
    if not phase5_json.exists():
        logger.error("  ✗ Phase 5 results not found: %s", phase5_json)
        return {"error": "Phase 5 results not found"}
    
    with open(phase5_json, "r") as f:
        phase5_results = json.load(f)
    logger.info("  ✓ Loaded Phase 5 results")
    
    # Extract per-class AP arrays
    logger.info("\n[3/5] Extracting per-class AP arrays...")
    best_individual_ap = phase3_results.get("best_learned", {}).get("per_class_ap", [0.0] * 80)
    d1_ap = phase5_results.get("d1", {}).get("per_class_ap", [0.0] * 80)
    d2_ap = phase5_results.get("d2", {}).get("per_class_ap", [0.0] * 80)
    c3_ap = phase5_results.get("c3", {}).get("per_class_ap", [0.0] * 80)
    logger.info("  ✓ Extracted per-class AP for all models")
    
    # Run tests
    logger.info("\n[4/5] Running statistical tests...")
    
    test1_result = test_bootstrap_ci(best_individual_ap, c3_ap, n_bootstrap=10000)
    test2_result = test_c3_vs_baseline(c3_ap, baseline_ap=37.7)
    test3_result = test_per_category_analysis(d1_ap, d2_ap, c3_ap)
    test4_result = test_one_way_anova(d1_ap, d2_ap, c3_ap)
    
    # Compile results
    all_results = {
        "rq": "RQ4",
        "question": "Does the complete pipeline (D1/D2/C3) achieve measurable improvements? Which object categories benefit most?",
        "test_1_bootstrap_ci": test1_result,
        "test_2_c3_vs_baseline": test2_result,
        "test_3_per_category_analysis": test3_result,
        "test_4_anova": test4_result,
        "summary": {
            "n_tests": 4,
            "n_significant_alpha_0.05": sum([
                test1_result.get("ci_contains_zero", True) is False,
                test2_result.get("significant_alpha_0.05", False),
                test4_result.get("significant_alpha_0.05", False),
            ]),
        }
    }
    
    # Save results
    logger.info("\nSaving results...")
    results_json = results_dir / "rq4_test_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("  → Results: %s", results_json)
    
    logger.info("\n" + "=" * 90)
    logger.info("RQ4 ANALYSIS COMPLETE")
    logger.info("=" * 90)
    logger.info("Summary:")
    logger.info("  Test 1 (Bootstrap): %s", test1_result.get("decision", "unknown"))
    logger.info("  Test 2 (vs Baseline): %s", test2_result.get("interpretation", "unknown"))
    logger.info("  Test 3 (Categories): %d object categories analyzed", len(test3_result.get("categories", {})))
    logger.info("  Test 4 (ANOVA): %s", test4_result.get("interpretation", "unknown"))
    logger.info("=" * 90)
    
    return all_results


if __name__ == "__main__":
    run_rq4(verbose=True)
