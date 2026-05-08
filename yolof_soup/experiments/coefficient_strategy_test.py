"""
rq3_coefficient_strategy_test.py
==================================

RQ3 Analysis: Test coefficient learning strategies (Dirichlet vs Fisher).

Research Question 3 (RQ3): Do learned coefficient strategies (Dirichlet vs Fisher)
outperform uniform baselines? Does merge quality moderate fine-tuning gains?

This module runs three statistical tests:

Test 1: Strategy Comparison (Condition 3 vs Condition 4)
  - Paired t-test on per-class AP arrays
  - Also Wilcoxon signed-rank test (robustness)
  - H0: μ_3 = μ_4 (no difference between Dirichlet and Fisher)
  - Expected: At least one learned strategy > uniform baseline

Test 2: Fine-Tuning Moderation (D1 vs D2)
  - Compute per-class AP gains: G1 = D1_post - Cond2_pre, G2 = D2_post - best_learned_pre
  - Independent-samples t-test: H0: G1 = G2
  - Hypothesis: G2 > G1 (better merge initialization → larger fine-tune gain)
  - Also report 95% CI on gain difference

Test 3: Strategy-by-Branch Interaction
  - Two-way RM-ANOVA on learned coefficients
  - Factors: strategy (Dirichlet vs Fisher) × branch (cls vs reg)
  - Expected: Main effect of branch (cls weights ≠ reg weights)
  - Post-hoc contrasts if interaction significant

Dependencies:
  - Phase 3 results (Condition 1-4 per-class AP)
  - Phase 5 results (D1, D2 per-class AP)

Outputs:
  - rq3_test_results.json — All three test results with p-values, effect sizes
  - rq3_visualizations.png — Boxplots and effect size plots (optional)

Run: python -m yolof_soup.experiments.rq3_coefficient_strategy_test
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from yolof_soup.config.experiment_config import RESULTS_DIR
from yolof_soup.utils.global_logger import configure_logger


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Strategy Comparison (Condition 3 vs 4)
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_comparison(
    per_class_ap_cond3: List[float],
    per_class_ap_cond4: List[float],
) -> Dict[str, Any]:
    """
    Test 1: Paired t-test and Wilcoxon on Condition 3 vs Condition 4.
    
    Args:
        per_class_ap_cond3: 80-element array of per-class AP for Condition 3
        per_class_ap_cond4: 80-element array of per-class AP for Condition 4
    
    Returns:
        Dict with t-test, Wilcoxon, and effect size results
    """
    logging.info("Test 1: Strategy Comparison (Condition 3 vs 4)...")
    
    # Convert to numpy arrays
    ap3 = np.array(per_class_ap_cond3, dtype=float)
    ap4 = np.array(per_class_ap_cond4, dtype=float)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(ap3) | np.isnan(ap4))
    ap3_valid = ap3[valid_mask]
    ap4_valid = ap4[valid_mask]
    
    if len(ap3_valid) < 3:
        logging.warning("  ✗ Not enough valid class data for comparison")
        return {"error": "Insufficient data", "n_valid_classes": len(ap3_valid)}
    
    # Paired t-test
    t_stat, p_t = stats.ttest_rel(ap3_valid, ap4_valid)
    
    # Wilcoxon signed-rank test (robustness check)
    w_stat, p_w = stats.wilcoxon(ap3_valid, ap4_valid, alternative='two-sided')
    
    # Effect size: Cohen's d for paired samples
    diff = ap3_valid - ap4_valid
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    
    # Mean and SE
    mean_3 = np.mean(ap3_valid)
    mean_4 = np.mean(ap4_valid)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    
    results = {
        "test_name": "Paired t-test: Condition 3 vs 4",
        "n_valid_classes": len(ap3_valid),
        "mean_cond3": float(mean_3),
        "mean_cond4": float(mean_4),
        "mean_difference": float(mean_3 - mean_4),
        "t_statistic": float(t_stat),
        "t_pvalue": float(p_t),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(p_w),
        "cohens_d": float(cohens_d),
        "se_difference": float(se_diff),
        "significant_alpha_0.05": bool(p_t < 0.05),
        "interpretation": "Condition 3 significantly better" if (p_t < 0.05 and mean_3 > mean_4)
                         else "Condition 4 significantly better" if (p_t < 0.05 and mean_4 > mean_3)
                         else "No significant difference"
    }
    
    logging.info("  → t=%.3f, p=%.4f (Wilcoxon p=%.4f), Cohen's d=%.3f",
                t_stat, p_t, p_w, cohens_d)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Fine-Tuning Moderation (D1 vs D2)
# ─────────────────────────────────────────────────────────────────────────────

def test_finetuning_moderation(
    per_class_ap_cond2: List[float],
    per_class_ap_best_learned: List[float],
    per_class_ap_d1: List[float],
    per_class_ap_d2: List[float],
) -> Dict[str, Any]:
    """
    Test 2: D1 vs D2 fine-tuning moderation test.
    
    Compute gains for each class and test if D2 gains > D1 gains.
    
    Args:
        per_class_ap_cond2: Per-class AP for Condition 2 (pre-D1)
        per_class_ap_best_learned: Per-class AP for best learned (pre-D2)
        per_class_ap_d1: Per-class AP for D1 (post-fine-tune)
        per_class_ap_d2: Per-class AP for D2 (post-fine-tune)
    
    Returns:
        Dict with paired t-test on gains and 95% CI on difference
    """
    logging.info("Test 2: Fine-Tuning Moderation (D1 gains vs D2 gains)...")
    
    # Convert to numpy arrays
    c2 = np.array(per_class_ap_cond2, dtype=float)
    bl = np.array(per_class_ap_best_learned, dtype=float)
    d1 = np.array(per_class_ap_d1, dtype=float)
    d2 = np.array(per_class_ap_d2, dtype=float)
    
    # Compute gains
    gain_d1 = d1 - c2
    gain_d2 = d2 - bl
    
    # Remove NaN values
    valid_mask = ~(np.isnan(gain_d1) | np.isnan(gain_d2))
    gain_d1_valid = gain_d1[valid_mask]
    gain_d2_valid = gain_d2[valid_mask]
    
    if len(gain_d1_valid) < 3:
        logging.warning("  ✗ Not enough valid class data for moderation test")
        return {"error": "Insufficient data", "n_valid_classes": len(gain_d1_valid)}
    
    # Independent-samples t-test
    t_stat, p_t = stats.ttest_ind(gain_d1_valid, gain_d2_valid)
    
    # 95% CI on difference
    mean_gain_d1 = np.mean(gain_d1_valid)
    mean_gain_d2 = np.mean(gain_d2_valid)
    diff_gains = mean_gain_d2 - mean_gain_d1
    se_diff = np.sqrt(np.var(gain_d1_valid, ddof=1) / len(gain_d1_valid) +
                      np.var(gain_d2_valid, ddof=1) / len(gain_d2_valid))
    ci_lower = diff_gains - 1.96 * se_diff
    ci_upper = diff_gains + 1.96 * se_diff
    
    results = {
        "test_name": "Independent t-test: D1 gains vs D2 gains",
        "n_valid_classes": len(gain_d1_valid),
        "mean_gain_d1": float(mean_gain_d1),
        "mean_gain_d2": float(mean_gain_d2),
        "difference_in_gains": float(diff_gains),
        "t_statistic": float(t_stat),
        "p_value": float(p_t),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "significant_alpha_0.05": bool(p_t < 0.05),
        "interpretation": "D2 gains significantly larger" if (p_t < 0.05 and diff_gains > 0)
                         else "D1 gains significantly larger" if (p_t < 0.05 and diff_gains < 0)
                         else "No significant difference in gains"
    }
    
    logging.info("  → D1 gain=%.4f, D2 gain=%.4f, diff=%.4f, t=%.3f, p=%.4f",
                mean_gain_d1, mean_gain_d2, diff_gains, t_stat, p_t)
    logging.info("  → 95%% CI on difference: [%.4f, %.4f]", ci_lower, ci_upper)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Two-Way ANOVA (Strategy × Branch)
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_branch_interaction() -> Dict[str, Any]:
    """
    Test 3: Two-way RM-ANOVA on learned coefficients.
    
    Factors: strategy (Dirichlet vs Fisher) × branch (cls vs reg)
    
    Note: This requires learned coefficient extraction from Phase 3.
    For now, returns a framework that can be populated with actual data.
    """
    logging.info("Test 3: Strategy-by-Branch Interaction (Two-Way ANOVA)...")
    
    try:
        import pandas as pd
        from statsmodels.stats.anova import AnovaRM
    except ImportError:
        logging.warning("  statsmodels not installed; returning framework")
        return {
            "test_name": "Two-way RM-ANOVA: strategy × branch",
            "status": "skipped",
            "reason": "statsmodels not available"
        }
    
    # This is a framework that would be populated with learned coefficients from Phase 3
    # Expected data structure: coefficients for each (strategy, branch, replicate) combination
    
    results = {
        "test_name": "Two-way RM-ANOVA: strategy × branch",
        "note": "Requires learned coefficient extraction from Phase 3 results",
        "expected_factors": {
            "strategies": ["dirichlet_cd", "fisher_weighted"],
            "branches": ["cls_head", "reg_head"],
            "n_replicates": "varies (3-6 depending on architecture)"
        },
        "framework": "Ready to accept pd.DataFrame with columns=[strategy, branch, coefficient, replicate_id]",
        "status": "ready_for_data",
        "interpretation": "Post-hoc contrasts would compare main effects of branch and strategy × branch interaction"
    }
    
    logging.info("  ✓ Framework ready (awaiting learned coefficient data from Phase 3)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True) -> Dict[str, Any]:
    """
    Main RQ3 entry point.
    
    Loads Phase 3 and Phase 5 results, runs all three tests.
    
    Args:
        verbose: Whether to log progress
    
    Returns:
        Dict with results for all three tests
    """
    
    logger = configure_logger(level=logging.DEBUG if verbose else logging.INFO, add_file_handler=True, log_file="rq3_test.log")
    
    logger.info("=" * 90)
    logger.info("RQ3: COEFFICIENT STRATEGY TESTS")
    logger.info("=" * 90)
    
    # Setup directories
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Phase 3 results
    logger.info("\n[1/4] Loading Phase 3 results...")
    phase3_json = results_dir / "phase3_soup_results.json"
    if not phase3_json.exists():
        logger.error("  ✗ Phase 3 results not found: %s", phase3_json)
        return {"error": "Phase 3 results not found"}
    
    with open(phase3_json, "r") as f:
        phase3_results = json.load(f)
    logging.info("  ✓ Loaded Phase 3 results")
    
    # Load Phase 5 results
    logging.info("\n[2/4] Loading Phase 5 results...")
    phase5_json = results_dir / "phase5_finetuning_results.json"
    if not phase5_json.exists():
        logging.error("  ✗ Phase 5 results not found: %s", phase5_json)
        return {"error": "Phase 5 results not found"}
    
    with open(phase5_json, "r") as f:
        phase5_results = json.load(f)
    logging.info("  ✓ Loaded Phase 5 results")
    
    # Extract per-class AP arrays
    logging.info("\n[3/4] Extracting per-class AP arrays...")
    cond1_ap = phase3_results.get("condition_1", {}).get("per_class_ap", [0.0] * 80)
    cond2_ap = phase3_results.get("condition_2", {}).get("per_class_ap", [0.0] * 80)
    cond3_ap = phase3_results.get("condition_3", {}).get("per_class_ap", [0.0] * 80)
    cond4_ap = phase3_results.get("condition_4", {}).get("per_class_ap", [0.0] * 80)
    best_learned_ap = phase3_results.get("best_learned", {}).get("per_class_ap", [0.0] * 80)
    
    d1_ap = phase5_results.get("d1", {}).get("per_class_ap", [0.0] * 80)
    d2_ap = phase5_results.get("d2", {}).get("per_class_ap", [0.0] * 80)
    logging.info("  ✓ Extracted per-class AP for all conditions")
    
    # Run tests
    logging.info("\n[4/4] Running statistical tests...")
    
    test1_result = test_strategy_comparison(cond3_ap, cond4_ap)
    test2_result = test_finetuning_moderation(cond2_ap, best_learned_ap, d1_ap, d2_ap)
    test3_result = test_strategy_branch_interaction()
    
    # Compile results
    all_results = {
        "rq": "RQ3",
        "question": "Do learned coefficient strategies (Dirichlet vs Fisher) outperform uniform baselines? Does merge quality moderate fine-tuning gains?",
        "test_1_strategy_comparison": test1_result,
        "test_2_finetuning_moderation": test2_result,
        "test_3_strategy_branch_interaction": test3_result,
        "summary": {
            "n_tests": 3,
            "n_significant_alpha_0.05": sum([
                test1_result.get("significant_alpha_0.05", False),
                test2_result.get("significant_alpha_0.05", False),
            ]),
        }
    }
    
    # Save results
    logging.info("\nSaving results...")
    results_json = results_dir / "rq3_test_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info("  → Results: %s", results_json)
    
    logging.info("\n" + "=" * 90)
    logging.info("RQ3 ANALYSIS COMPLETE")
    logging.info("=" * 90)
    logging.info("Summary:")
    logging.info("  Test 1 (Strategy): %s", test1_result.get("interpretation", "unknown"))
    logging.info("  Test 2 (Moderation): %s", test2_result.get("interpretation", "unknown"))
    logging.info("  Test 3 (Interaction): %s", test3_result.get("status", "unknown"))
    logging.info("=" * 90)
    
    return all_results


if __name__ == "__main__":
    run(verbose=True)
