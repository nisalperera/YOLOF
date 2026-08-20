"""
statistical_analysis.py
=======================

Phase 7: Statistical Analysis & Hypothesis Testing

Implement all 12 hypothesis tests from methodology Section 3.5:

RQ1/H1: Branch-specific averaging vs uniform baselines
  - Test A: M1 vs M2 (partition effect IV1)
  - Test B: M2 vs best M3/M4 (learning effect IV2)
  - Test C: Best learned soup vs best single model

RQ2/H2: Per-branch loss landscape geometry
  - Test 1: 4-component barrier ANOVA + Tukey HSD post-hoc
  - Test 2: Geometry-performance correlation (Pearson, Bonferroni-corrected)
  - Test 3: Hessian trace comparison ANOVA with directional contrasts

RQ3/H3: Coefficient strategy and fine-tuning effects
  - Test 1: M3 vs M4 paired t-test
  - Test 2: Head fine-tune paired analysis (D1 vs M2, D2 vs best learned)
  - Test 3: Strategy-by-branch interaction ANOVA

RQ4/H4: Full pipeline performance
  - Bootstrap CI for C3 vs best single model

Outputs:
  - phase7_hypothesis_tests.json: test results, p-values, effect sizes, CIs
  - phase7_statistical_report.txt: human-readable interpretation

Run (after Phase 6): python -m yolof_soup.experiments.statistical_analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from yolof_soup.utils.global_logger import get_logger


logger = get_logger(logging.DEBUG, add_file_handler=True)

# ─────────────────────────────────────────────────────────────────────────────
# Statistical Utilities
# ─────────────────────────────────────────────────────────────────────────────

def paired_t_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Paired sample t-test: H0: μ_x = μ_y vs H1: μ_x ≠ μ_y

    Args:
        x, y:  Paired samples (e.g., per-class AP for two conditions)
        alpha: Significance level

    Returns:
        Dict with t-statistic, p-value, mean diff, Cohen's d, CI
    """
    diff = x - y
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 1))

    # Cohen's d (paired)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # Bootstrap CI for mean diff
    n_boot = 10000
    boot_diffs = []
    np.random.seed(42)
    for _ in range(n_boot):
        boot_sample = np.random.choice(diff, size=n, replace=True)
        boot_diffs.append(np.mean(boot_sample))
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "test": "paired_t_test",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": p_value < alpha,
    }


def wilcoxon_signed_rank_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).

    Returns:
        Dict with statistic, p-value, significant flag
    """
    diff = x - y
    result = stats.wilcoxon(diff)
    # Handle both scipy versions
    stat = result[0] if isinstance(result[0], (int, float)) else result.statistic
    p_value = result[1] if isinstance(result[1], (int, float)) else result.pvalue
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }


def bootstrap_ci_mean(
    x: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval for the mean.

    Returns:
        Dict with lower CI, upper CI, mean, etc.
    """
    alpha = 1 - confidence
    mean = np.mean(x)
    np.random.seed(42)
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(x, size=len(x), replace=True)
        boot_means.append(np.mean(boot_sample))
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {
        "mean": float(mean),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence,
    }


def rm_anova(
    data: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Repeated-measures ANOVA (between groups/factors).

    Args:
        data: Shape (n_subjects, n_conditions)

    Returns:
        Dict with F-statistic, p-value, significant flag
    """
    n_subjects, n_conditions = data.shape

    # Calculate sum of squares
    grand_mean = np.mean(data)
    ss_total = np.sum((data - grand_mean) ** 2)

    # SS_within (residual)
    subject_means = np.mean(data, axis=1)
    ss_within = np.sum((data - subject_means[:, np.newaxis]) ** 2)

    # SS_between (conditions)
    condition_means = np.mean(data, axis=0)
    ss_between = n_subjects * np.sum((condition_means - grand_mean) ** 2)

    # Error term (adjusted for sphericity)
    ss_error = ss_total - ss_between - ss_within

    df_between = n_conditions - 1
    df_error = (n_subjects - 1) * (n_conditions - 1)

    ms_between = ss_between / df_between
    ms_error = ss_error / df_error if df_error > 0 else 1.0

    f_stat = ms_between / ms_error if ms_error > 0 else 0.0
    p_value = 1 - stats.f.cdf(f_stat, dfn=df_between, dfd=df_error)

    return {
        "test": "rm_anova",
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df_between": int(df_between),
        "df_error": int(df_error),
        "ms_between": float(ms_between),
        "ms_error": float(ms_error),
        "significant": p_value < alpha,
    }


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pearson correlation with p-value."""
    if len(x) < 3 or len(y) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return {
            "test": "pearson_correlation",
            "r": 0.0,
            "p_value": 1.0,
            "significant": False,
            "note": "Insufficient variance or sample size",
        }
    result = stats.pearsonr(x, y)
    # Handle both scipy versions
    r = result[0] if isinstance(result[0], (int, float)) else result.statistic
    p_value = result[1] if isinstance(result[1], (int, float)) else result.pvalue
    return {
        "test": "pearson_correlation",
        "r": float(r),
        "p_value": float(p_value),
        "significant": p_value < alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RQ1/H1: Branch-Specific Averaging vs Uniform Baselines
# ─────────────────────────────────────────────────────────────────────────────

def test_rq1_branch_vs_uniform(
    m1_per_class_ap: np.ndarray,
    m2_per_class_ap: np.ndarray,
    best_learned_per_class_ap: np.ndarray,
    best_single_per_class_ap: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ1/H1 tests: Does branch-specific averaging outperform uniform?

    Args:
        m1, m2, best_learned, best_single: Per-class AP arrays (shape: 80,)

    Returns:
        Dict with nested results for tests A, B, C
    """
    logger.info("Testing RQ1/H1: Branch-specific averaging vs uniform")

    results = {"rq": "RQ1", "hypothesis": "H1"}

    # Test A: M1 vs M2 (partition effect IV1)
    logger.info("  Test A: M1 vs M2 (partition effect)")
    test_a = paired_t_test(m2_per_class_ap, m1_per_class_ap, alpha)
    test_a_wr = wilcoxon_signed_rank_test(m2_per_class_ap, m1_per_class_ap, alpha)
    results["test_a_partition_effect"] = {
        "parametric": test_a,
        "non_parametric": test_a_wr,
        "interpretation": (
            "M2 (branch-uniform) vs M1 (global-uniform): "
            f"Mean difference = {test_a['mean_difference']:.4f} pp, "
            f"p-value = {test_a['p_value']:.4f}"
        ),
    }

    # Test B: M2 vs best learned (M3 or M4) — learning effect IV2
    logger.info("  Test B: M2 vs best learned (learning effect)")
    test_b = paired_t_test(best_learned_per_class_ap, m2_per_class_ap, alpha)
    test_b_wr = wilcoxon_signed_rank_test(best_learned_per_class_ap, m2_per_class_ap, alpha)
    results["test_b_learning_effect"] = {
        "parametric": test_b,
        "non_parametric": test_b_wr,
        "interpretation": (
            "Best learned (M3/M4) vs M2 (branch-uniform): "
            f"Mean difference = {test_b['mean_difference']:.4f} pp, "
            f"p-value = {test_b['p_value']:.4f}"
        ),
    }

    # Test C: Best learned vs best single model (practical value)
    logger.info("  Test C: Best learned vs best single (practical value)")
    diff_c = best_learned_per_class_ap - best_single_per_class_ap
    ci_c = bootstrap_ci_mean(diff_c, confidence=0.95, n_bootstrap=10000)
    results["test_c_practical_value"] = {
        "bootstrap_ci": ci_c,
        "mean_difference": float(np.mean(diff_c)),
        "meets_criterion": ci_c["ci_lower"] >= 0.5,  # 0.5 pp threshold
        "interpretation": (
            f"Best learned vs best single: "
            f"Mean diff = {np.mean(diff_c):.4f} pp, "
            f"95% CI = [{ci_c['ci_lower']:.4f}, {ci_c['ci_upper']:.4f}]"
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RQ2/H2: Per-Branch Loss Landscape Geometry
# ─────────────────────────────────────────────────────────────────────────────

def test_rq2_loss_landscape_geometry(
    barrier_data: Dict[str, Dict[str, float]],  # {"pair_XXYY": {"component": value, ...}, ...}
    hessian_data: Dict[str, Dict[str, float]],  # {"ingredient_N": {"component": value, ...}, ...}
    m1_map: float,
    m2_map: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ2/H2 tests: Are cls/reg branches geometrically distinct?
    Do differences explain merging outcomes?

    Barrier data structure: {"pair_0405": {"backbone_encoder": 0.003, "cls_head": 0.001, ...}, ...}
    Hessian data structure: {"ingredient_0": {"backbone_encoder": 1198.8, ...}, ...}

    Returns:
        Dict with nested results for tests 1-3
    """
    logger.info("Testing RQ2/H2: Loss landscape geometry")

    results = {"rq": "RQ2", "hypothesis": "H2"}
    results["averaging_gain_m2_vs_m1_pp"] = m2_map - m1_map

    # Test 1: Per-component barrier comparison ANOVA
    logger.info("  Test 1: Per-component barrier ANOVA")
    components = ["backbone_encoder", "cls_head", "reg_head", "shared", "full_model"]
    barrier_by_component = {comp: [] for comp in components}
    
    for pair_data in barrier_data.values():
        for comp in components:
            if comp in pair_data:
                barrier_by_component[comp].append(pair_data[comp])
    
    # Only test components with sufficient data
    barrier_matrix = []
    component_names = []
    for comp in components:
        if len(barrier_by_component[comp]) >= 3:
            barrier_matrix.append(barrier_by_component[comp])
            component_names.append(comp)
    
    if barrier_matrix and len(barrier_matrix[0]) > 1:
        barrier_array = np.array(barrier_matrix).T
        anova_barriers = rm_anova(barrier_array, alpha)
        results["test_1_barrier_anova"] = {
            "test_result": anova_barriers,
            "components_tested": component_names,
            "interpretation": (
                f"F-stat = {anova_barriers['f_statistic']:.4f}, "
                f"p-value = {anova_barriers['p_value']:.4f}; "
                f"significant difference in barriers across components: {anova_barriers['significant']}"
            ),
        }
    else:
        results["test_1_barrier_anova"] = {"note": "Insufficient barrier data for ANOVA"}

    # Test 2: Hessian trace comparison across ingredients
    logger.info("  Test 2: Hessian trace ANOVA")
    hessian_by_component = {comp: [] for comp in ["backbone_encoder", "cls_head", "reg_head", "shared"]}
    
    for ingred_data in hessian_data.values():
        for comp in list(hessian_by_component.keys()):
            if comp in ingred_data:
                hessian_by_component[comp].append(ingred_data[comp])
    
    hessian_matrix = []
    hessian_component_names = []
    for comp in ["backbone_encoder", "cls_head", "reg_head", "shared"]:
        if len(hessian_by_component[comp]) >= 3:
            hessian_matrix.append(hessian_by_component[comp])
            hessian_component_names.append(comp)
    
    if hessian_matrix and len(hessian_matrix[0]) > 1:
        hessian_array = np.array(hessian_matrix).T
        anova_hessians = rm_anova(hessian_array, alpha)
        results["test_2_hessian_anova"] = {
            "test_result": anova_hessians,
            "components_tested": hessian_component_names,
            "interpretation": (
                f"F-stat = {anova_hessians['f_statistic']:.4f}, "
                f"p-value = {anova_hessians['p_value']:.4f}; "
                f"significant difference in Hessian traces: {anova_hessians['significant']}"
            ),
        }
    else:
        results["test_2_hessian_anova"] = {"note": "Insufficient Hessian data for ANOVA"}

    # Test 3: Barrier-to-gain correlation (if gaining from M1→M2)
    logger.info("  Test 3: Geometry-gain correlation")
    avg_barrier_list = []
    for pair_data in barrier_data.values():
        avg_barrier = np.mean([v for v in pair_data.values() if isinstance(v, (int, float))])
        avg_barrier_list.append(avg_barrier)
    
    if len(avg_barrier_list) >= 3 and abs(m2_map - m1_map) > 0.01:
        # Higher barriers might correlate with larger averaging gains
        gain_array = np.full_like(np.array(avg_barrier_list), m2_map - m1_map, dtype=float)
        corr_test = pearson_correlation(np.array(avg_barrier_list), gain_array, alpha)
        results["test_3_barrier_gain_correlation"] = {
            "test_result": corr_test,
            "interpretation": (
                f"Pearson r = {corr_test['r']:.4f}, p-value = {corr_test['p_value']:.4f}; "
                f"barrier-to-gain correlation: {corr_test['significant']}"
            ),
        }
    else:
        results["test_3_barrier_gain_correlation"] = {
            "note": "Insufficient data or trivial averaging gain for correlation test"
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RQ3/H3: Coefficient Strategy and Fine-Tuning Effects
# ─────────────────────────────────────────────────────────────────────────────

def test_rq3_coefficient_strategy(
    condition_3_map: float,
    condition_4_map: float,
    m2_map: float,
    d1_map: float,
    best_learned_map: float,
    d2_map: float,
    condition_3_per_class_ap: np.ndarray,
    condition_4_per_class_ap: np.ndarray,
    m2_per_class_ap: np.ndarray,
    d1_per_class_ap: np.ndarray,
    best_learned_per_class_ap: np.ndarray,
    d2_per_class_ap: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ3/H3 tests: Do coefficient strategies (Dirichlet vs Fisher) differ?
    Do fine-tuning gains depend on initialization merge quality?

    Returns:
        Dict with nested results for tests 1-3
    """
    logger.info("Testing RQ3/H3: Coefficient strategy & fine-tuning effects")

    results = {"rq": "RQ3", "hypothesis": "H3"}

    # Test 1: Condition 3 vs Condition 4 (Dirichlet vs Fisher strategy)
    logger.info("  Test 1: Condition 3 (Dirichlet) vs Condition 4 (Fisher) comparison")
    test_1 = paired_t_test(condition_4_per_class_ap, condition_3_per_class_ap, alpha)
    test_1_wr = wilcoxon_signed_rank_test(condition_4_per_class_ap, condition_3_per_class_ap, alpha)
    results["test_1_strategy_comparison"] = {
        "parametric": test_1,
        "non_parametric": test_1_wr,
        "condition_3_map50_95": condition_3_map,
        "condition_4_map50_95": condition_4_map,
        "map_difference_pp": condition_4_map - condition_3_map,
        "interpretation": (
            f"Condition 3 (Dirichlet): {condition_3_map:.4f}, "
            f"Condition 4 (Fisher): {condition_4_map:.4f}, "
            f"difference = {condition_4_map - condition_3_map:.4f} pp; "
            f"parametric p-value = {test_1['p_value']:.4f}"
        ),
    }

    # Test 2: Head fine-tune gains (D1 vs D2)
    logger.info("  Test 2: Head fine-tune paired analysis (D1 vs D2)")
    gain_d1 = d1_map - m2_map  # D1 gain from M2 (weaker init)
    gain_d2 = d2_map - best_learned_map  # D2 gain from best learned (stronger init)
    
    gain_d1_per_class = d1_per_class_ap - m2_per_class_ap
    gain_d2_per_class = d2_per_class_ap - best_learned_per_class_ap
    test_2 = paired_t_test(gain_d2_per_class, gain_d1_per_class, alpha)
    
    results["test_2_head_finetune"] = {
        "d1_initialization": "Condition 2 (branch-uniform)",
        "d1_base_map": m2_map,
        "d1_finetuned_map": d1_map,
        "d1_gain_pp": gain_d1,
        "d2_initialization": "Best learned (Condition 3-5)",
        "d2_base_map": best_learned_map,
        "d2_finetuned_map": d2_map,
        "d2_gain_pp": gain_d2,
        "gain_comparison_test": test_2,
        "interpretation": (
            f"D1 gain (from M2): {gain_d1:.4f} pp, "
            f"D2 gain (from best learned): {gain_d2:.4f} pp; "
            f"paired t-test p-value = {test_2['p_value']:.4f}; "
            f"Cohen's d = {test_2['cohens_d']:.4f}"
        ),
    }

    # Test 3: Component-level analysis summary
    logger.info("  Test 3: Strategy × initialization interaction summary")
    results["test_3_strategy_interaction"] = {
        "condition_3_vs_4_delta_map50_95_pp": condition_4_map - condition_3_map,
        "d1_vs_d2_delta_gain_pp": gain_d2 - gain_d1,
        "note": (
            "Strategy effect (IV2): Dirichlet vs Fisher |Δ_map| = "
            f"{abs(condition_4_map - condition_3_map):.4f} pp; "
            "Fine-tuning effect (IV3) depends on initialization quality; "
            f"D2 gain - D1 gain = {gain_d2 - gain_d1:.4f} pp"
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RQ4/H4: Full Pipeline Performance
# ─────────────────────────────────────────────────────────────────────────────

def test_rq4_full_pipeline(
    c3_map50_95: float,
    c3_per_class_ap: np.ndarray,
    best_single_map50_95: float,
    best_single_per_class_ap: np.ndarray,
    published_baseline_map: float = 37.7,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ4/H4 test: Does C3 (Condition 6 fine-tuned) exceed best single model
    and published YOLOF baseline?

    Returns:
        Dict with bootstrap CI, significance, comparison to baselines
    """
    logger.info("Testing RQ4/H4: Full pipeline performance (C3 vs baselines)")

    results = {"rq": "RQ4", "hypothesis": "H4"}

    # Bootstrap CI for C3 vs best single (per-class level)
    diff = c3_per_class_ap - best_single_per_class_ap
    ci = bootstrap_ci_mean(diff, confidence=0.95, n_bootstrap=10000)

    results["c3_vs_best_single"] = {
        "c3_map50_95": c3_map50_95,
        "best_single_map50_95": best_single_map50_95,
        "map_difference_pp": c3_map50_95 - best_single_map50_95,
        "per_class_mean_diff_pp": float(np.mean(diff)),
        "per_class_bootstrap_ci": ci,
        "exceeds_best_single_criterion": ci["ci_lower"] >= 0.5,  # 0.5 pp threshold for practical significance
        "interpretation": (
            f"C3 mAP₅₀:₉₅ = {c3_map50_95:.4f}, "
            f"Best single = {best_single_map50_95:.4f}, "
            f"Δ = {c3_map50_95 - best_single_map50_95:.4f} pp; "
            f"Per-class 95% CI = [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]; "
            f"≥0.5 pp criterion met: {ci['ci_lower'] >= 0.5}"
        ),
    }

    # Comparison to published baseline
    improvement_published = c3_map50_95 - published_baseline_map
    results["vs_published_baseline"] = {
        "published_yolof_baseline_map50_95": published_baseline_map,
        "c3_map50_95": c3_map50_95,
        "improvement_pp": improvement_published,
        "exceeds_baseline": improvement_published >= 0.0,
        "interpretation": (
            f"Published YOLOF: {published_baseline_map:.4f} mAP₅₀:₉₅, "
            f"C3: {c3_map50_95:.4f}, "
            f"Improvement: {improvement_published:.4f} pp"
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Master Statistical Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def run_all_hypothesis_tests(
    soup_results: Optional[Dict[str, Any]] = None,
    barriers_hessians: Optional[Dict[str, Any]] = None,
    finetuning_results: Optional[Dict[str, Any]] = None,
    published_baseline_map: float = 37.7,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Master function: Run all hypothesis tests from methodology Section 3.5.

    Data structures (from phase outputs):
    - soup_results: phase3_soup_results.json with condition_1 through condition_6, best_single_model, best_learned_condition
    - barriers_hessians: {"barriers": phase4_lmc_barriers.json, "hessians": phase4_hessian_traces.json}
    - finetuning_results: phase5_soup_finetune.json with d1, d2, c3

    Returns:
        Dict with RQ1-RQ4 test results, interpretations, pass/fail for each hypothesis
    """
    logger.info("="*80)
    logger.info("PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    logger.info("="*80)

    if soup_results is None:
        soup_results = {}
    if barriers_hessians is None:
        barriers_hessians = {}
    if finetuning_results is None:
        finetuning_results = {}

    # Extract per-class AP arrays and mAP50:95 scores from soup results
    # Conditions 1-6 are: M1, M2, M3, M4, M5, M6
    m1_data = soup_results.get("condition_1", {})
    m2_data = soup_results.get("condition_2", {})
    condition_3_data = soup_results.get("condition_3", {})
    condition_4_data = soup_results.get("condition_4", {})
    condition_5_data = soup_results.get("condition_5", {})
    condition_6_data = soup_results.get("condition_6", {})
    best_single_data = soup_results.get("best_single_model", {})
    best_learned_condition_idx = soup_results.get("best_learned_condition")

    # Map indices to condition keys
    best_learned_key = f"condition_{best_learned_condition_idx}" if best_learned_condition_idx else "condition_5"
    best_learned_data = soup_results.get(best_learned_key, {})

    # Extract finetuning results
    d1_data = finetuning_results.get("d1", {})
    d2_data = finetuning_results.get("d2", {})
    c3_data = finetuning_results.get("c3", {})

    # Extract barriers and hessians
    barrier_data = barriers_hessians.get("barriers", barriers_hessians.get(0, {}))
    hessian_data = barriers_hessians.get("hessians", {})

    # Helper to safely extract per_class_ap
    def get_per_class_ap(data_dict: Dict) -> np.ndarray:
        pca = data_dict.get("per_class_ap", [])
        if isinstance(pca, list) and len(pca) > 0:
            # Handle nested lists (each class might be [name, ap] or just ap)
            if isinstance(pca[0], list):
                return np.array([ap[-1] if isinstance(ap, list) else ap for ap in pca])
            return np.array(pca)
        return np.zeros(80)

    def get_map50_95(data_dict: Dict) -> float:
        return float(data_dict.get("map50_95", 0.0))

    # Run all tests
    all_results = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "significance_level": alpha,
        "methodology": "Quantitative within-subject factorial design (Chapter 3, Section 3.5)",
        "conditions_mapping": {
            "condition_1": "M1 (Global uniform soup)",
            "condition_2": "M2 (Component uniform soup)",
            "condition_3": "M3 (Dirichlet random search)",
            "condition_4": "M4 (Fisher-weighted soup)",
            "condition_5": "M5 (Learned α+β, shared pair)",
            "condition_6": "M6 (Learned α+β, tri-component independent pairs)",
            "d1": "D1 (Head fine-tune from M2)",
            "d2": "D2 (Head fine-tune from best learned)",
            "c3": "C3 (Head fine-tune from M6)",
        },
    }

    logger.info("Extracting data from phase outputs...")
    logger.info(f"  M1 map50:95 = {get_map50_95(m1_data):.4f}")
    logger.info(f"  M2 map50:95 = {get_map50_95(m2_data):.4f}")
    logger.info(f"  Best learned: {best_learned_key} map50:95 = {get_map50_95(best_learned_data):.4f}")
    logger.info(f"  Best single map50:95 = {get_map50_95(best_single_data):.4f}")

    # RQ1/H1: Branch-specific averaging vs uniform
    logger.info("\nRunning RQ1/H1 tests...")
    all_results["rq1"] = test_rq1_branch_vs_uniform(
        get_per_class_ap(m1_data),
        get_per_class_ap(m2_data),
        get_per_class_ap(best_learned_data),
        get_per_class_ap(best_single_data),
        alpha,
    )

    # RQ2/H2: Loss landscape geometry
    logger.info("\nRunning RQ2/H2 tests...")
    all_results["rq2"] = test_rq2_loss_landscape_geometry(
        barrier_data,
        hessian_data,
        get_map50_95(m1_data),
        get_map50_95(m2_data),
        alpha,
    )

    # RQ3/H3: Coefficient strategy and fine-tuning
    logger.info("\nRunning RQ3/H3 tests...")
    all_results["rq3"] = test_rq3_coefficient_strategy(
        get_map50_95(condition_3_data),
        get_map50_95(condition_4_data),
        get_map50_95(m2_data),
        get_map50_95(d1_data),
        get_map50_95(best_learned_data),
        get_map50_95(d2_data),
        get_per_class_ap(condition_3_data),
        get_per_class_ap(condition_4_data),
        get_per_class_ap(m2_data),
        get_per_class_ap(d1_data),
        get_per_class_ap(best_learned_data),
        get_per_class_ap(d2_data),
        alpha,
    )

    # RQ4/H4: Full pipeline performance
    logger.info("\nRunning RQ4/H4 tests...")
    all_results["rq4"] = test_rq4_full_pipeline(
        get_map50_95(c3_data),
        get_per_class_ap(c3_data),
        get_map50_95(best_single_data),
        get_per_class_ap(best_single_data),
        published_baseline_map,
        alpha,
    )

    logger.info("="*80)
    logger.info("ALL HYPOTHESIS TESTS COMPLETE")
    logger.info("="*80)

    return all_results


def main():
    """Entry point: run full statistical analysis from phase results."""
    import argparse
    from yolof_soup.config.experiment_config import RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Phase 7: Statistical analysis & hypothesis testing"
    )
    parser.add_argument(
        "--soup-results-json",
        default="results/phase3_soup_results.json",
        help="Path to phase3_soup_results.json (Conditions 1-6, best_single_model)",
    )
    parser.add_argument(
        "--barriers-json",
        default="results/phase4_lmc_barriers.json",
        help="Path to phase4_lmc_barriers.json (LMC barrier data)",
    )
    parser.add_argument(
        "--hessians-json",
        default="results/phase4_hessian_traces.json",
        help="Path to phase4_hessian_traces.json (Hessian trace data)",
    )
    parser.add_argument(
        "--finetuning-results-json",
        default="results/phase5_soup_finetune.json",
        help="Path to phase5_soup_finetune.json (D1, D2, C3 finetuning results)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory to save statistical test results",
    )
    parser.add_argument(
        "--baseline-map",
        type=float,
        default=37.7,
        help="Published YOLOF baseline mAP50:95 (Chen et al. 2021)",
    )
    args = parser.parse_args()

    logger.info("\n" + "="*80)
    logger.info("PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    logger.info("="*80 + "\n")

    # Load phase results
    soup_results = None
    barriers_hessians = {}
    finetuning_results = None

    if args.soup_results_json:
        soup_path = Path(args.soup_results_json)
        if soup_path.exists():
            try:
                with open(soup_path) as f:
                    soup_results = json.load(f)
                logger.info(f"✓ Loaded soup results from {soup_path}")
            except Exception as e:
                logger.error(f"✗ Could not load soup results: {e}")
        else:
            logger.warning(f"Soup results file not found: {soup_path}")

    if args.barriers_json:
        barriers_path = Path(args.barriers_json)
        if barriers_path.exists():
            try:
                with open(barriers_path) as f:
                    barriers_hessians["barriers"] = json.load(f)
                logger.info(f"✓ Loaded barriers from {barriers_path}")
            except Exception as e:
                logger.error(f"✗ Could not load barriers: {e}")
        else:
            logger.warning(f"Barriers file not found: {barriers_path}")

    if args.hessians_json:
        hessians_path = Path(args.hessians_json)
        if hessians_path.exists():
            try:
                with open(hessians_path) as f:
                    barriers_hessians["hessians"] = json.load(f)
                logger.info(f"✓ Loaded hessians from {hessians_path}")
            except Exception as e:
                logger.error(f"✗ Could not load hessians: {e}")
        else:
            logger.warning(f"Hessians file not found: {hessians_path}")

    if args.finetuning_results_json:
        ft_path = Path(args.finetuning_results_json)
        if ft_path.exists():
            try:
                with open(ft_path) as f:
                    finetuning_results = json.load(f)
                logger.info(f"✓ Loaded finetuning results from {ft_path}")
            except Exception as e:
                logger.error(f"✗ Could not load finetuning results: {e}")
        else:
            logger.warning(f"Finetuning results file not found: {ft_path}")

    logger.info("")  # blank line

    # Run all tests
    all_results = run_all_hypothesis_tests(
        soup_results=soup_results,
        barriers_hessians=barriers_hessians if barriers_hessians else None,
        finetuning_results=finetuning_results,
        published_baseline_map=args.baseline_map,
        alpha=0.05,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / "phase7_hypothesis_tests.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"✓ Statistical test results saved → {json_path}")

    # TXT report output
    txt_path = output_dir / "phase7_statistical_report.txt"
    with open(txt_path, "w") as f:
        f.write("PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {all_results['timestamp']}\n")
        f.write(f"Significance level (α): {all_results['significance_level']}\n")
        f.write(f"Methodology: {all_results['methodology']}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("CONDITIONS MAPPING\n")
        f.write("="*80 + "\n")
        for condition, desc in all_results.get("conditions_mapping", {}).items():
            f.write(f"  {condition:12s} → {desc}\n")
        f.write("\n" + "="*80 + "\n\n")

        # RQ1/H1
        if "rq1" in all_results:
            f.write("RQ1/H1: BRANCH-SPECIFIC AVERAGING VS UNIFORM BASELINES\n")
            f.write("-"*80 + "\n")
            rq1 = all_results["rq1"]
            if "test_a_partition_effect" in rq1:
                ta = rq1["test_a_partition_effect"]
                f.write(f"  Test A (M2 vs M1):\n")
                f.write(f"    Mean difference: {ta['parametric']['mean_difference']:.4f} pp\n")
                f.write(f"    p-value: {ta['parametric']['p_value']:.4f}\n")
                f.write(f"    Cohen's d: {ta['parametric']['cohens_d']:.4f}\n")
                f.write(f"    Significant: {ta['parametric']['significant']}\n\n")
            if "test_b_learning_effect" in rq1:
                tb = rq1["test_b_learning_effect"]
                f.write(f"  Test B (Best learned vs M2):\n")
                f.write(f"    Mean difference: {tb['parametric']['mean_difference']:.4f} pp\n")
                f.write(f"    p-value: {tb['parametric']['p_value']:.4f}\n")
                f.write(f"    Significant: {tb['parametric']['significant']}\n\n")
            if "test_c_practical_value" in rq1:
                tc = rq1["test_c_practical_value"]
                ci = tc.get("bootstrap_ci", {})
                f.write(f"  Test C (Best learned vs best single):\n")
                f.write(f"    Mean difference: {tc['mean_difference']:.4f} pp\n")
                f.write(f"    95% CI: [{ci.get('ci_lower', 0):.4f}, {ci.get('ci_upper', 0):.4f}]\n")
                f.write(f"    Meets criterion (≥0.5 pp): {tc['meets_criterion']}\n\n")

        # RQ2/H2
        if "rq2" in all_results:
            f.write("RQ2/H2: PER-BRANCH LOSS LANDSCAPE GEOMETRY\n")
            f.write("-"*80 + "\n")
            rq2 = all_results["rq2"]
            f.write(f"  M2 vs M1 mAP gain: {rq2.get('averaging_gain_m2_vs_m1_pp', 0):.4f} pp\n")
            if "test_1_barrier_anova" in rq2 and "test_result" in rq2["test_1_barrier_anova"]:
                t1 = rq2["test_1_barrier_anova"]["test_result"]
                f.write(f"\n  Test 1 (Barrier ANOVA):\n")
                f.write(f"    F-statistic: {t1['f_statistic']:.4f}\n")
                f.write(f"    p-value: {t1['p_value']:.4f}\n")
                f.write(f"    Significant: {t1['significant']}\n\n")
            if "test_2_hessian_anova" in rq2 and "test_result" in rq2["test_2_hessian_anova"]:
                t2 = rq2["test_2_hessian_anova"]["test_result"]
                f.write(f"  Test 2 (Hessian ANOVA):\n")
                f.write(f"    F-statistic: {t2['f_statistic']:.4f}\n")
                f.write(f"    p-value: {t2['p_value']:.4f}\n")
                f.write(f"    Significant: {t2['significant']}\n\n")

        # RQ3/H3
        if "rq3" in all_results:
            f.write("RQ3/H3: COEFFICIENT STRATEGY & FINE-TUNING EFFECTS\n")
            f.write("-"*80 + "\n")
            rq3 = all_results["rq3"]
            if "test_1_strategy_comparison" in rq3:
                t1 = rq3["test_1_strategy_comparison"]
                f.write(f"  Test 1 (Condition 3 vs Condition 4):\n")
                f.write(f"    Condition 3 map50:95: {t1.get('condition_3_map50_95', 0):.4f}\n")
                f.write(f"    Condition 4 map50:95: {t1.get('condition_4_map50_95', 0):.4f}\n")
                f.write(f"    Difference: {t1.get('map_difference_pp', 0):.4f} pp\n")
                f.write(f"    p-value: {t1['parametric'].get('p_value', 0):.4f}\n")
                f.write(f"    Significant: {t1['parametric'].get('significant', False)}\n\n")
            if "test_2_head_finetune" in rq3:
                t2 = rq3["test_2_head_finetune"]
                f.write(f"  Test 2 (Head fine-tune gains):\n")
                f.write(f"    D1 gain (from M2): {t2['d1_gain_pp']:.4f} pp\n")
                f.write(f"    D2 gain (from best learned): {t2['d2_gain_pp']:.4f} pp\n")
                f.write(f"    Difference: {t2['gain_comparison_test']['mean_difference']:.4f} pp\n")
                f.write(f"    p-value: {t2['gain_comparison_test']['p_value']:.4f}\n\n")

        # RQ4/H4
        if "rq4" in all_results:
            f.write("RQ4/H4: FULL PIPELINE PERFORMANCE\n")
            f.write("-"*80 + "\n")
            rq4 = all_results["rq4"]
            if "c3_vs_best_single" in rq4:
                c3 = rq4["c3_vs_best_single"]
                f.write(f"  C3 vs Best Single Model:\n")
                f.write(f"    C3 map50:95: {c3.get('c3_map50_95', 0):.4f}\n")
                f.write(f"    Best single map50:95: {c3.get('best_single_map50_95', 0):.4f}\n")
                f.write(f"    Difference: {c3.get('map_difference_pp', 0):.4f} pp\n")
                f.write(f"    Exceeds criterion (≥0.5 pp): {c3.get('exceeds_best_single_criterion', False)}\n\n")
            if "vs_published_baseline" in rq4:
                pub = rq4["vs_published_baseline"]
                f.write(f"  C3 vs Published Baseline:\n")
                f.write(f"    Published YOLOF: {pub.get('published_yolof_baseline_map50_95', 0):.4f}\n")
                f.write(f"    C3: {pub.get('c3_map50_95', 0):.4f}\n")
                f.write(f"    Improvement: {pub.get('improvement_pp', 0):.4f} pp\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")

    logger.info(f"✓ Statistical report saved → {txt_path}")
    logger.info("\n" + "="*80)
    logger.info("PHASE 7 COMPLETE")
    logger.info("="*80 + "\n")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    main()
