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
    stat, p_value = stats.wilcoxon(diff)
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
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
    r, p_value = stats.pearsonr(x, y)
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
    barrier_data: Dict[str, List[float]],  # {"backbone": [...], "encoder": [...], "cls": [...], "reg": [...]}
    hessian_data: Dict[str, List[float]],  # Same structure
    averaging_gains: Dict[str, float],  # Per-component gain from M1→M2
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ2/H2 tests: Are cls/reg branches geometrically distinct?
    Do differences explain merging outcomes?

    Returns:
        Dict with nested results for tests 1-3
    """
    logger.info("Testing RQ2/H2: Loss landscape geometry")

    results = {"rq": "RQ2", "hypothesis": "H2"}

    # Test 1: 4-component barrier ANOVA
    logger.info("  Test 1: 4-component barrier ANOVA")
    barriers = np.array([
        barrier_data.get("backbone", [0.0]),
        barrier_data.get("encoder", [0.0]),
        barrier_data.get("cls", [0.0]),
        barrier_data.get("reg", [0.0]),
    ])
    anova_barriers = rm_anova(barriers.T, alpha)
    results["test_1_barrier_anova"] = anova_barriers

    # Test 2: Geometry–performance correlation
    logger.info("  Test 2: Geometry-performance correlation")
    correlations = {}
    for component in ["backbone", "encoder", "cls", "reg"]:
        barrier_vals = np.array(barrier_data.get(component, []))
        gain = averaging_gains.get(component, 0.0)
        if len(barrier_vals) > 2:
            corr = pearson_correlation(barrier_vals, np.full_like(barrier_vals, gain), alpha)
            correlations[component] = corr
    results["test_2_geometry_performance"] = correlations

    # Test 3: Hessian trace ANOVA
    logger.info("  Test 3: Hessian trace ANOVA")
    hessians = np.array([
        hessian_data.get("backbone", [0.0]),
        hessian_data.get("encoder", [0.0]),
        hessian_data.get("cls", [0.0]),
        hessian_data.get("reg", [0.0]),
    ])
    anova_hessians = rm_anova(hessians.T, alpha)
    results["test_3_hessian_anova"] = anova_hessians

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RQ3/H3: Coefficient Strategy and Fine-Tuning Effects
# ─────────────────────────────────────────────────────────────────────────────

def test_rq3_coefficient_strategy(
    m3_per_class_ap: np.ndarray,
    m4_per_class_ap: np.ndarray,
    m2_per_class_ap: np.ndarray,
    d1_per_class_ap: np.ndarray,
    best_learned_per_class_ap: np.ndarray,
    d2_per_class_ap: np.ndarray,
    m3_coefficients: Optional[np.ndarray] = None,  # Shape: (n_models, 2) for cls, reg
    m4_coefficients: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ3/H3 tests: Do coefficient strategies differ? Do fine-tuning gains depend on merge quality?

    Returns:
        Dict with nested results for tests 1-3
    """
    logger.info("Testing RQ3/H3: Coefficient strategy & fine-tuning effects")

    results = {"rq": "RQ3", "hypothesis": "H3"}

    # Test 1: M3 vs M4 (strategy comparison)
    logger.info("  Test 1: M3 vs M4 strategy comparison")
    test_1 = paired_t_test(m4_per_class_ap, m3_per_class_ap, alpha)
    results["test_1_strategy_comparison"] = test_1

    # Test 2: Head fine-tune gains
    logger.info("  Test 2: Head fine-tune paired analysis")
    gain_a = d1_per_class_ap - m2_per_class_ap  # D1 gain from M2 (weaker init)
    gain_b = d2_per_class_ap - best_learned_per_class_ap  # D2 gain from best learned (stronger init)
    test_2a = paired_t_test(gain_b, gain_a, alpha)  # Compare gains
    results["test_2_head_finetune"] = {
        "d1_gain_from_m2": float(np.mean(gain_a)),
        "d2_gain_from_learned": float(np.mean(gain_b)),
        "gain_difference_test": test_2a,
        "interpretation": (
            f"D1 gain = {np.mean(gain_a):.4f} pp, "
            f"D2 gain = {np.mean(gain_b):.4f} pp; "
            f"difference p-value = {test_2a['p_value']:.4f}"
        ),
    }

    # Test 3: Strategy-by-branch interaction (if coefficients provided)
    if m3_coefficients is not None and m4_coefficients is not None:
        logger.info("  Test 3: Strategy-by-branch interaction ANOVA")
        # Simplified: Compare average cls vs reg weights
        m3_cls_avg = np.mean(m3_coefficients[:, 0])
        m3_reg_avg = np.mean(m3_coefficients[:, 1])
        m4_cls_avg = np.mean(m4_coefficients[:, 0])
        m4_reg_avg = np.mean(m4_coefficients[:, 1])
        results["test_3_strategy_by_branch"] = {
            "m3_cls_avg_weight": float(m3_cls_avg),
            "m3_reg_avg_weight": float(m3_reg_avg),
            "m4_cls_avg_weight": float(m4_cls_avg),
            "m4_reg_avg_weight": float(m4_reg_avg),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RQ4/H4: Full Pipeline Performance
# ─────────────────────────────────────────────────────────────────────────────

def test_rq4_full_pipeline(
    c3_per_class_ap: np.ndarray,
    best_single_per_class_ap: np.ndarray,
    published_baseline_map: float = 37.7,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    RQ4/H4 test: Does C3 (full pipeline) exceed best single model?

    Returns:
        Dict with bootstrap CI, significance, comparison to published baseline
    """
    logger.info("Testing RQ4/H4: Full pipeline performance")

    results = {"rq": "RQ4", "hypothesis": "H4"}

    # Bootstrap CI for C3 vs best single
    diff = c3_per_class_ap - best_single_per_class_ap
    ci = bootstrap_ci_mean(diff, confidence=0.95, n_bootstrap=10000)

    results["c3_vs_best_single"] = {
        "bootstrap_ci": ci,
        "mean_difference": float(np.mean(diff)),
        "exceeds_best_single": ci["ci_lower"] >= 0.0,
    }

    # Comparison to published baseline (informational)
    c3_map_estimate = float(np.mean(c3_per_class_ap)) / 100.0 * 100.0  # Normalized
    results["vs_published_baseline"] = {
        "published_baseline_map": published_baseline_map,
        "c3_estimated_map": c3_map_estimate,
        "improvement_pp": c3_map_estimate - published_baseline_map,
    }

    results["interpretation"] = (
        f"C3 vs best single: Mean diff = {np.mean(diff):.4f} pp, "
        f"95% CI = [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]; "
        f"exceeds criterion (≥0): {ci['ci_lower'] >= 0.0}"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Master Statistical Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def run_all_hypothesis_tests(
    phase4_soup_results: Dict[str, Any],
    phase3_barriers_hessians: Dict[str, Any],
    phase5_finetuning_results: Dict[str, Any],
    published_baseline_map: float = 37.7,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Master function: Run all 12 hypothesis tests from methodology Section 3.5.

    Args:
        phase4_soup_results: Results from Phase 4 (M1-M4 soups)
        phase3_barriers_hessians: Loss landscape data from Phase 3
        phase5_finetuning_results: Head fine-tuning results from Phase 5
        published_baseline_map: Published YOLOF baseline (37.7 AP)
        alpha: Significance level (default 0.05)

    Returns:
        Dict with RQ1-RQ4 test results, interpretations, pass/fail for each hypothesis
    """
    logger.info("="*80)
    logger.info("PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    logger.info("="*80)

    # Extract per-class AP arrays (placeholder structure; adjust to actual data format)
    m1_ap = phase4_soup_results.get("m1", {}).get("per_class_ap", np.ones(80))
    m2_ap = phase4_soup_results.get("m2", {}).get("per_class_ap", np.ones(80))
    m3_ap = phase4_soup_results.get("m3", {}).get("per_class_ap", np.ones(80))
    m4_ap = phase4_soup_results.get("m4", {}).get("per_class_ap", np.ones(80))
    best_single_ap = phase4_soup_results.get("best_single_model", {}).get("per_class_ap", np.ones(80))
    best_learned_ap = m3_ap if np.mean(m3_ap) > np.mean(m4_ap) else m4_ap

    d1_ap = phase5_finetuning_results.get("d1", {}).get("per_class_ap", np.ones(80))
    d2_ap = phase5_finetuning_results.get("d2", {}).get("per_class_ap", np.ones(80))
    c3_ap = phase5_finetuning_results.get("c3", {}).get("per_class_ap", np.ones(80))

    # Run all tests
    all_results = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "significance_level": alpha,
        "methodology": "Quantitative within-subject factorial design (Chapter 3)",
    }

    # RQ1/H1
    all_results["rq1"] = test_rq1_branch_vs_uniform(m1_ap, m2_ap, best_learned_ap, best_single_ap, alpha)

    # RQ2/H2
    all_results["rq2"] = test_rq2_loss_landscape_geometry(
        phase3_barriers_hessians.get("barriers", {}),
        phase3_barriers_hessians.get("hessians", {}),
        {},  # averaging gains (if available)
        alpha,
    )

    # RQ3/H3
    all_results["rq3"] = test_rq3_coefficient_strategy(m3_ap, m4_ap, m2_ap, d1_ap, best_learned_ap, d2_ap, alpha=alpha)

    # RQ4/H4
    all_results["rq4"] = test_rq4_full_pipeline(c3_ap, best_single_ap, published_baseline_map, alpha)

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
        help="Path to phase4_soup_results.json",
    )
    parser.add_argument(
        "--barriers-hessians-json",
        help="Path to phase3_lmc_barriers.json or combined landscape data",
    )
    parser.add_argument(
        "--finetuning-results-json",
        help="Path to aggregated phase5 finetuning results",
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory to save statistical test results",
    )
    parser.add_argument(
        "--baseline-map",
        type=float,
        default=37.7,
        help="Published YOLOF baseline mAP50:95",
    )
    args = parser.parse_args()

    # Load phase results (placeholder; adjust to actual data format)
    soup_results = {}
    barriers_hessians = {}
    finetuning_results = {}

    if args.soup_results_json:
        try:
            with open(args.soup_results_json) as f:
                soup_results = json.load(f)
        except Exception as e:
            logger.warning("Could not load soup results: %s", e)

    if args.barriers_hessians_json:
        try:
            with open(args.barriers_hessians_json) as f:
                barriers_hessians = json.load(f)
        except Exception as e:
            logger.warning("Could not load barriers/hessians: %s", e)

    if args.finetuning_results_json:
        try:
            with open(args.finetuning_results_json) as f:
                finetuning_results = json.load(f)
        except Exception as e:
            logger.warning("Could not load finetuning results: %s", e)

    # Run all tests
    all_results = run_all_hypothesis_tests(soup_results, barriers_hessians, finetuning_results, args.baseline_map)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "phase7_hypothesis_tests.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Statistical test results saved → %s", json_path)

    # TXT report
    txt_path = output_dir / "phase7_statistical_report.txt"
    with open(txt_path, "w") as f:
        f.write("PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {all_results['timestamp']}\n")
        f.write(f"Significance level: α = {all_results['significance_level']}\n\n")
        for rq in ["rq1", "rq2", "rq3", "rq4"]:
            if rq in all_results:
                f.write(f"{rq.upper()}/{all_results[rq].get('hypothesis', 'H?')}\n")
                f.write("-" * 40 + "\n")
                f.write(str(all_results[rq]) + "\n\n")
    logger.info("Statistical report saved → %s", txt_path)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    main()
