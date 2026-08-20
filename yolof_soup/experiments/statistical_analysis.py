"""
statistical_analysis.py
=======================

Phase 7: Statistical Analysis & Hypothesis Testing

Implements the finalized thesis hypothesis set from Chapter 1 and the
supporting descriptive analyses from Chapter 3, Section 3.5.

Outputs:
  - phase7_hypothesis_tests.json: test results, p-values, effect sizes, CIs
  - phase7_statistical_report.txt: human-readable interpretation

Run (after Phase 6): python -m yolof_soup.experiments.statistical_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from yolof_soup.utils.global_logger import get_logger


logger = get_logger(logging.DEBUG, add_file_handler=True)

ALPHA = 0.05
N_BOOTSTRAP = 10_000
RNG_SEED = 42
PRACTICAL_MARGIN_PP = 0.5
BETA_NULL = 1.0
BONFERRONI_COUNT = 3
BONFERRONI_ALPHA = ALPHA / BONFERRONI_COUNT
PUBLISHED_BASELINE_MAP = 37.7
N_CLASSES = 80


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def extract_per_class_ap(data_dict: Dict[str, Any], *, field_name: str) -> np.ndarray:
    """Extract an 80-class AP vector from a phase result record."""
    if not isinstance(data_dict, dict):
        raise TypeError(f"{field_name} must be a dict, got {type(data_dict).__name__}.")

    per_class_ap = data_dict.get("per_class_ap")
    if per_class_ap is None:
        raise KeyError(f"{field_name} is missing required key 'per_class_ap'.")

    values: List[float] = []
    if isinstance(per_class_ap, list) and per_class_ap:
        first_item = per_class_ap[0]
        if isinstance(first_item, (list, tuple)):
            for index, entry in enumerate(per_class_ap):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    raise ValueError(
                        f"{field_name}.per_class_ap[{index}] must be [class_name, AP, AR]; got {entry!r}"
                    )
                values.append(float(entry[1]))
        else:
            values = [float(item) for item in per_class_ap]
    else:
        raise ValueError(f"{field_name}.per_class_ap must be a non-empty list.")

    array = np.asarray(values, dtype=float)
    if array.size != N_CLASSES:
        raise ValueError(f"{field_name}.per_class_ap must contain {N_CLASSES} classes, got {array.size}.")
    return array


def extract_map50_95(data_dict: Dict[str, Any], *, field_name: str) -> float:
    """Extract headline mAP50:95, falling back to the mean per-class AP if needed."""
    if not isinstance(data_dict, dict):
        raise TypeError(f"{field_name} must be a dict, got {type(data_dict).__name__}.")

    for key in ("map50_95", "AP"):
        value = data_dict.get(key)
        if value is not None:
            return float(value)

    per_class_ap = data_dict.get("per_class_ap")
    if isinstance(per_class_ap, list) and per_class_ap:
        if isinstance(per_class_ap[0], (list, tuple)):
            values = [float(entry[1]) for entry in per_class_ap if isinstance(entry, (list, tuple)) and len(entry) >= 2]
        else:
            values = [float(item) for item in per_class_ap]
        if values:
            return float(np.mean(values))

    raise KeyError(f"{field_name} does not contain map50_95, AP, or usable per_class_ap values.")


def score_entry(data_dict: Dict[str, Any], *, field_name: str) -> float:
    return extract_map50_95(data_dict, field_name=field_name)


def bootstrap_ci_mean(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> Dict[str, Any]:
    """Bootstrap confidence interval for the mean."""
    array = _as_float_array(values)
    if array.size == 0:
        raise ValueError("bootstrap_ci_mean requires at least one value.")

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(array, size=array.size, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    return {
        "mean": float(array.mean()),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "confidence_level": float(confidence),
        "n_bootstrap": int(n_bootstrap),
    }


def paired_t_ci(diff: np.ndarray, *, confidence: float = 0.95, alternative: str = "two-sided") -> Dict[str, Any]:
    """Analytical t-distribution CI, guaranteed consistent with the paired/one-sample t-test p-value."""
    array = _as_float_array(diff)
    n = array.size
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / np.sqrt(n))
    alpha = 1.0 - confidence

    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        return {"ci_lower": mean - t_crit * sem, "ci_upper": mean + t_crit * sem}
    elif alternative == "greater":
        t_crit = stats.t.ppf(1 - alpha, df=n - 1)
        return {"ci_lower": mean - t_crit * sem, "ci_upper": float("inf")}
    else:
        t_crit = stats.t.ppf(1 - alpha, df=n - 1)
        return {"ci_lower": float("-inf"), "ci_upper": mean + t_crit * sem}


def one_sample_t_from_diff(
    diff: np.ndarray,
    *,
    alpha: float = ALPHA,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """One-sample t-test on a vector of paired differences."""
    array = _as_float_array(diff)
    if array.size < 2:
        raise ValueError("One-sample t-test requires at least 2 paired observations.")

    t_stat, p_value = stats.ttest_1samp(array, 0.0, alternative=alternative)
    ci = paired_t_ci(array, confidence=0.95, alternative=alternative)
    std = array.std(ddof=1)
    return {
        "test": "one_sample_t_test_on_diff",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(array.mean()),
        "std_difference": float(std),
        "cohens_d": float(array.mean() / std) if std > 0 else 0.0,
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
        "significant": bool(p_value < alpha),
    }


def directional_one_sample_t(
    sample: np.ndarray,
    popmean: float,
    *,
    alternative: str = "greater",
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    array = _as_float_array(sample)
    if array.size < 2:
        raise ValueError("One-sample t-test requires at least 2 observations.")

    t_stat, p_value = stats.ttest_1samp(array, popmean, alternative=alternative)
    diff = array - popmean
    ci = paired_t_ci(diff, confidence=0.95, alternative=alternative)
    std = diff.std(ddof=1)
    return {
        "test": "directional_one_sample_t",
        "null_value": float(popmean),
        "alternative": alternative,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_deviation": float(diff.mean()),
        "cohens_d": float(diff.mean() / std) if std > 0 else 0.0,
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
        "significant": bool(p_value < alpha),
    }


def directional_paired_t(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = "greater",
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    x_array = _as_float_array(x)
    y_array = _as_float_array(y)
    if x_array.size != y_array.size:
        raise ValueError("Paired t-test requires arrays of equal length.")
    if x_array.size < 2:
        raise ValueError("Paired t-test requires at least 2 paired observations.")

    diff = x_array - y_array
    t_stat, p_value = stats.ttest_1samp(diff, 0.0, alternative=alternative)
    ci = paired_t_ci(diff, confidence=0.95, alternative=alternative)
    std = diff.std(ddof=1)
    return {
        "test": "directional_paired_t",
        "alternative": alternative,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(diff.mean()),
        "cohens_d": float(diff.mean() / std) if std > 0 else 0.0,
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
        "significant": bool(p_value < alpha),
    }


def two_tailed_paired_t(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    x_array = _as_float_array(x)
    y_array = _as_float_array(y)
    if x_array.size != y_array.size:
        raise ValueError("Paired t-test requires arrays of equal length.")
    if x_array.size < 2:
        raise ValueError("Paired t-test requires at least 2 paired observations.")

    diff = x_array - y_array
    t_stat, p_value = stats.ttest_1samp(diff, 0.0, alternative="two-sided")
    ci = paired_t_ci(diff, confidence=0.95, alternative="two-sided")
    std = diff.std(ddof=1)
    return {
        "test": "paired_t_test",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(diff.mean()),
        "cohens_d": float(diff.mean() / std) if std > 0 else 0.0,
        "ci_lower": float(ci["ci_lower"]),
        "ci_upper": float(ci["ci_upper"]),
        "significant": bool(p_value < alpha),
    }


def greenhouse_geisser_epsilon(data: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon for repeated-measures data."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("greenhouse_geisser_epsilon requires a 2D array with at least 2 conditions.")

    centered = array - array.mean(axis=1, keepdims=True)
    cov = np.cov(centered, rowvar=False, ddof=1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.real(eigvals))[::-1]
    positive = eigvals[eigvals > 1e-12]
    k = array.shape[1]
    p = k - 1
    if positive.size < p:
        positive = eigvals[:p]
    positive = np.asarray(positive[:p], dtype=float)
    if np.any(positive <= 0) or positive.size != p:
        return float(1.0)

    numerator = float(np.square(positive.sum()))
    denominator = float(p * np.square(positive).sum())
    if denominator <= 0:
        return float(1.0)
    epsilon = numerator / denominator
    return float(np.clip(epsilon, 1.0 / p, 1.0))


def mauchly_sphericity_test(data: np.ndarray) -> Dict[str, Any]:
    """Mauchly's sphericity test with a fallback approximation when pingouin is unavailable."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        return {
            "test": "mauchly_sphericity",
            "sphericity": True,
            "w": 1.0,
            "chi2": 0.0,
            "dof": 0,
            "p_value": 1.0,
            "epsilon_gg": 1.0,
            "note": "Sphericity test not applicable for fewer than 3 conditions.",
        }

    try:
        import pingouin as pg  # type: ignore

        df = pd.DataFrame(array)
        spher, W, chi2, dof, pval = pg.sphericity(df, method="mauchly")
        epsilon = pg.epsilon(df, correction="gg")
        return {
            "test": "mauchly_sphericity",
            "sphericity": bool(spher),
            "w": float(W),
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(pval),
            "epsilon_gg": float(epsilon),
            "note": "Computed with pingouin.sphericity.",
        }
    except Exception:
        n, k = array.shape
        centered = array - array.mean(axis=1, keepdims=True)
        cov = np.cov(centered, rowvar=False, ddof=1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.real(eigvals))[::-1]
        positive = eigvals[eigvals > 1e-12]
        p = k - 1
        if positive.size < p:
            positive = eigvals[:p]
        positive = np.asarray(positive[:p], dtype=float)
        if np.any(positive <= 0) or positive.size != p:
            return {
                "test": "mauchly_sphericity",
                "sphericity": False,
                "w": 0.0,
                "chi2": float("inf"),
                "dof": int(p * (p - 1) / 2),
                "p_value": 0.0,
                "epsilon_gg": float(1.0 / p),
                "note": "Fallback Mauchly approximation failed due to non-positive eigenvalues.",
            }

        W = float(np.prod(positive) / (np.mean(positive) ** p))
        W = max(W, 1e-300)
        correction = 1.0 - ((2 * p * p + p + 2) / (6.0 * p * max(n - 1, 1)))
        correction = max(correction, 1e-12)
        chi2 = -(n - 1) * correction * np.log(W)
        dof = int(p * (p - 1) / 2)
        p_value = float(stats.chi2.sf(chi2, dof)) if np.isfinite(chi2) else 0.0
        epsilon = greenhouse_geisser_epsilon(array)
        return {
            "test": "mauchly_sphericity",
            "sphericity": bool(p_value >= ALPHA),
            "w": float(W),
            "chi2": float(chi2),
            "dof": dof,
            "p_value": p_value,
            "epsilon_gg": float(epsilon),
            "note": "Fallback Mauchly approximation used because pingouin is unavailable.",
        }


def repeated_measures_anova(data: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    """Repeated-measures ANOVA with Greenhouse-Geisser correction when sphericity is violated."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(labels):
        raise ValueError("repeated_measures_anova expects data shaped (subjects, conditions).")

    records: List[Dict[str, Any]] = []
    for subject_index in range(array.shape[0]):
        for condition_index, condition_name in enumerate(labels):
            records.append(
                {
                    "subject": subject_index,
                    "condition": condition_name,
                    "ap": float(array[subject_index, condition_index]),
                }
            )

    df_long = pd.DataFrame(records)
    anova = AnovaRM(df_long, depvar="ap", subject="subject", within=["condition"]).fit()
    table = anova.anova_table
    row = table.loc["condition"] if "condition" in table.index else table.iloc[0]

    F_value = float(row["F Value"])
    df_num = float(row["Num DF"])
    df_den = float(row["Den DF"])
    p_value = float(row["Pr > F"])

    sphericity = mauchly_sphericity_test(array)
    epsilon = float(sphericity.get("epsilon_gg", 1.0))
    violated = not bool(sphericity.get("sphericity", True))

    if violated:
        corrected_df_num = df_num * epsilon
        corrected_df_den = df_den * epsilon
        corrected_p_value = float(stats.f.sf(F_value, corrected_df_num, corrected_df_den))
        corrected = True
    else:
        corrected_df_num = df_num
        corrected_df_den = df_den
        corrected_p_value = p_value
        corrected = False

    anova_table = {
        str(index): {
            key: (float(value) if hasattr(value, "item") else value)
            for key, value in table.loc[index].to_dict().items()
        }
        for index in table.index
    }

    return {
        "test": "repeated_measures_anova",
        "f_statistic": F_value,
        "df_num": df_num,
        "df_den": df_den,
        "p_value": p_value,
        "corrected": corrected,
        "corrected_df_num": float(corrected_df_num),
        "corrected_df_den": float(corrected_df_den),
        "corrected_p_value": float(corrected_p_value),
        "epsilon_gg": epsilon,
        "sphericity_test": sphericity,
        "anova_table": anova_table,
    }


def tukey_hsd_summary(data: np.ndarray, labels: Sequence[str], alpha: float = ALPHA) -> Dict[str, Any]:
    """Tukey HSD post-hoc summary table, computed on within-subject-demeaned residuals."""
    array = np.asarray(data, dtype=float)
    subject_means = array.mean(axis=1, keepdims=True)
    grand_mean = array.mean()
    residuals = array - subject_means + grand_mean  # remove subject effect, keep condition effect

    long_values: List[float] = []
    group_labels: List[str] = []
    for condition_index, condition_name in enumerate(labels):
        condition_values = residuals[:, condition_index]
        long_values.extend(condition_values.tolist())
        group_labels.extend([condition_name] * len(condition_values))

    tukey = pairwise_tukeyhsd(endog=np.asarray(long_values, dtype=float), groups=np.asarray(group_labels), alpha=alpha)
    summary = tukey.summary()
    summary_rows = [list(row) for row in summary.data[1:]]
    headers = list(summary.data[0])
    return {
        "test": "tukey_hsd",
        "alpha": alpha,
        "summary_text": str(summary),
        "headers": headers,
        "rows": summary_rows,
        "note": "Computed on within-subject-demeaned residuals to match the repeated-measures design used in the omnibus ANOVA.",
    }


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def first_existing_path(candidates: Sequence[str | Path]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_json_file(path: Path) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def normalize_condition_key_map(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)
    for key in ("D1", "D2", "C3"):
        lower_key = key.lower()
        if key in normalized and lower_key not in normalized:
            normalized[lower_key] = normalized[key]
    return normalized


def load_best_single_model(
    soup_results: Dict[str, Any],
    ingredient_results: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Pick the best individual pool model from ingredient results, falling back to phase-3 metadata."""
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(ingredient_results, dict) and ingredient_results:
        for key, value in ingredient_results.items():
            if isinstance(value, dict) and ("per_class_ap" in value or "map50_95" in value or "AP" in value):
                candidates.append((key, value))

    if candidates:
        best_key, best_entry = max(candidates, key=lambda item: score_entry(item[1], field_name=f"ingredient:{item[0]}"))
        return best_entry, best_key

    fallback = soup_results.get("best_single_model")
    if isinstance(fallback, dict):
        return fallback, "best_single_model"

    raise KeyError("Could not resolve the best single model from ingredient results or phase-3 soup metadata.")


def _pair_component_aliases(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    cls_value = None
    reg_value = None
    for key in ("cls", "cls_head", "backbone_cls", "component_cls"):
        if key in record:
            cls_value = float(record[key])
            break
    for key in ("reg", "reg_head", "bbox", "bbox_head", "component_reg"):
        if key in record:
            reg_value = float(record[key])
            break
    return cls_value, reg_value


def aggregate_cls_reg_barriers(barrier_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return paired cls/reg barriers across all available model pairs."""
    pair_to_cls: Dict[str, List[float]] = {}
    pair_to_reg: Dict[str, List[float]] = {}

    def add_record(pair_name: str, record: Dict[str, Any]) -> None:
        cls_value, reg_value = _pair_component_aliases(record)
        if cls_value is None or reg_value is None:
            return
        pair_to_cls.setdefault(pair_name, []).append(cls_value)
        pair_to_reg.setdefault(pair_name, []).append(reg_value)

    if barrier_data:
        top_level_values = list(barrier_data.values())
        if top_level_values and all(isinstance(value, dict) for value in top_level_values):
            nested = any(
                value and all(isinstance(inner, dict) for inner in value.values())
                for value in top_level_values
            )
            if nested:
                for base_group in barrier_data.values():
                    for pair_name, record in base_group.items():
                        if isinstance(record, dict):
                            add_record(pair_name, record)
            else:
                for pair_name, record in barrier_data.items():
                    if isinstance(record, dict):
                        add_record(pair_name, record)

    pair_names = sorted(set(pair_to_cls) & set(pair_to_reg))
    if not pair_names:
        raise KeyError("No paired cls/reg barrier values could be extracted from barrier data.")

    cls_values = np.array([np.mean(pair_to_cls[pair]) for pair in pair_names], dtype=float)
    reg_values = np.array([np.mean(pair_to_reg[pair]) for pair in pair_names], dtype=float)
    return cls_values, reg_values, pair_names


def load_condition_data(results: Dict[str, Any], key: str, *, field_name: str) -> Tuple[float, np.ndarray]:
    if key not in results:
        raise KeyError(f"Missing required key '{key}' in {field_name}.")
    record = results[key]
    return extract_map50_95(record, field_name=f"{field_name}.{key}"), extract_per_class_ap(record, field_name=f"{field_name}.{key}")


# -----------------------------------------------------------------------------
# Hypothesis tests
# -----------------------------------------------------------------------------

def test_rq1_branch_specific_vs_uniform(
    condition_1_map: float,
    condition_1_ap: np.ndarray,
    condition_6_map: float,
    condition_6_ap: np.ndarray,
    best_single_map: float,
    best_single_ap: np.ndarray,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """RQ1/H1: M6 vs Condition 1 and M6 vs best single pool model."""
    logger.info("Testing RQ1/H1: best tri-component learned soup against baseline and pool best")

    diff_vs_c1 = condition_6_ap - condition_1_ap
    diff_vs_best_single = condition_6_ap - best_single_ap

    test_c1 = one_sample_t_from_diff(diff_vs_c1, alpha=alpha, alternative="two-sided")
    test_best = one_sample_t_from_diff(diff_vs_best_single, alpha=alpha, alternative="two-sided")

    h1_support_c1 = bool(test_c1["ci_lower"] >= 0.0 and test_c1["mean_difference"] >= PRACTICAL_MARGIN_PP)
    h1_support_best = bool(test_best["ci_lower"] >= 0.0 and test_best["mean_difference"] >= PRACTICAL_MARGIN_PP)

    return {
        "rq": "RQ1",
        "hypothesis": "H1",
        "best_tri_component_condition": "condition_6",
        "comparison_a": {
            "comparison": "M6 vs Condition 1",
            "headline_map_difference_pp": float(condition_6_map - condition_1_map),
            "test_result": test_c1,
            "decision_criterion": {
                "ci_lower_ge_0": bool(test_c1["ci_lower"] >= 0.0),
                "mean_difference_ge_0_5_pp": bool(test_c1["mean_difference"] >= PRACTICAL_MARGIN_PP),
                "supported": h1_support_c1,
            },
            "interpretation": (
                f"M6 vs Condition 1: headline ΔmAP = {condition_6_map - condition_1_map:.4f} pp, "
                f"mean per-class Δ = {test_c1['mean_difference']:.4f} pp, p = {test_c1['p_value']:.4f}"
            ),
        },
        "comparison_b": {
            "comparison": "M6 vs best single pool model",
            "best_single_map50_95": float(best_single_map),
            "headline_map_difference_pp": float(condition_6_map - best_single_map),
            "best_single_source": None,
            "test_result": test_best,
            "decision_criterion": {
                "ci_lower_ge_0": bool(test_best["ci_lower"] >= 0.0),
                "mean_difference_ge_0_5_pp": bool(test_best["mean_difference"] >= PRACTICAL_MARGIN_PP),
                "supported": h1_support_best,
            },
            "interpretation": (
                f"M6 vs best single: headline ΔmAP = {condition_6_map - best_single_map:.4f} pp, "
                f"mean per-class Δ = {test_best['mean_difference']:.4f} pp, p = {test_best['p_value']:.4f}"
            ),
        },
    }


def descriptive_loss_landscape_geometry(
    barrier_data: Dict[str, Any],
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """Descriptive paired two-tailed t-test for B_cls vs B_reg across model pairs."""
    logger.info("Descriptive analysis: paired cls/reg barrier comparison")

    cls_values, reg_values, pair_names = aggregate_cls_reg_barriers(barrier_data)
    diff = cls_values - reg_values
    test = two_tailed_paired_t(cls_values, reg_values, alpha=alpha)

    return {
        "analysis": "descriptive_geometry",
        "n_pairs": int(len(pair_names)),
        "pairs_used": pair_names,
        "sphericity_correction_applicable": False,
        "test_name": "paired_two_tailed_t_test",
        "cls_minus_reg": {
            "mean_difference": float(diff.mean()),
            "std_difference": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
            "test_result": test,
            "interpretation": (
                f"B_cls vs B_reg across {len(pair_names)} pairs: mean Δ = {diff.mean():.4f}, "
                f"observed p-value = {test['p_value']:.4f}. Sphericity correction is not applicable for a two-group paired comparison."
            ),
        },
    }


def test_rq2_weighting_strategies(
    condition_2_ap: np.ndarray,
    condition_3_ap: np.ndarray,
    condition_4_ap: np.ndarray,
    condition_5_ap: np.ndarray,
    condition_6_ap: np.ndarray,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """RQ2/H2: repeated-measures ANOVA across Conditions 2-6 with Tukey HSD."""
    logger.info("Testing RQ2/H2: repeated-measures ANOVA across weighting strategies")

    labels = ["Condition 2", "Condition 3", "Condition 4", "Condition 5", "Condition 6"]
    data = np.column_stack([condition_2_ap, condition_3_ap, condition_4_ap, condition_5_ap, condition_6_ap])

    anova = repeated_measures_anova(data, labels)
    sphericity = anova["sphericity_test"]
    corrected_p = anova["corrected_p_value"]
    omnibus_significant = bool(corrected_p < alpha)

    tukey_result = None
    if omnibus_significant:
        tukey_result = tukey_hsd_summary(data, labels, alpha=alpha)

    pairwise_compound_decisions: List[Dict[str, Any]] = []
    compound_reject = False
    if tukey_result is not None:
        for row in tukey_result["rows"]:
            group_a, group_b, meandiff, p_adj, lower, upper, reject = row
            meets_magnitude = bool(abs(float(meandiff)) >= PRACTICAL_MARGIN_PP)
            meets_pvalue = bool(float(p_adj) < alpha)
            meets_both = bool(meets_magnitude and meets_pvalue)
            compound_reject = compound_reject or meets_both
            pairwise_compound_decisions.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "mean_difference": float(meandiff),
                    "abs_mean_difference_ge_0_5_pp": meets_magnitude,
                    "p_adj_lt_0_05": meets_pvalue,
                    "meets_compound_criterion": meets_both,
                    "lower_ci": float(lower),
                    "upper_ci": float(upper),
                    "tukey_reject": bool(reject),
                }
            )

    overall_verdict = "H2 null rejected" if compound_reject else "H2 null retained"

    return {
        "rq": "RQ3",
        "hypothesis": "H2",
        "conditions": labels,
        "omnibus_anova": anova,
        "sphericity_result": sphericity,
        "greenhouse_geisser_applied": bool(anova["corrected"]),
        "omnibus_significant": omnibus_significant,
        "tukey_hsd": tukey_result,
        "pairwise_compound_decisions": pairwise_compound_decisions,
        "overall_verdict": overall_verdict,
        "interpretation": (
            f"RM-ANOVA across conditions 2-6: F = {anova['f_statistic']:.4f}, "
            f"p = {corrected_p:.4f}, epsilon_GG = {anova['epsilon_gg']:.4f}; "
            f"sphericity violated = {not bool(sphericity.get('sphericity', True))}."
        ),
    }


def test_rq3_m6_vs_m5(
    condition_5_map: float,
    condition_5_ap: np.ndarray,
    condition_6_map: float,
    condition_6_ap: np.ndarray,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """RQ3/H3: directional paired t-test M6 vs M5."""
    logger.info("Testing RQ3/H3: M6 vs M5 directional paired t-test")

    test = directional_paired_t(condition_6_ap, condition_5_ap, alternative="greater", alpha=alpha)
    observed_direction = "M6 > M5" if test["mean_difference"] >= 0 else "M5 > M6"

    return {
        "rq": "RQ3",
        "hypothesis": "H3",
        "comparison": "M6 vs M5",
        "condition_6_map50_95": float(condition_6_map),
        "condition_5_map50_95": float(condition_5_map),
        "headline_map_difference_pp": float(condition_6_map - condition_5_map),
        "observed_direction": observed_direction,
        "test_result": test,
        "interpretation": (
            f"M6 vs M5: headline ΔmAP = {condition_6_map - condition_5_map:.4f} pp, "
            f"mean per-class Δ = {test['mean_difference']:.4f} pp, p(one-sided) = {test['p_value']:.4f}, "
            f"direction = {observed_direction}."
        ),
    }


def _beta_series_from_value(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, dict):
        collected: List[np.ndarray] = []
        for key in ("values", "replicates", "seeds", "samples", "runs", "history"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                collected.append(np.asarray(candidate, dtype=float).ravel())
        for key in ("cls", "bbox", "obj"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                collected.append(np.asarray(candidate, dtype=float).ravel())
            elif candidate is not None:
                collected.append(np.asarray([candidate], dtype=float))
        if collected:
            merged = np.concatenate([arr[np.isfinite(arr)] for arr in collected if arr.size > 0])
            return merged
        return np.asarray([], dtype=float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float).ravel()
    return np.asarray([value], dtype=float)


def _extract_beta_block(condition_6: Dict[str, Any]) -> Dict[str, np.ndarray]:
    beta_block = None
    for key in ("beta_values", "betas", "beta_parameters", "temperature_values", "calibration_betas", "beta_replicates"):
        if key in condition_6:
            beta_block = condition_6[key]
            break

    if beta_block is None:
        beta_block = {
            "cls": condition_6.get("beta_cls"),
            "bbox": condition_6.get("beta_bbox"),
            "obj": condition_6.get("beta_obj"),
            "cls_replicates": condition_6.get("beta_cls_replicates"),
            "bbox_replicates": condition_6.get("beta_bbox_replicates"),
            "obj_replicates": condition_6.get("beta_obj_replicates"),
        }

    if isinstance(beta_block, list):
        def gather(component: str) -> np.ndarray:
            values: List[float] = []
            for item in beta_block:
                if isinstance(item, dict):
                    raw = item.get(component)
                    values.extend(_beta_series_from_value(raw).tolist())
            return np.asarray(values, dtype=float)

        return {"cls": gather("cls"), "bbox": gather("bbox"), "obj": gather("obj")}

    return {
        "cls": _beta_series_from_value(beta_block.get("cls") if isinstance(beta_block, dict) else beta_block),
        "bbox": _beta_series_from_value(beta_block.get("bbox") if isinstance(beta_block, dict) else beta_block),
        "obj": _beta_series_from_value(beta_block.get("obj") if isinstance(beta_block, dict) else beta_block),
    }


def test_rq4b_beta_deviation(
    condition_6: Dict[str, Any],
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """Descriptive analysis: one-sample t-tests for beta_cls, beta_bbox, beta_obj against 1.0."""
    logger.info("Descriptive analysis: beta deviation from 1.0 with Bonferroni correction")

    beta_series = _extract_beta_block(condition_6)
    results: Dict[str, Any] = {
        "analysis": "descriptive_calibration",
        "null_value": BETA_NULL,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "tests": {},
    }

    for name, series in beta_series.items():
        if series.size < 2:
            results["tests"][name] = {
                "beta_values": series.tolist(),
                "note": "Insufficient replicate values for a one-sample t-test after searching calibration replicates.",
            }
            continue
        test = directional_one_sample_t(series, BETA_NULL, alternative="two-sided", alpha=BONFERRONI_ALPHA)
        test["bonferroni_significant"] = bool(test["p_value"] < BONFERRONI_ALPHA)
        test["mean_value"] = float(series.mean())
        test["mean_deviation"] = float(series.mean() - BETA_NULL)
        results["tests"][name] = test

    return results


def test_rq4_full_pipeline(
    c3_map: float,
    c3_ap: np.ndarray,
    best_single_map: float,
    best_single_ap: np.ndarray,
    published_baseline_per_class_ap: Optional[np.ndarray] = None,
    published_baseline_map: float = PUBLISHED_BASELINE_MAP,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """RQ4/H4: full pipeline comparisons at headline and per-class level."""
    logger.info("Testing RQ4/H4: corrected C3 comparisons")

    if np.allclose(c3_ap, 0.0) and abs(c3_map) < 1e-12:
        raise ValueError(
            "C3 is still all zeros after loading. Check the phase-5 evaluation output, checkpoint path, and eval config."
        )

    best_single_test = directional_paired_t(c3_ap, best_single_ap, alternative="greater", alpha=alpha)
    headline_baseline_test = directional_one_sample_t(c3_ap, published_baseline_map, alternative="greater", alpha=alpha)

    if published_baseline_per_class_ap is not None and published_baseline_per_class_ap.size == c3_ap.size:
        per_class_baseline_test = directional_paired_t(c3_ap, published_baseline_per_class_ap, alternative="greater", alpha=alpha)
        per_class_baseline_available = True
    else:
        per_class_baseline_test = None
        per_class_baseline_available = False

    return {
        "rq": "RQ4",
        "hypothesis": "H4",
        "c3_map50_95": float(c3_map),
        "best_single_map50_95": float(best_single_map),
        "published_baseline_map50_95": float(published_baseline_map),
        "headline_comparison_to_published_baseline": {
            "headline_map_difference_pp": float(c3_map - published_baseline_map),
            "test_result": headline_baseline_test,
            "practical_margin_met": bool(headline_baseline_test["ci_lower"] >= PRACTICAL_MARGIN_PP),
            "interpretation": (
                f"C3 vs published baseline: mean deviation = {headline_baseline_test['mean_deviation']:.4f} pp, "
                f"one-sided p = {headline_baseline_test['p_value']:.4f}, practical margin >= {PRACTICAL_MARGIN_PP:.1f} pp = {headline_baseline_test['ci_lower'] >= PRACTICAL_MARGIN_PP}."
            ),
        },
        "headline_comparison_to_best_single": {
            "headline_map_difference_pp": float(c3_map - best_single_map),
            "test_result": best_single_test,
            "practical_margin_met": bool(best_single_test["ci_lower"] >= PRACTICAL_MARGIN_PP),
            "interpretation": (
                f"C3 vs best single: mean per-class Δ = {best_single_test['mean_difference']:.4f} pp, "
                f"one-sided p = {best_single_test['p_value']:.4f}, practical margin >= {PRACTICAL_MARGIN_PP:.1f} pp = {best_single_test['ci_lower'] >= PRACTICAL_MARGIN_PP}."
            ),
        },
        "per_class_comparison_to_best_single": {
            "available": True,
            "test_result": best_single_test,
            "practical_margin_met": bool(best_single_test["ci_lower"] >= PRACTICAL_MARGIN_PP),
        },
        "per_class_comparison_to_published_baseline": {
            "available": per_class_baseline_available,
            "test_result": per_class_baseline_test,
            "practical_margin_met": bool(per_class_baseline_test["ci_lower"] >= PRACTICAL_MARGIN_PP) if per_class_baseline_test else None,
        },
    }


# -----------------------------------------------------------------------------
# Master runner
# -----------------------------------------------------------------------------

def run_all_hypothesis_tests(
    soup_results: Dict[str, Any],
    ingredient_results: Optional[Dict[str, Any]] = None,
    barriers_hessians: Optional[Dict[str, Any]] = None,
    finetuning_results: Optional[Dict[str, Any]] = None,
    *,
    source_paths: Optional[Dict[str, str]] = None,
    published_baseline_map: float = PUBLISHED_BASELINE_MAP,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    logger.info("=" * 80)

    if barriers_hessians is None:
        barriers_hessians = {}
    if finetuning_results is None:
        finetuning_results = {}
    if source_paths is None:
        source_paths = {}

    soup_results = normalize_condition_key_map(dict(soup_results))

    condition_1_map, condition_1_ap = load_condition_data(soup_results, "condition_1", field_name="soup_results")
    condition_2_map, condition_2_ap = load_condition_data(soup_results, "condition_2", field_name="soup_results")
    condition_3_map, condition_3_ap = load_condition_data(soup_results, "condition_3", field_name="soup_results")
    condition_4_map, condition_4_ap = load_condition_data(soup_results, "condition_4", field_name="soup_results")
    condition_5_map, condition_5_ap = load_condition_data(soup_results, "condition_5", field_name="soup_results")
    condition_6_map, condition_6_ap = load_condition_data(soup_results, "condition_6", field_name="soup_results")

    if ingredient_results is None:
        ingredient_results = {}

    best_single_entry, best_single_source = load_best_single_model(soup_results, ingredient_results)
    best_single_map = extract_map50_95(best_single_entry, field_name=f"best_single:{best_single_source}")
    best_single_ap = extract_per_class_ap(best_single_entry, field_name=f"best_single:{best_single_source}")

    barrier_data = barriers_hessians.get("barriers", {})
    hessian_data = barriers_hessians.get("hessians", {})

    c3_record = finetuning_results.get("c3", finetuning_results.get("C3", {}))
    d1_record = finetuning_results.get("d1", finetuning_results.get("D1", {}))
    d2_record = finetuning_results.get("d2", finetuning_results.get("D2", {}))

    c3_map = extract_map50_95(c3_record, field_name="finetuning_results.c3")
    c3_ap = extract_per_class_ap(c3_record, field_name="finetuning_results.c3")

    results: Dict[str, Any] = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "significance_level": alpha,
        "published_baseline_map50_95": published_baseline_map,
        "source_paths": source_paths,
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
        "data_summary": {
            "condition_1_map50_95": float(condition_1_map),
            "condition_2_map50_95": float(condition_2_map),
            "condition_3_map50_95": float(condition_3_map),
            "condition_4_map50_95": float(condition_4_map),
            "condition_5_map50_95": float(condition_5_map),
            "condition_6_map50_95": float(condition_6_map),
            "best_single_map50_95": float(best_single_map),
            "c3_map50_95": float(c3_map),
            "d1_map50_95": float(extract_map50_95(d1_record, field_name="finetuning_results.d1")) if d1_record else None,
            "d2_map50_95": float(extract_map50_95(d2_record, field_name="finetuning_results.d2")) if d2_record else None,
        },
    }

    logger.info("Loaded phase outputs")
    logger.info("  M1 map50:95 = %.4f", condition_1_map)
    logger.info("  M2 map50:95 = %.4f", condition_2_map)
    logger.info("  M6 map50:95 = %.4f", condition_6_map)
    logger.info("  Best single map50:95 = %.4f (%s)", best_single_map, best_single_source)
    logger.info("  C3 map50:95 = %.4f", c3_map)

    results["rq1"] = test_rq1_branch_specific_vs_uniform(
        condition_1_map,
        condition_1_ap,
        condition_6_map,
        condition_6_ap,
        best_single_map,
        best_single_ap,
        alpha=alpha,
    )

    results["descriptive_geometry"] = descriptive_loss_landscape_geometry(barrier_data, alpha=alpha)
    results["descriptive_geometry"]["hessian_data_present"] = bool(hessian_data)

    results["rq2"] = test_rq2_weighting_strategies(
        condition_2_ap,
        condition_3_ap,
        condition_4_ap,
        condition_5_ap,
        condition_6_ap,
        alpha=alpha,
    )

    results["rq3"] = test_rq3_m6_vs_m5(condition_5_map, condition_5_ap, condition_6_map, condition_6_ap, alpha=alpha)
    results["descriptive_calibration"] = test_rq4b_beta_deviation(soup_results.get("condition_6", {}), alpha=alpha)
    results["rq4"] = test_rq4_full_pipeline(
        c3_map,
        c3_ap,
        best_single_map,
        best_single_ap,
        published_baseline_per_class_ap=None,
        published_baseline_map=published_baseline_map,
        alpha=alpha,
    )

    results["notes"] = {
        "rq2_sphericity_correction_applicable": False,
        "c3_checkpoint_path": c3_record.get("checkpoint") if isinstance(c3_record, dict) else None,
        "best_single_source": best_single_source,
    }

    logger.info("=" * 80)
    logger.info("ALL HYPOTHESIS TESTS COMPLETE")
    logger.info("=" * 80)
    return results


# -----------------------------------------------------------------------------
# Report writer
# -----------------------------------------------------------------------------

def _write_line(handle, text: str = "") -> None:
    handle.write(text + "\n")


def write_statistical_report(results: Dict[str, Any], output_path: Path) -> None:
    with open(output_path, "w") as handle:
        _write_line(handle, "PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TEST RESULTS")
        _write_line(handle, "=" * 80)
        _write_line(handle, f"Generated: {results['timestamp']}")
        _write_line(handle, f"Significance level (α): {results['significance_level']}")
        _write_line(handle, f"Methodology: {results['methodology']}")
        _write_line(handle)
        _write_line(handle, "=" * 80)
        _write_line(handle, "CONDITIONS MAPPING")
        _write_line(handle, "=" * 80)
        for condition, description in results.get("conditions_mapping", {}).items():
            _write_line(handle, f"  {condition:12s} -> {description}")
        _write_line(handle)
        _write_line(handle, "=" * 80)

        rq1_data = results.get("rq1", {})
        rq2_data = results.get("rq2", {})
        rq3_data = results.get("rq3", {})
        rq4_data = results.get("rq4", {})
        descriptive_geometry_data = results.get("descriptive_geometry", {})
        descriptive_calibration_data = results.get("descriptive_calibration", {})

        # RQ1/H1
        _write_line(handle, "RQ1/H1: BRANCH-SPECIFIC AVERAGING VS UNIFORM BASELINES")
        _write_line(handle, "-" * 80)
        if rq1_data:
            comp_a = rq1_data.get("comparison_a", {})
            comp_b = rq1_data.get("comparison_b", {})
            if comp_a:
                tr = comp_a.get("test_result", {})
                dc = comp_a.get("decision_criterion", {})
                _write_line(handle, "  Test A (M6 vs Condition 1):")
                _write_line(handle, f"    Headline mAP difference: {comp_a.get('headline_map_difference_pp', 0.0):.4f} pp")
                _write_line(handle, f"    Mean difference: {tr.get('mean_difference', 0.0):.4f} pp")
                _write_line(handle, f"    p-value: {tr.get('p_value', 0.0):.4f}")
                _write_line(handle, f"    Cohen's d: {tr.get('cohens_d', 0.0):.4f}")
                _write_line(handle, f"    95% CI: [{tr.get('ci_lower', 0.0):.4f}, {tr.get('ci_upper', 0.0):.4f}]")
                _write_line(handle, f"    Decision criterion (CI lower >= 0 AND Δ >= 0.5 pp): {dc.get('supported', False)}")
                _write_line(handle)
            if comp_b:
                tr = comp_b.get("test_result", {})
                dc = comp_b.get("decision_criterion", {})
                _write_line(handle, "  Test B (M6 vs best single pool model):")
                _write_line(handle, f"    Headline mAP difference: {comp_b.get('headline_map_difference_pp', 0.0):.4f} pp")
                _write_line(handle, f"    Mean difference: {tr.get('mean_difference', 0.0):.4f} pp")
                _write_line(handle, f"    p-value: {tr.get('p_value', 0.0):.4f}")
                _write_line(handle, f"    Cohen's d: {tr.get('cohens_d', 0.0):.4f}")
                _write_line(handle, f"    95% CI: [{tr.get('ci_lower', 0.0):.4f}, {tr.get('ci_upper', 0.0):.4f}]")
                _write_line(handle, f"    Decision criterion (CI lower >= 0 AND Δ >= 0.5 pp): {dc.get('supported', False)}")
                _write_line(handle)

        # RQ2/H2
        _write_line(handle, "RQ2/H2: STRATEGY EQUIVALENCE ACROSS CONDITIONS 2-6")
        _write_line(handle, "-" * 80)
        if rq2_data:
            anova = rq2_data.get("omnibus_anova", {})
            spher = rq2_data.get("sphericity_result", {})
            _write_line(handle, "  Omnibus repeated-measures ANOVA (Conditions 2-6):")
            _write_line(handle, f"    F: {anova.get('f_statistic', 0.0):.4f}")
            _write_line(handle, f"    Uncorrected df: ({anova.get('df_num', 0.0):.4f}, {anova.get('df_den', 0.0):.4f})")
            if anova.get("corrected", False):
                _write_line(handle, f"    Greenhouse-Geisser corrected df: ({anova.get('corrected_df_num', 0.0):.4f}, {anova.get('corrected_df_den', 0.0):.4f})")
                _write_line(handle, f"    Corrected p-value: {anova.get('corrected_p_value', 0.0):.4f}")
            else:
                _write_line(handle, f"    p-value: {anova.get('p_value', 0.0):.4f}")
            _write_line(handle, f"    Mauchly W: {spher.get('w', 1.0):.4f}")
            _write_line(handle, f"    Sphericity p-value: {spher.get('p_value', 1.0):.4f}")
            _write_line(handle, f"    Greenhouse-Geisser epsilon: {anova.get('epsilon_gg', 1.0):.4f}")
            _write_line(handle, f"    Sphericity violated: {not bool(spher.get('sphericity', True))}")
            _write_line(handle)
            if rq2_data.get("tukey_hsd"):
                _write_line(handle, "  Tukey HSD post-hoc summary:")
                _write_line(handle, rq2_data["tukey_hsd"]["summary_text"])
                _write_line(handle)
            _write_line(handle, f"  Compound verdict: {rq2_data.get('overall_verdict', 'n/a')}")
            if rq2_data.get("pairwise_compound_decisions"):
                _write_line(handle, "  Pairwise compound criteria:")
                for item in rq2_data["pairwise_compound_decisions"]:
                    _write_line(handle, f"    {item['group_a']} vs {item['group_b']}: |Δ|>=0.5 pp = {item['abs_mean_difference_ge_0_5_pp']}, p-adj<0.05 = {item['p_adj_lt_0_05']}, both = {item['meets_compound_criterion']}")
                _write_line(handle)

        # RQ3/H3
        _write_line(handle, "RQ3/H3: M6 VS M5")
        _write_line(handle, "-" * 80)
        if rq3_data:
            tr = rq3_data.get("test_result", {})
            _write_line(handle, f"  Observed direction: {rq3_data.get('observed_direction', 'n/a')}")
            _write_line(handle, f"  Mean difference: {tr.get('mean_difference', 0.0):.4f} pp")
            _write_line(handle, f"  p-value: {tr.get('p_value', 0.0):.4f}")
            _write_line(handle, f"  Cohen's d: {tr.get('cohens_d', 0.0):.4f}")
            _write_line(handle, f"  95% CI: [{tr.get('ci_lower', 0.0):.4f}, {tr.get('ci_upper', 0.0):.4f}]")
            _write_line(handle)

        # RQ4/H4
        _write_line(handle, "RQ4/H4: FULL PIPELINE PERFORMANCE")
        _write_line(handle, "-" * 80)
        if rq4_data:
            base = rq4_data.get("headline_comparison_to_published_baseline", {})
            best = rq4_data.get("headline_comparison_to_best_single", {})
            per_class_best = rq4_data.get("per_class_comparison_to_best_single", {})
            per_class_base = rq4_data.get("per_class_comparison_to_published_baseline", {})
            tr_base = base.get("test_result", {})
            tr_best = best.get("test_result", {})
            _write_line(handle, "  Headline comparison to published YOLOF baseline:")
            _write_line(handle, f"    Headline mAP difference: {base.get('headline_map_difference_pp', 0.0):.4f} pp")
            _write_line(handle, f"    Mean deviation: {tr_base.get('mean_deviation', 0.0):.4f} pp")
            _write_line(handle, f"    p-value: {tr_base.get('p_value', 0.0):.4f}")
            _write_line(handle, f"    Cohen's d: {tr_base.get('cohens_d', 0.0):.4f}")
            _write_line(handle, f"    95% CI: [{tr_base.get('ci_lower', 0.0):.4f}, {tr_base.get('ci_upper', 0.0):.4f}]")
            _write_line(handle, f"    Practical margin >= {PRACTICAL_MARGIN_PP:.1f} pp: {base.get('practical_margin_met', False)}")
            _write_line(handle)
            _write_line(handle, "  Headline comparison to best individual pool model:")
            _write_line(handle, f"    Headline mAP difference: {best.get('headline_map_difference_pp', 0.0):.4f} pp")
            _write_line(handle, f"    Mean difference: {tr_best.get('mean_difference', 0.0):.4f} pp")
            _write_line(handle, f"    p-value: {tr_best.get('p_value', 0.0):.4f}")
            _write_line(handle, f"    Cohen's d: {tr_best.get('cohens_d', 0.0):.4f}")
            _write_line(handle, f"    95% CI: [{tr_best.get('ci_lower', 0.0):.4f}, {tr_best.get('ci_upper', 0.0):.4f}]")
            _write_line(handle, f"    Practical margin >= {PRACTICAL_MARGIN_PP:.1f} pp: {best.get('practical_margin_met', False)}")
            _write_line(handle)
            _write_line(handle, "  Per-class comparison to best individual pool model:")
            _write_line(handle, f"    Available: {per_class_best.get('available', False)}")
            if per_class_best.get('available', False) and per_class_best.get('test_result'):
                tr_pc_best = per_class_best['test_result']
                _write_line(handle, f"    Mean per-class difference: {tr_pc_best.get('mean_difference', 0.0):.4f} pp")
                _write_line(handle, f"    p-value: {tr_pc_best.get('p_value', 0.0):.4f}")
                _write_line(handle, f"    95% CI: [{tr_pc_best.get('ci_lower', 0.0):.4f}, {tr_pc_best.get('ci_upper', 0.0):.4f}]")
                _write_line(handle, f"    Practical margin >= {PRACTICAL_MARGIN_PP:.1f} pp: {per_class_best.get('practical_margin_met', False)}")
            _write_line(handle)
            _write_line(handle, "  Per-class comparison to published YOLOF baseline:")
            if not per_class_base.get('available', False):
                _write_line(handle, "    Unavailable: per-class baseline values were not found in the supplied YOLOF baseline outputs.")
            else:
                tr_pc_base = per_class_base['test_result']
                _write_line(handle, f"    Mean per-class difference: {tr_pc_base.get('mean_difference', 0.0):.4f} pp")
                _write_line(handle, f"    p-value: {tr_pc_base.get('p_value', 0.0):.4f}")
                _write_line(handle, f"    95% CI: [{tr_pc_base.get('ci_lower', 0.0):.4f}, {tr_pc_base.get('ci_upper', 0.0):.4f}]")
                _write_line(handle, f"    Practical margin >= {PRACTICAL_MARGIN_PP:.1f} pp: {per_class_base.get('practical_margin_met', False)}")
            _write_line(handle)

        # Descriptive: geometry
        _write_line(handle, "DESCRIPTIVE ANALYSIS: PER-BRANCH LOSS LANDSCAPE GEOMETRY (not a formal hypothesis)")
        _write_line(handle, "-" * 80)
        if descriptive_geometry_data:
            cls_reg = descriptive_geometry_data.get("cls_minus_reg", {})
            tr = cls_reg.get("test_result", {})
            _write_line(handle, f"  Paired comparison across {descriptive_geometry_data.get('n_pairs', 0)} model pairs:")
            _write_line(handle, f"    Mean difference (B_cls - B_reg): {cls_reg.get('mean_difference', 0.0):.4f}")
            _write_line(handle, f"    Observed p-value (descriptive only): {tr.get('p_value', 0.0):.4f}")
            _write_line(handle, f"    Cohen's d: {tr.get('cohens_d', 0.0):.4f}")
            _write_line(handle, f"    95% CI: [{tr.get('ci_lower', 0.0):.4f}, {tr.get('ci_upper', 0.0):.4f}]")
            _write_line(handle)

        # Descriptive: calibration
        _write_line(handle, "DESCRIPTIVE ANALYSIS: TEMPERATURE CALIBRATION DEVIATION (not a formal hypothesis)")
        _write_line(handle, "-" * 80)
        if descriptive_calibration_data:
            _write_line(handle, f"  Bonferroni alpha: {descriptive_calibration_data.get('bonferroni_alpha', BONFERRONI_ALPHA):.4f}")
            for name, test in descriptive_calibration_data.get("tests", {}).items():
                _write_line(handle, f"  beta_{name}:")
                if "note" in test:
                    _write_line(handle, f"    {test['note']}")
                else:
                    _write_line(handle, f"    Mean value: {test.get('mean_value', 0.0):.6f}")
                    _write_line(handle, f"    Mean deviation from 1.0: {test.get('mean_deviation', 0.0):.6f}")
                    _write_line(handle, f"    Observed p-value (descriptive only): {test.get('p_value', 0.0):.4f}")
                    _write_line(handle, f"    Bonferroni significant: {test.get('bonferroni_significant', False)}")
                _write_line(handle)

        _write_line(handle, "=" * 80)
        _write_line(handle, "END OF REPORT")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> Dict[str, Any]:
    from yolof_soup.config.experiment_config import RESULTS_DIR

    parser = argparse.ArgumentParser(description="Phase 7: Statistical analysis & hypothesis testing")
    parser.add_argument(
        "--soup-results-json",
        default=str(Path(RESULTS_DIR) / "phase3_soup_results.json"),
        help="Path to phase3_soup_results.json",
    )
    parser.add_argument(
        "--ingredient-results-json",
        default=str(Path(RESULTS_DIR) / "phase1_ingredient_results.json"),
        help="Path to phase1_ingredient_results.json",
    )
    parser.add_argument(
        "--barriers-json",
        default=str(Path(RESULTS_DIR) / "phase4_lmc_barriers.json"),
        help="Path to phase4_lmc_barriers.json or a compatible barrier JSON",
    )
    parser.add_argument(
        "--hessians-json",
        default=str(Path(RESULTS_DIR) / "phase4_hessian_traces.json"),
        help="Path to phase4_hessian_traces.json",
    )
    parser.add_argument(
        "--finetuning-results-json",
        default=str(Path(RESULTS_DIR) / "phase5_finetuning_results.json"),
        help="Path to phase5_finetuning_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory to save statistical test results",
    )
    parser.add_argument(
        "--baseline-map",
        type=float,
        default=PUBLISHED_BASELINE_MAP,
        help="Published YOLOF baseline mAP50:95",
    )
    args = parser.parse_args()

    soup_path = first_existing_path([args.soup_results_json, Path(RESULTS_DIR) / "phase3_soup_results.json"])
    if soup_path is None:
        raise FileNotFoundError("Could not locate phase-3 soup results JSON.")
    soup_results = load_json_file(soup_path)

    ingredient_path = first_existing_path([args.ingredient_results_json, Path(RESULTS_DIR) / "phase1_ingredient_results.json"])
    ingredient_results = load_json_file(ingredient_path) if ingredient_path and ingredient_path.exists() else {}

    barrier_path = first_existing_path([
        args.barriers_json,
        Path(RESULTS_DIR) / "phase4_lmc_barriers.json",
        Path(RESULTS_DIR) / "phase4_barrier_results.json",
        Path(RESULTS_DIR) / "h2_barrier_pair_averaged_15rows.json",
    ])
    if barrier_path is None:
        raise FileNotFoundError("Could not locate barrier JSON for RQ2/H2.")
    barriers_hessians = {"barriers": load_json_file(barrier_path)}

    hessian_path = first_existing_path([args.hessians_json, Path(RESULTS_DIR) / "phase4_hessian_traces.json"])
    if hessian_path and hessian_path.exists():
        barriers_hessians["hessians"] = load_json_file(hessian_path)

    finetune_candidates = [
        args.finetuning_results_json,
        Path(RESULTS_DIR) / "phase5_finetuning_results.json",
        Path(RESULTS_DIR) / "phase5_soup_finetune.json",
        Path(RESULTS_DIR) / "phase4_finetune_results.json",
    ]
    finetune_path = first_existing_path(finetune_candidates)
    if finetune_path is None:
        raise FileNotFoundError("Could not locate phase-5 finetuning results JSON.")
    finetuning_results = load_json_file(finetune_path)

    results = run_all_hypothesis_tests(
        soup_results=soup_results,
        ingredient_results=ingredient_results,
        barriers_hessians=barriers_hessians,
        finetuning_results=finetuning_results,
        source_paths={
            "soup_results_json": str(soup_path),
            "ingredient_results_json": str(ingredient_path) if ingredient_path else None,
            "barriers_json": str(barrier_path),
            "hessians_json": str(hessian_path) if hessian_path else None,
            "finetuning_results_json": str(finetune_path),
        },
        published_baseline_map=args.baseline_map,
        alpha=ALPHA,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase7_hypothesis_tests.json"
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info("Statistical test results saved -> %s", json_path)

    txt_path = output_dir / "phase7_statistical_report.txt"
    write_statistical_report(results, txt_path)
    logger.info("Statistical report saved -> %s", txt_path)

    logger.info("PHASE 7 COMPLETE")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")
    main()
