"""
utils/stats_utils.py
=====================
All statistical tests required by the thesis methodology.

  RQ1/H1   wilcoxon_one_tailed()   + cohens_d()
  RQ2/H2a  mann_whitney_u_test()   + wilcoxon_paired()
  RQ2/H2b  spearman_r()
  RQ3/H3   compare_diversity_gain()   (descriptive)
  RQ4/H4   compare_domain_gains()     (CI-based falsification)
  General  bootstrap_ci()

Dependencies: scipy, numpy (already required by YOLOF).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from yolof_soup.utils.global_logger import get_logger

logger = get_logger(logging.DEBUG, add_file_handler=True)


# ── RQ1/H1 ───────────────────────────────────────────────

def wilcoxon_one_tailed(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test, one-tailed (H1: x > y).
    Used for RQ1/H1: head-specific soup AP > global soup AP.
    """
    if len(x) != len(y):
        raise ValueError(f"Equal lengths required; got {len(x)} vs {len(y)}.")
    if len(x) < 3:
        logger.warning("Wilcoxon needs n≥3; got n=%d.", len(x))

    stat_two, p_two = stats.wilcoxon(x, y, alternative="two-sided")
    _,         p_gt = stats.wilcoxon(x, y, alternative="greater")

    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    direction      = "head > global" if mean_x > mean_y else "global >= head"

    return dict(
        test           = "Wilcoxon signed-rank",
        statistic      = float(stat_two),
        p_two_sided    = float(p_two),
        p_one_sided_gt = float(p_gt),
        mean_x=mean_x, mean_y=mean_y, n=len(x), alpha=alpha,
        significant    = bool(p_gt < alpha),
        direction      = direction,
        h1_supported   = bool(p_gt < alpha and direction == "head > global"),
    )


def cohens_d(x: List[float], y: List[float]) -> float:
    """
    Cohen's d = (mean_x − mean_y) / pooled_std.
    Returns raw difference when n < 2 in either group.
    """
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    n_x, n_y = len(x), len(y)
    var_x = float(np.var(x, ddof=1)) if n_x >= 2 else 0.0
    var_y = float(np.var(y, ddof=1)) if n_y >= 2 else 0.0
    pooled = float(np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y)
                            / max(n_x + n_y - 2, 1)))
    d = float((mean_x - mean_y) / pooled) if pooled > 0 else float(mean_x - mean_y)
    logger.info("[Cohen's d] d=%.4f  (mean_x=%.4f  mean_y=%.4f)", d, mean_x, mean_y)
    return d


# ── RQ2/H2a ──────────────────────────────────────────────

def mann_whitney_u_test(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    One-sided Mann-Whitney U (H: x < y).
    Used for RQ2/H2a: decoder barriers < backbone-encoder barriers.
    """
    stat_two, p_two = stats.mannwhitneyu(x, y, alternative="two-sided")
    _,         p_lt = stats.mannwhitneyu(x, y, alternative="less")

    return dict(
        test              = "Mann-Whitney U",
        statistic         = float(stat_two),
        p_two_sided       = float(p_two),
        p_one_sided_less  = float(p_lt),
        median_x          = float(np.median(x)),
        median_y          = float(np.median(y)),
        n_x=len(x), n_y=len(y), alpha=alpha,
        significant_less  = bool(p_lt < alpha),
        h2a_supported     = bool(p_lt < alpha),
    )


def wilcoxon_paired(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Paired Wilcoxon signed-rank (two-sided) for matched pairs.
    Used for RQ2/H2a sharpness comparison.
    Falls back to descriptive stats when n < 5.
    """
    if len(x) != len(y):
        raise ValueError(f"Equal lengths required; got {len(x)} vs {len(y)}.")
    if len(x) < 5:
        return dict(
            test="Wilcoxon paired",
            note=f"n={len(x)}<5 — descriptive only",
            mean_diff=float(np.mean(np.array(x) - np.array(y))),
            n=len(x),
        )
    stat, p = stats.wilcoxon(x, y, alternative="two-sided")
    return dict(
        test="Wilcoxon paired",
        statistic=float(stat), p_value=float(p),
        mean_diff=float(np.mean(np.array(x) - np.array(y))),
        n=len(x), alpha=alpha,
        significant=bool(p < alpha),
    )


# ── RQ2/H2b ──────────────────────────────────────────────

def spearman_r(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Spearman rank correlation.
    Used for RQ2/H2b: decoder barrier vs mAP gain.
    h2b_supported = significant AND r < 0 (lower barrier → higher gain).
    """
    r, p = stats.spearmanr(x, y)
    return dict(
        test="Spearman r",
        r=float(r), p_value=float(p),
        n=len(x), alpha=alpha,
        significant=bool(p < alpha),
        h2b_supported=bool(p < alpha and r < 0),
    )


# ── RQ3/H3 (descriptive) ─────────────────────────────────

def compare_diversity_gain(delta_head: float, delta_global: float) -> Dict:
    """
    Directional comparison of N=3→6 mAP gain for head-specific vs global.
    N=2 conditions → no significance test; directional evidence only.
    """
    rel   = delta_head - delta_global
    h3dir = "head > global" if delta_head > delta_global else "global >= head"
    return dict(
        test                     = "Diversity gain comparison (descriptive)",
        delta_head               = float(delta_head),
        delta_global             = float(delta_global),
        relative_gain_head       = float(rel),
        h3_direction             = h3dir,
        h3_tentatively_supported = bool(delta_head > delta_global),
        note = "N=2 conditions — directional evidence only.",
    )


# ── RQ4/H4 ───────────────────────────────────────────────

def compare_domain_gains(
    delta_coco:  float,
    delta_voc:   float,
    ci_voc:      Optional[Dict] = None,
    alpha:       float = 0.05,
) -> Dict:
    """
    H4 falsification: cross-domain mAP gain > in-domain mAP gain?

    Args:
        delta_coco: (head_soup − global_soup) AP on COCO held-out split.
        delta_voc:  (head_soup − global_soup) AP50 on Pascal VOC 2007.
        ci_voc:     bootstrap_ci() output for VOC head mAP scores.
        alpha:      Significance level for CI interpretation.

    Returns:
        Dict with H4 verdict and supporting statistics.
    """
    rel   = delta_voc - delta_coco
    ci_ok = bool(ci_voc["lower"] > 0.0) if ci_voc and "lower" in ci_voc else None
    return dict(
        test                  = "Cross-domain gain comparison",
        delta_coco            = float(delta_coco),
        delta_voc             = float(delta_voc),
        relative_gain_voc     = float(rel),
        h4_direction          = "cross-domain > in-domain" if rel > 0 else "in-domain >= cross-domain",
        ci_voc_excludes_zero  = ci_ok,
        h4_supported          = bool(rel > 0),
        note = ("H4 requires delta_voc > delta_coco AND ci_voc above zero. "
                "Failure falsifies H4 but does not affect H1."),
    )


# ── Bootstrap CI ─────────────────────────────────────────

def bootstrap_ci(
    values:      List[float],
    n_bootstrap: int   = 10_000,
    ci:          float = 0.95,
    seed:        int   = 42,
) -> Dict:
    """
    Non-parametric bootstrap 95 % CI for the mean.
    Works with n=1 (returns the point estimate for all bounds).
    """
    rng   = np.random.default_rng(seed)
    arr   = np.array(values, dtype=float)
    point = float(np.mean(arr))

    if len(arr) == 1:
        return dict(mean=point, lower=point, upper=point,
                    ci_level=ci, note="n=1 — CI equals point estimate")

    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_bootstrap)]
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 - (1 - ci) / 2)))
    return dict(mean=point, lower=lo, upper=hi, ci_level=ci, n=len(arr))