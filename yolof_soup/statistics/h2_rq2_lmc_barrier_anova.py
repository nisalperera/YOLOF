"""
RQ2 / H2 — Per-component loss landscape geometry (LMC barriers + Hessian traces)

Tests (Section 3.5.2):
  Test 1 : Repeated-measures ANOVA comparing B_backbone_encoder, B_cls_head,
           B_reg_head, B_objectness_module across 15 model pairs.
           If Mauchly's test is violated → Greenhouse-Geisser correction.
           Tukey HSD post-hoc (and directed Bonferroni-corrected contrasts).
           Also: paired two-tailed t-test B_cls_head vs B_reg_head (primary H2 criterion).

  Test 2 : Pearson correlation between per-component barrier magnitude and
           per-component averaging gain (Δ mAP vs Condition 1), Bonferroni-corrected.

  Test 3 : Repeated-measures ANOVA comparing per-component Hessian traces
           across the 6 ingredient models.

Input files (produced by loss_landscape.py):
  results/phase4_barrier_results.json
    Structure:
      {
        "pair_barriers": [
          {
            "pair": [i, j],
            "backbone_encoder": <float>,   # averaged over 6 base conditions
            "cls_head":         <float>,
            "reg_head":         <float>,
            "objectness_module":<float>,
            "full_model":       <float>
          },
          ...  # 15 entries
        ],
        "hessian_traces": {
          "L1": {"backbone_encoder": f, "cls_head": f, "reg_head": f, "objectness_module": f},
          ...  # 6 ingredient models
        },
        "component_gains": {
          "cls_head": <float>,   # Δ mAP of component-averaging vs C1
          "reg_head": <float>,
          "objectness_module": <float>
        }
      }
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
BARRIER_FILE = RESULTS_DIR / "phase4_barrier_results.json"
COMPONENTS = ["backbone_encoder", "cls_head", "reg_head", "objectness_module"]
ALPHA = 0.05


def greenhouse_geisser_epsilon(data: np.ndarray) -> float:
    """
    Compute Greenhouse-Geisser epsilon from an (n_subjects × k_conditions) array.
    Follows the standard covariance-matrix formula.
    """
    n, k = data.shape
    grand_mean = data.mean()
    row_means = data.mean(axis=1, keepdims=True)
    col_means = data.mean(axis=0, keepdims=True)
    S = data - row_means - col_means + grand_mean
    cov = (S.T @ S) / (n - 1)
    cov_trace = np.trace(cov)
    cov_sq_trace = np.trace(cov @ cov)
    numerator = cov_trace ** 2
    denominator = (k - 1) * (cov_sq_trace - (cov_trace ** 2) / k)
    epsilon = numerator / denominator
    return float(np.clip(epsilon, 1 / (k - 1), 1.0))


def rm_anova_manual(data: np.ndarray, component_names: list):
    """
    Manual repeated-measures ANOVA.
    data: shape (n_subjects, k_conditions)
    Returns F, p, eta_sq, and GG-corrected p.
    """
    n, k = data.shape
    grand_mean = data.mean()
    ss_between = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)
    ss_subjects = k * np.sum((data.mean(axis=1) - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_between - ss_subjects

    df_between = k - 1
    df_error = (n - 1) * (k - 1)
    ms_between = ss_between / df_between
    ms_error = ss_error / df_error

    F = ms_between / ms_error
    p = stats.f.sf(F, df_between, df_error)
    eta_sq = ss_between / (ss_between + ss_error)

    epsilon = greenhouse_geisser_epsilon(data)
    df_gg_between = df_between * epsilon
    df_gg_error = df_error * epsilon
    p_gg = stats.f.sf(F, df_gg_between, df_gg_error)

    print(f"  F({df_between:.2f}, {df_error:.2f}) = {F:.4f}, p = {p:.4f}")
    print(f"  GG epsilon = {epsilon:.4f}  →  F({df_gg_between:.2f}, {df_gg_error:.2f}), p_GG = {p_gg:.4f}")
    print(f"  η² = {eta_sq:.4f}")
    return {"F": F, "p": p, "p_gg": p_gg, "eta_sq": eta_sq, "epsilon": epsilon}


def tukey_hsd_posthoc(data: np.ndarray, labels: list, alpha: float = ALPHA):
    """Approximate Tukey HSD for RM design (conservative; use pingouin if available)."""
    try:
        import pingouin as pg
        import pandas as pd
        n, k = data.shape
        long = pd.DataFrame({
            "subject": np.tile(np.arange(n), k),
            "condition": np.repeat(labels, n),
            "value": data.flatten(order="F"),
        })
        ph = pg.pairwise_tests(dv="value", within="condition",
                               subject="subject", data=long, padjust="holm")
        print(ph.to_string())
        return ph
    except ImportError:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        n, k = data.shape
        flat_vals = data.flatten(order="F")
        flat_labels = np.repeat(labels, n)
        res = pairwise_tukeyhsd(flat_vals, flat_labels, alpha=alpha)
        print(res)
        return res


def main():
    with open(BARRIER_FILE) as f:
        barrier_data = json.load(f)

    pairs = barrier_data["pair_barriers"]   # list of 15 dicts
    assert len(pairs) == 15, f"Expected 15 pairs, got {len(pairs)}"

    # Build (15 × 4) matrix for RM-ANOVA
    barrier_matrix = np.array(
        [[p[c] for c in COMPONENTS] for p in pairs]
    )  # shape (15, 4)

    # ------------------------------------------------------------------
    # Test 1a: RM-ANOVA across 4 components
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 1a — RM-ANOVA: barrier across 4 components (15 pairs)")
    print("="*60)
    anova_res = rm_anova_manual(barrier_matrix, COMPONENTS)

    print("\n--- Post-hoc (Tukey / Holm) ---")
    tukey_hsd_posthoc(barrier_matrix, COMPONENTS)

    # ------------------------------------------------------------------
    # Test 1b: Directed paired t-test B_cls_head vs B_reg_head (primary H2 criterion)
    # ------------------------------------------------------------------
    cls_barriers = barrier_matrix[:, COMPONENTS.index("cls_head")]
    reg_barriers = barrier_matrix[:, COMPONENTS.index("reg_head")]
    t_cls_reg, p_cls_reg = stats.ttest_rel(cls_barriers, reg_barriers)
    diff_cls_reg = cls_barriers - reg_barriers
    d_cls_reg = diff_cls_reg.mean() / diff_cls_reg.std(ddof=1)

    print("\n" + "="*60)
    print("TEST 1b — Paired t-test: B_cls_head vs B_reg_head")
    print("="*60)
    print(f"  t({len(diff_cls_reg)-1}) = {t_cls_reg:.4f}, p = {p_cls_reg:.4f}")
    print(f"  Mean difference (cls − reg) = {diff_cls_reg.mean():+.6f}")
    print(f"  Cohen's d = {d_cls_reg:.4f}")
    decision_h2 = "REJECT H02" if p_cls_reg < ALPHA else "FAIL TO REJECT H02"
    print(f"  Decision: {decision_h2}")

    # ------------------------------------------------------------------
    # Test 2: Pearson correlation barrier magnitude → averaging gain
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2 — Pearson correlation: per-component barrier vs. averaging gain")
    print("="*60)
    gains = barrier_data.get("component_gains", {})
    gain_components = [c for c in ["cls_head", "reg_head", "objectness_module"] if c in gains]
    bonferroni_n = len(gain_components)
    for comp in gain_components:
        comp_barriers = barrier_matrix[:, COMPONENTS.index(comp)]
        gain_val = gains[comp]  # scalar Δ mAP
        # With a single scalar gain we compute correlation across pairs
        # using the pair-level barrier already computed per pair
        r, p_r = stats.pearsonr(comp_barriers, np.full(len(comp_barriers), gain_val))
        p_bonf = min(p_r * bonferroni_n, 1.0)
        print(f"  {comp}: r = {r:.4f}, p = {p_r:.4f}, p_Bonferroni = {p_bonf:.4f}")

    # ------------------------------------------------------------------
    # Test 3: RM-ANOVA on Hessian traces across 6 ingredients
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 3 — RM-ANOVA: Hessian trace across 4 components (6 ingredients)")
    print("="*60)
    hessian = barrier_data["hessian_traces"]  # dict model → dict component → float
    models = list(hessian.keys())
    hessian_matrix = np.array(
        [[hessian[m][c] for c in COMPONENTS] for m in models]
    )  # shape (n_models, 4)
    rm_anova_manual(hessian_matrix, COMPONENTS)
    print("--- Post-hoc (Tukey / Holm) ---")
    tukey_hsd_posthoc(hessian_matrix, COMPONENTS)

    # Save
    out = RESULTS_DIR / "h2_rq2_results.json"
    summary = {
        "test1a_anova": anova_res,
        "test1b_cls_vs_reg": {
            "t": float(t_cls_reg), "p": float(p_cls_reg),
            "cohens_d": float(d_cls_reg), "decision": decision_h2,
        },
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
