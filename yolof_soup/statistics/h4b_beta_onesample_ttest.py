"""
RQ4 / H4b — Per-component temperature scalars β deviate from 1.0

Tests (Section 3.5.4, Tests 2 & 3):
  Test 2 : One-sample t-tests for β_cls, β_bbox, β_obj against μ₀ = 1.0
           with Bonferroni correction across 3 tests (α_eff = 0.05 / 3 ≈ 0.0167).
  Test 3 : One-way ANOVA testing β_cls ≠ β_bbox ≠ β_obj (heterogeneity).
           Because β scalars are single learned values (not distributions),
           we use the optimised β from Condition 6 (M6) across the N = 6
           ingredient soup replicates (bootstrap resampling used for inference).

Input file:
  results/phase3_soup_results.json
    Key: condition_6 → "beta_values":
         {
           "cls":  [b1, b2, ..., b6],   # one β per bootstrap replicate or per seed
           "bbox": [...],
           "obj":  [...]
         }
    If the key contains single floats rather than lists, a single-point note is printed
    and parametric tests are skipped (bootstrap CIs are used instead).
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
BETA_NULL = 1.0
ALPHA_RAW = 0.05
N_COMPARISONS = 3
ALPHA_BONF = ALPHA_RAW / N_COMPARISONS  # ≈ 0.0167
N_BOOT = 10_000
RNG_SEED = 42


def bootstrap_ci_mean(arr: np.ndarray, seed: int = RNG_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                     for _ in range(N_BOOT)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    beta_data = soup["condition_6"].get("beta_values", None)
    if beta_data is None:
        raise KeyError(
            "'beta_values' not found in condition_6. "
            "Ensure the soup construction script saves β_cls, β_bbox, β_obj."
        )

    components = ["cls", "bbox", "obj"]
    beta_arrays = {}
    for comp in components:
        val = beta_data[comp]
        beta_arrays[comp] = np.array(val) if isinstance(val, list) else np.array([val])

    # ------------------------------------------------------------------
    # Test 2 — One-sample t-tests vs β = 1.0  (Bonferroni corrected)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2 — One-sample t-tests: β vs 1.0  (Bonferroni α = {:.4f})".format(ALPHA_BONF))
    print("="*60)
    t2_results = {}
    for comp in components:
        arr = beta_arrays[comp]
        if len(arr) < 2:
            ci_lo, ci_hi = bootstrap_ci_mean(np.array([arr[0]] * 30 + [arr[0]]))
            print(f"  β_{comp} = {arr[0]:.6f}  (single value — parametric test not applicable)")
            print(f"  95 % boot CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
            t2_results[comp] = {"beta_mean": float(arr[0]), "note": "single value"}
        else:
            t_val, p_val = stats.ttest_1samp(arr, BETA_NULL)
            p_bonf = min(p_val * N_COMPARISONS, 1.0)
            ci_lo, ci_hi = bootstrap_ci_mean(arr)
            decision = "REJECT" if p_bonf < ALPHA_RAW else "FAIL TO REJECT"
            print(f"  β_{comp}: mean = {arr.mean():.6f}, "
                  f"t({len(arr)-1}) = {t_val:.4f}, p = {p_val:.4f}, "
                  f"p_Bonf = {p_bonf:.4f}  →  {decision}")
            print(f"           95 % boot CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
            t2_results[comp] = {
                "beta_mean": float(arr.mean()),
                "t": float(t_val), "p": float(p_val), "p_bonferroni": float(p_bonf),
                "ci_lower": ci_lo, "ci_upper": ci_hi, "decision": decision,
            }

    # ------------------------------------------------------------------
    # Test 3 — One-way ANOVA: β heterogeneity across components
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 3 — One-way ANOVA: β heterogeneity (cls vs bbox vs obj)")
    print("="*60)
    arrays_for_anova = [beta_arrays[c] for c in components]
    if all(len(a) >= 2 for a in arrays_for_anova):
        F_val, p_anova = stats.f_oneway(*arrays_for_anova)
        print(f"  F(2, {sum(len(a) for a in arrays_for_anova)-3}) = {F_val:.4f}, "
              f"p = {p_anova:.4f}")
        anova_decision = "REJECT" if p_anova < ALPHA_RAW else "FAIL TO REJECT"
        print(f"  Decision (β_cls = β_bbox = β_obj): {anova_decision}")
    else:
        print("  Insufficient replicates for one-way ANOVA; reporting means only.")
        for comp in components:
            print(f"  β_{comp} = {beta_arrays[comp][0]:.6f}")
        p_anova = None
        anova_decision = "N/A"

    out = RESULTS_DIR / "h4b_results.json"
    with open(out, "w") as f:
        json.dump({
            "test2_onesample_ttests": t2_results,
            "test3_anova": {
                "F": float(F_val) if p_anova is not None else None,
                "p": float(p_anova) if p_anova is not None else None,
                "decision": anova_decision,
            },
        }, f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
