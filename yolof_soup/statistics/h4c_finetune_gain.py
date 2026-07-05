"""
RQ4 / H4c — Decoder-only post-merge fine-tuning gain;
            merge quality moderates fine-tuning gain (D1 vs D2 vs C3)

Tests (Section 3.5.4, Tests 4 & 5):
  Test 4 : Paired t-tests within each fine-tuning pair:
             Pair A : Condition 2 soup vs D1 fine-tuned
             Pair B : best of Conditions 3–5 vs D2 fine-tuned
             Pair C : Condition 6 (M6) vs C3 fine-tuned
           Decision: H04c rejected for a pair if 95 % boot CI lower bound ≥ 0.

  Test 5 : Independent-samples gain comparison across Pairs A, B, C:
             - Compute gain array (80 per-class Δ AP) for each pair
             - Kruskal-Wallis test (non-parametric; small per-pair n=80 but
               independent across pairs) followed by Dunn's post-hoc.
             - C3 mAP50:95 vs D2 mAP50:95 vs published 37.7 AP baseline
               (Chen et al., 2021) via 95 % bootstrap CI.

Input file:
  results/phase3_soup_results.json    (keys: condition_2, condition_5 or best_condition,
                                             condition_6)
  results/phase4_finetune_results.json  (keys: D1, D2, C3 → per_class_ap + map50_95)
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
FINETUNE_FILE = RESULTS_DIR / "phase4_finetune_results.json"
BASELINE_MAP = 37.7   # published YOLOF-R50 mAP₅₀:₉₅ (Chen et al., 2021)
N_BOOT = 10_000
RNG_SEED = 42
ALPHA = 0.05


def bootstrap_ci_mean(arr: np.ndarray, seed: int = RNG_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                     for _ in range(N_BOOT)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_test(name: str, pre: np.ndarray, post: np.ndarray):
    """Paired t-test + bootstrap CI on per-class AP gain (post − pre)."""
    diff = post - pre
    t_val, p_val = stats.ttest_rel(post, pre)
    ci_lo, ci_hi = bootstrap_ci_mean(diff)
    cohens_d = diff.mean() / diff.std(ddof=1)
    decision = "REJECT H04c" if ci_lo >= 0 else "FAIL TO REJECT H04c"
    print(f"\n--- {name} ---")
    print(f"  t({len(diff)-1}) = {t_val:.4f}, p = {p_val:.4f}")
    print(f"  Mean gain = {diff.mean():+.4f} pp")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  95 % boot CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Decision: {decision}")
    return {"t": float(t_val), "p": float(p_val), "cohens_d": float(cohens_d),
            "mean_gain": float(diff.mean()), "ci_lower": ci_lo, "ci_upper": ci_hi,
            "decision": decision, "gain_array": diff.tolist()}


def main():
    with open(SOUP_FILE) as f:
        soup = json.load(f)
    with open(FINETUNE_FILE) as f:
        ft = json.load(f)

    c2  = np.array(soup["condition_2"]["per_class_ap"])
    c6  = np.array(soup["condition_6"]["per_class_ap"])
    D1  = np.array(ft["D1"]["per_class_ap"])
    D2  = np.array(ft["D2"]["per_class_ap"])
    C3  = np.array(ft["C3"]["per_class_ap"])

    # Best of Conditions 3–5 (init for D2)
    best_key = max(
        ["condition_3", "condition_4", "condition_5"],
        key=lambda k: soup[k].get("map50_95", np.array(soup[k]["per_class_ap"]).mean()),
    )
    best_c35 = np.array(soup[best_key]["per_class_ap"])
    print(f"Best condition 3–5 (D2 init): {best_key}")

    # ------------------------------------------------------------------
    # Test 4 — Paired t-tests for each fine-tuning pair
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 4 — Post-merge fine-tuning gain (paired t-tests)")
    print("="*60)
    pA = paired_test("Pair A: Condition 2 → D1", c2, D1)
    pB = paired_test(f"Pair B: {best_key} → D2", best_c35, D2)
    pC = paired_test("Pair C: Condition 6 (M6) → C3", c6, C3)

    # ------------------------------------------------------------------
    # Test 5 — Merge quality moderates fine-tuning gain
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 5 — Merge quality moderates fine-tuning gain")
    print("="*60)

    gainA = np.array(pA["gain_array"])
    gainB = np.array(pB["gain_array"])
    gainC = np.array(pC["gain_array"])

    # Kruskal-Wallis across the three gain arrays
    kw_stat, kw_p = stats.kruskal(gainA, gainB, gainC)
    print(f"  Kruskal-Wallis: H = {kw_stat:.4f}, p = {kw_p:.4f}")

    # Dunn-style pairwise Mann-Whitney U (Bonferroni corrected)
    pairs = [("A vs B", gainA, gainB), ("A vs C", gainA, gainC), ("B vs C", gainB, gainC)]
    for label, g1, g2 in pairs:
        u, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        p_bonf = min(p_u * 3, 1.0)
        print(f"  {label}: U = {u:.1f}, p = {p_u:.4f}, p_Bonf = {p_bonf:.4f}")

    # Headline mAP comparison: C3 vs D2 vs published baseline
    map_D1 = ft["D1"].get("map50_95", D1.mean())
    map_D2 = ft["D2"].get("map50_95", D2.mean())
    map_C3 = ft["C3"].get("map50_95", C3.mean())
    print(f"\n  Headline mAP50:95:")
    print(f"    D1  = {map_D1:.2f}")
    print(f"    D2  = {map_D2:.2f}")
    print(f"    C3  = {map_C3:.2f}")
    print(f"    Published baseline = {BASELINE_MAP:.1f}")

    # Bootstrap CI: C3 − D2
    diff_C3_D2 = C3 - D2
    ci_lo, ci_hi = bootstrap_ci_mean(diff_C3_D2)
    print(f"  C3 − D2  mean Δ = {diff_C3_D2.mean():+.4f} pp, "
          f"95 % boot CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Bootstrap CI: C3 − published baseline (using per-class values directly)
    diff_C3_base = C3 - BASELINE_MAP  # scalar baseline; distributional CI on C3
    ci_lo_b, ci_hi_b = bootstrap_ci_mean(diff_C3_base)
    print(f"  C3 vs 37.7 pp baseline: mean Δ = {diff_C3_base.mean():+.4f} pp, "
          f"95 % boot CI = [{ci_lo_b:.4f}, {ci_hi_b:.4f}]")

    out = RESULTS_DIR / "h4c_results.json"
    with open(out, "w") as f:
        json.dump({
            "test4_pairA": {k: v for k, v in pA.items() if k != "gain_array"},
            "test4_pairB": {k: v for k, v in pB.items() if k != "gain_array"},
            "test4_pairC": {k: v for k, v in pC.items() if k != "gain_array"},
            "test5_kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
            "test5_C3_vs_D2": {"mean_diff": float(diff_C3_D2.mean()),
                                "ci_lower": ci_lo, "ci_upper": ci_hi},
            "test5_C3_vs_baseline": {"mean_diff": float(diff_C3_base.mean()),
                                      "ci_lower": ci_lo_b, "ci_upper": ci_hi_b},
        }, f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
