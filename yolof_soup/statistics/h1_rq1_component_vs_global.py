"""
RQ1 / H1 — Component-specific decoder averaging vs. global uniform soup
           and vs. best individual ingredient model.

Tests (Section 3.5.1):
  Comparison A : C1 (global uniform) vs C2 (component uniform)  → isolates IV1
  Comparison B : C2 vs best of {C3, C4, C5, C6}                → isolates IV2
  Comparison C : best learned soup vs best single ingredient     → practical value

For each comparison:
  - Paired-sample t-test over 80 per-class AP values (scipy)
  - Cohen's d on paired differences
  - 10 000-resample bootstrap 95 % CI on mean difference
  - Wilcoxon signed-rank test (non-parametric robustness check)

Decision rule  (pre-specified in Ch. 3, §3.5.1):
  H01 rejected for Comparison C if CI lower bound ≥ 0 AND ΔmAP ≥ 0.5 pp.

Input files (all JSON, produced by soup_construction.py / quality_audit.py):
  - results/phase3_soup_results.json      keys: condition_1 … condition_6
  - results/phase1_ingredient_results.json keys: L1 … L4, R1, R2

Each entry must contain a list "per_class_ap" of length 80 (COCO category order).
"""

import json
import pathlib
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
INGREDIENT_FILE = RESULTS_DIR / "phase1_ingredient_results.json"
N_BOOT = 10_000
RNG_SEED = 42
PRACTICAL_THRESHOLD = 0.5  # pp; pre-specified in §3.5.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """Return (mean_diff, ci_lower, ci_upper) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(n_boot)])
    return diff.mean(), *np.percentile(boot, [2.5, 97.5])


def cohens_d_paired(diff: np.ndarray) -> float:
    return diff.mean() / diff.std(ddof=1)


def run_comparison(name: str, ap_a: np.ndarray, ap_b: np.ndarray):
    """Print all statistics for one comparison (B − A)."""
    assert len(ap_a) == 80 and len(ap_b) == 80, "Arrays must have length 80."
    diff = ap_b - ap_a

    t, p = stats.ttest_rel(ap_b, ap_a)
    d_val = cohens_d_paired(diff)
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diff)
    w_stat, w_p = stats.wilcoxon(ap_b, ap_a)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Paired t-test : t({len(diff)-1}) = {t:.4f}, p = {p:.4f}")
    print(f"  Mean ΔmAP     : {mean_diff:+.4f} pp  (B − A)")
    print(f"  Cohen's d     : {d_val:.4f}")
    print(f"  95 % boot CI  : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Wilcoxon      : W = {w_stat:.1f}, p = {w_p:.4f}")
    decision = (
        "REJECT H01" if ci_lo >= 0 and mean_diff >= PRACTICAL_THRESHOLD
        else "FAIL TO REJECT H01"
    )
    print(f"  Decision      : {decision}")
    return {
        "comparison": name,
        "t": t, "p": p, "cohens_d": d_val,
        "mean_diff": mean_diff, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load soup results
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    c1 = np.array(soup["condition_1"]["per_class_ap"])
    c2 = np.array(soup["condition_2"]["per_class_ap"])

    # Best of Conditions 3–6 by overall mAP50:95
    learned_conditions = {
        k: np.array(soup[k]["per_class_ap"])
        for k in ["condition_3", "condition_4", "condition_5", "condition_6"]
    }
    best_learned_key = max(
        learned_conditions,
        key=lambda k: soup[k].get("map50_95", np.array(soup[k]["per_class_ap"]).mean()),
    )
    best_learned = learned_conditions[best_learned_key]
    print(f"Best learned condition: {best_learned_key}")

    # Load ingredient results
    with open(INGREDIENT_FILE) as f:
        ingredients = json.load(f)

    best_ingredient_key = max(
        ingredients,
        key=lambda k: ingredients[k].get("map50_95",
                                          np.array(ingredients[k]["per_class_ap"]).mean()),
    )
    best_ingredient = np.array(ingredients[best_ingredient_key]["per_class_ap"])
    print(f"Best ingredient model : {best_ingredient_key}")

    results = []
    results.append(run_comparison("Comparison A: C1 (global) → C2 (component uniform)", c1, c2))
    results.append(run_comparison(f"Comparison B: C2 → {best_learned_key} (best learned)", c2, best_learned))
    results.append(run_comparison(
        f"Comparison C: {best_ingredient_key} (best single) → {best_learned_key} (best soup)",
        best_ingredient, best_learned,
    ))

    # Save summary
    out = RESULTS_DIR / "h1_rq1_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
