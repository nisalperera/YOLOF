"""
RQ4 / H4a — M6 (tri-component learned α+β, independent pairs)
            vs M5 (shared α+β pair)

Test (Section 3.5.4, Test 1):
  Paired-sample t-test over 80 per-class AP values.
  Cohen's d, 10 000-resample bootstrap 95 % CI.
  Wilcoxon signed-rank robustness check.

Decision rule: reject H04a if p < 0.05 AND 95 % CI lower bound ≥ 0.

Input file:
  results/phase3_soup_results.json
    Keys: condition_5 → per_class_ap, condition_6 → per_class_ap
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
N_BOOT = 10_000
RNG_SEED = 42
ALPHA = 0.05


def main():
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    m5 = np.array(soup["condition_5"]["per_class_ap"])
    m6 = np.array(soup["condition_6"]["per_class_ap"])
    assert len(m5) == 80 and len(m6) == 80

    diff = m6 - m5
    t_stat, p_value = stats.ttest_rel(m6, m5)
    cohens_d = diff.mean() / diff.std(ddof=1)

    rng = np.random.default_rng(RNG_SEED)
    boot_means = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                           for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    w_stat, w_p = stats.wilcoxon(m6, m5)

    print("\n" + "="*60)
    print("H4a — Paired t-test: M6 vs M5 (per-class AP, n=80)")
    print("="*60)
    print(f"  t({len(diff)-1}) = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"  Mean Δ (M6 − M5) = {diff.mean():+.4f} pp")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  95 % boot CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Wilcoxon: W = {w_stat:.1f}, p = {w_p:.4f}")

    decision = (
        "REJECT H04a" if p_value < ALPHA and ci_lo >= 0
        else "FAIL TO REJECT H04a"
    )
    print(f"  Decision: {decision}")

    out = RESULTS_DIR / "h4a_results.json"
    with open(out, "w") as f:
        json.dump({
            "t": float(t_stat), "p": float(p_value),
            "cohens_d": float(cohens_d),
            "ci_lower": float(ci_lo), "ci_upper": float(ci_hi),
            "wilcoxon_W": float(w_stat), "wilcoxon_p": float(w_p),
            "decision": decision,
        }, f, indent=2)
    print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()
