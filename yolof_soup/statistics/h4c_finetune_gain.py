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

"""
RQ4 / H4c — Post-merge fine-tuning gain; merge quality moderates gain (D1/D2/C3)
per_class_ap format: [[class_name, AP, AR], ...] — AP (index 1) used, AR discarded.
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
FINETUNE_FILE = RESULTS_DIR / "phase4_finetune_results.json"
BASELINE_MAP = 37.7
N_BOOT = 10_000
RNG_SEED = 42
ALPHA = 0.05


def extract_ap_array(per_class_ap, expected_len=80, field_name="per_class_ap"):
    if not isinstance(per_class_ap, list):
        raise TypeError(f"{field_name} must be a list of [class_name, AP, AR] triples.")
    ap_values, class_names = [], []
    for i, entry in enumerate(per_class_ap):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(f"{field_name}[{i}] must be [class_name, AP, AR]; got {entry!r}")
        class_name, ap = entry[0], entry[1]
        if not isinstance(ap, (int, float)):
            raise TypeError(f"{field_name}[{i}] AP must be numeric; got {type(ap)} -> {ap!r}")
        class_names.append(class_name)
        ap_values.append(float(ap))
    arr = np.array(ap_values, dtype=float)
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"{field_name} must contain {expected_len} classes, got {len(arr)}.")
    return arr, class_names


def check_class_order(names_a, names_b, label_a, label_b):
    if names_a != names_b:
        raise ValueError(f"Class order mismatch between {label_a} and {label_b}.")


def bootstrap_ci_mean(arr, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(N_BOOT)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_test(name, pre, post):
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


def score_entry(entry):
    if "map50_95" in entry and entry["map50_95"] is not None:
        return float(entry["map50_95"])
    arr, _ = extract_ap_array(entry["per_class_ap"])
    return float(arr.mean())


def main():
    with open(SOUP_FILE) as f:
        soup = json.load(f)
    with open(FINETUNE_FILE) as f:
        ft = json.load(f)

    c2, names_c2 = extract_ap_array(soup["condition_2"]["per_class_ap"], field_name="condition_2.per_class_ap")
    c6, names_c6 = extract_ap_array(soup["condition_6"]["per_class_ap"], field_name="condition_6.per_class_ap")
    D1, names_D1 = extract_ap_array(ft["D1"]["per_class_ap"], field_name="D1.per_class_ap")
    D2, names_D2 = extract_ap_array(ft["D2"]["per_class_ap"], field_name="D2.per_class_ap")
    C3, names_C3 = extract_ap_array(ft["C3"]["per_class_ap"], field_name="C3.per_class_ap")

    for names, label in [(names_c6, "condition_6"), (names_D1, "D1"), (names_D2, "D2"), (names_C3, "C3")]:
        check_class_order(names_c2, names, "condition_2", label)

    best_key = max(["condition_3", "condition_4", "condition_5"], key=lambda k: score_entry(soup[k]))
    best_c35, names_best = extract_ap_array(soup[best_key]["per_class_ap"], field_name=f"{best_key}.per_class_ap")
    check_class_order(names_c2, names_best, "condition_2", best_key)
    print(f"Best condition 3–5 (D2 init): {best_key}")

    print("\n" + "="*60)
    print("TEST 4 — Post-merge fine-tuning gain (paired t-tests)")
    print("="*60)
    pA = paired_test("Pair A: Condition 2 → D1", c2, D1)
    pB = paired_test(f"Pair B: {best_key} → D2", best_c35, D2)
    pC = paired_test("Pair C: Condition 6 (M6) → C3", c6, C3)

    print("\n" + "="*60)
    print("TEST 5 — Merge quality moderates fine-tuning gain")
    print("="*60)
    gainA = np.array(pA["gain_array"])
    gainB = np.array(pB["gain_array"])
    gainC = np.array(pC["gain_array"])

    kw_stat, kw_p = stats.kruskal(gainA, gainB, gainC)
    print(f"  Kruskal-Wallis: H = {kw_stat:.4f}, p = {kw_p:.4f}")

    for label, g1, g2 in [("A vs B", gainA, gainB), ("A vs C", gainA, gainC), ("B vs C", gainB, gainC)]:
        u, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        p_bonf = min(p_u * 3, 1.0)
        print(f"  {label}: U = {u:.1f}, p = {p_u:.4f}, p_Bonf = {p_bonf:.4f}")

    map_D2 = ft["D2"].get("map50_95", D2.mean())
    map_C3 = ft["C3"].get("map50_95", C3.mean())
    print(f"\n  Headline mAP50:95: D2 = {map_D2:.2f}  |  C3 = {map_C3:.2f}  |  Baseline = {BASELINE_MAP:.1f}")

    diff_C3_D2 = C3 - D2
    ci_lo, ci_hi = bootstrap_ci_mean(diff_C3_D2)
    print(f"  C3 − D2 mean Δ = {diff_C3_D2.mean():+.4f} pp, 95 % boot CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

    diff_C3_base = C3 - BASELINE_MAP
    ci_lo_b, ci_hi_b = bootstrap_ci_mean(diff_C3_base)
    print(f"  C3 vs baseline mean Δ = {diff_C3_base.mean():+.4f} pp, 95 % boot CI = [{ci_lo_b:.4f}, {ci_hi_b:.4f}]")

    out = RESULTS_DIR / "h4c_results.json"
    with open(out, "w") as f:
        json.dump({
            "test4_pairA": {k: v for k, v in pA.items() if k != "gain_array"},
            "test4_pairB": {k: v for k, v in pB.items() if k != "gain_array"},
            "test4_pairC": {k: v for k, v in pC.items() if k != "gain_array"},
            "test5_kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
            "test5_C3_vs_D2": {"mean_diff": float(diff_C3_D2.mean()), "ci_lower": ci_lo, "ci_upper": ci_hi},
            "test5_C3_vs_baseline": {"mean_diff": float(diff_C3_base.mean()), "ci_lower": ci_lo_b, "ci_upper": ci_hi_b},
        }, f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()