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

from datetime import datetime
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_final_eval_results_2026-08-30_22-26-46.json"
INGREDIENT_FILE = RESULTS_DIR / "phase1_ingredient_results_2026-08-30_23-44-21.json"
N_BOOT = 10_000
RNG_SEED = 42
PRACTICAL_THRESHOLD = 0.5
NUM_CLASSES = 70


def extract_ap_array(per_class_ap, expected_len=NUM_CLASSES, field_name="per_class_ap"):
    """
    per_class_ap is a list of [class_name, AP, AR] triples.
    Returns a flat float array of the 80 AP values (index 1), in file order.
    """
    if not isinstance(per_class_ap, list):
        raise TypeError(f"{field_name} must be a list of [class_name, AP, AR] triples.")

    ap_values = []
    class_names = []
    for i, entry in enumerate(per_class_ap):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(
                f"{field_name}[{i}] must be [class_name, AP, AR]; got {entry!r}"
            )
        class_name, ap = entry[0], entry[1]
        if not isinstance(ap, (int, float)):
            raise TypeError(
                f"{field_name}[{i}] AP value must be numeric; got {type(ap)} -> {ap!r}"
            )
        class_names.append(class_name)
        ap_values.append(float(ap))

    arr = np.array(ap_values, dtype=float)

    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(
            f"{field_name} must contain {expected_len} classes, got {len(arr)}."
        )
    return arr, class_names


def score_entry(entry):
    if "map50_95" in entry and entry["map50_95"] is not None:
        return float(entry["map50_95"])
    ap_arr, _ = extract_ap_array(entry["per_class_ap"])
    return float(ap_arr.mean())


def bootstrap_ci(diff, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean()
        for _ in range(n_boot)
    ])
    return diff.mean(), *np.percentile(boot, [2.5, 97.5])


def cohens_d_paired(diff):
    return diff.mean() / diff.std(ddof=1)


def run_comparison(name, ap_a, ap_b):
    assert len(ap_a) == NUM_CLASSES and len(ap_b) == NUM_CLASSES, f"AP arrays must be of length {NUM_CLASSES} (COCO categories)"
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


def main():
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    c1, class_names_ref = extract_ap_array(soup["condition_1"]["per_class_ap"], field_name="condition_1.per_class_ap")
    c2, class_names_c2 = extract_ap_array(soup["condition_2"]["per_class_ap"], field_name="condition_2.per_class_ap")

    # Sanity check: class order must match across conditions for paired tests to be valid
    if class_names_ref != class_names_c2:
        raise ValueError(
            "Class order mismatch between condition_1 and condition_2 per_class_ap arrays. "
            "Paired tests require identical category order across all conditions."
        )

    learned_conditions = {}
    for k in ["condition_3", "condition_4", "condition_5", "condition_6"]:
        arr, names = extract_ap_array(soup[k]["per_class_ap"], field_name=f"{k}.per_class_ap")
        if names != class_names_ref:
            raise ValueError(f"Class order mismatch in {k}.per_class_ap vs condition_1.")
        learned_conditions[k] = arr

    best_learned_key = max(learned_conditions, key=lambda k: score_entry(soup[k]))
    best_learned = learned_conditions[best_learned_key]
    print(f"Best learned condition: {best_learned_key}")

    with open(INGREDIENT_FILE) as f:
        ingredients = json.load(f)

    best_ingredient_key = max(ingredients, key=lambda k: score_entry(ingredients[k]))
    best_ingredient, names_ing = extract_ap_array(
        ingredients[best_ingredient_key]["per_class_ap"],
        field_name=f"{best_ingredient_key}.per_class_ap"
    )
    if names_ing != class_names_ref:
        raise ValueError(
            f"Class order mismatch: {best_ingredient_key}.per_class_ap vs condition_1.per_class_ap."
        )
    print(f"Best ingredient model : {best_ingredient_key}")

    results = []
    results.append(run_comparison("Comparison A: C1 (global) → C2 (component uniform)", c1, c2))
    results.append(run_comparison(f"Comparison B: C2 → {best_learned_key} (best learned)", c2, best_learned))
    results.append(run_comparison(
        f"Comparison C: {best_ingredient_key} (best single) → {best_learned_key} (best soup)",
        best_ingredient, best_learned,
    ))

    out = RESULTS_DIR / f"h1_rq1_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()