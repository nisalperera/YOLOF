"""
RQ3 / H3 — Coefficient learning strategy comparison (Conditions 2–5)

Tests (Section 3.5.3):
  Test 1 : One-way RM-ANOVA over 80 per-class AP values, Conditions 2–5
           (Greenhouse-Geisser corrected; Tukey/Bonferroni post-hoc).
  Test 2 : Paired t-test Condition 3 (Dirichlet) vs Condition 4 (Fisher).
  Test 3 : Two-way RM-ANOVA: strategy (Dirichlet vs Fisher) × component
           (cls, bbox, obj), applied to learned coefficient magnitudes.

Decision rule: H03 rejected if ≥ 1 pairwise contrast p < 0.05 AND ΔmAP ≥ 0.5 pp.

Input files:
  results/phase3_soup_results.json
    Keys required: condition_2 … condition_5, each with "per_class_ap" (len 80).
    Keys for Test 3: condition_3 and condition_4 must additionally contain
      "coefficients": {"cls": [...], "bbox": [...], "obj": [...]}
      where each list has N=6 values (one per ingredient model).
"""

"""
RQ3 / H3 — Coefficient learning strategy comparison (Conditions 2–5)
per_class_ap format: [[class_name, AP, AR], ...] — AP (index 1) used, AR discarded.
"""

import json
import pathlib
import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
SOUP_FILE = RESULTS_DIR / "phase3_soup_results.json"
N_CLASSES = 80
ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.5


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


def greenhouse_geisser_epsilon(data: np.ndarray) -> float:
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
    return float(np.clip(numerator / denominator, 1 / (k - 1), 1.0))


def rm_anova(data: np.ndarray, labels: list):
    n, k = data.shape
    grand_mean = data.mean()
    ss_between = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)
    ss_subjects = k * np.sum((data.mean(axis=1) - grand_mean) ** 2)
    ss_error = np.sum((data - grand_mean) ** 2) - ss_between - ss_subjects
    df_b, df_e = k - 1, (n - 1) * (k - 1)
    F = (ss_between / df_b) / (ss_error / df_e)
    p = stats.f.sf(F, df_b, df_e)
    eps = greenhouse_geisser_epsilon(data)
    p_gg = stats.f.sf(F, df_b * eps, df_e * eps)
    eta_sq = ss_between / (ss_between + ss_error)
    print(f"  F({df_b}, {df_e}) = {F:.4f}, p = {p:.4f}")
    print(f"  GG epsilon = {eps:.4f}  →  p_GG = {p_gg:.4f}  |  η² = {eta_sq:.4f}")
    return {"F": F, "p": p, "p_gg": p_gg, "eta_sq": eta_sq, "epsilon": eps}


def post_hoc_pairwise(data: np.ndarray, labels: list):
    from itertools import combinations
    pairs = list(combinations(range(len(labels)), 2))
    n_comp = len(pairs)
    results = []
    for i, j in pairs:
        d = data[:, j] - data[:, i]
        t, p = stats.ttest_rel(data[:, j], data[:, i])
        p_bonf = min(p * n_comp, 1.0)
        mean_d = d.mean()
        print(f"  {labels[i]} vs {labels[j]}: t = {t:.4f}, p = {p:.4f}, "
              f"p_Bonf = {p_bonf:.4f}, mean Δ = {mean_d:+.4f}")
        results.append({"A": labels[i], "B": labels[j], "t": t, "p": p,
                        "p_bonferroni": p_bonf, "mean_diff": mean_d})
    return results


def main():
    with open(SOUP_FILE) as f:
        soup = json.load(f)

    raw_keys = ["condition_2", "condition_3", "condition_4", "condition_5"]
    labels = ["Condition2", "Condition3", "Condition4", "M5"]

    arrays, ref_names = {}, None
    for k, lbl in zip(raw_keys, labels):
        arr, names = extract_ap_array(soup[k]["per_class_ap"], field_name=f"{k}.per_class_ap")
        if ref_names is None:
            ref_names = names
        else:
            check_class_order(ref_names, names, "condition_2", k)
        arrays[lbl] = arr

    data_matrix = np.column_stack([arrays[l] for l in labels])

    print("\n" + "="*60)
    print("TEST 1 — One-way RM-ANOVA: Conditions 2, 3, 4, M5")
    print("="*60)
    anova_res = rm_anova(data_matrix, labels)

    print("\n--- Bonferroni-corrected pairwise post-hoc contrasts ---")
    posthoc_res = post_hoc_pairwise(data_matrix, labels)

    max_diff = max(abs(r["mean_diff"]) for r in posthoc_res)
    sig_contrast = any(r["p_bonferroni"] < ALPHA for r in posthoc_res)
    decision_h3 = ("REJECT H03" if sig_contrast and max_diff >= PRACTICAL_THRESHOLD
                   else "FAIL TO REJECT H03")
    print(f"\n  Decision: {decision_h3}  (max |Δ| = {max_diff:.4f} pp, sig_contrast = {sig_contrast})")

    print("\n" + "="*60)
    print("TEST 2 — Paired t-test: Condition 3 (Dirichlet) vs Condition 4 (Fisher)")
    print("="*60)
    c3, c4 = arrays["Condition3"], arrays["Condition4"]
    t2, p2 = stats.ttest_rel(c4, c3)
    diff2 = c4 - c3
    d2 = diff2.mean() / diff2.std(ddof=1)
    print(f"  t({len(diff2)-1}) = {t2:.4f}, p = {p2:.4f}")
    print(f"  Mean Δ (Fisher − Dirichlet) = {diff2.mean():+.4f} pp")
    print(f"  Cohen's d = {d2:.4f}")

    print("\n" + "="*60)
    print("TEST 3 — 2-way RM-ANOVA: strategy × component (coefficient magnitudes)")
    print("="*60)
    coef3 = soup["condition_3"].get("coefficients", None)
    coef4 = soup["condition_4"].get("coefficients", None)
    if coef3 and coef4:
        components = ["cls", "bbox", "obj"]
        coef_matrix = np.array([[coef3[c] for c in components],
                                [coef4[c] for c in components]])
        coef_matrix = np.transpose(coef_matrix, (2, 0, 1))
        try:
            import pingouin as pg
            import pandas as pd
            N = coef_matrix.shape[0]
            records = []
            for subj in range(N):
                for s_i, strat in enumerate(["Dirichlet", "Fisher"]):
                    for c_i, comp in enumerate(components):
                        records.append({"subject": subj, "strategy": strat,
                                        "component": comp, "coef": coef_matrix[subj, s_i, c_i]})
            df_long = pd.DataFrame(records)
            aov2 = pg.rm_anova(dv="coef", within=["strategy", "component"],
                               subject="subject", data=df_long, detailed=True)
            print(aov2.to_string())
        except ImportError:
            print("  pingouin not available; reporting marginal effects.")
            strat_d_mean = coef_matrix[:, 0, :].mean(axis=1)
            strat_f_mean = coef_matrix[:, 1, :].mean(axis=1)
            t_s, p_s = stats.ttest_rel(strat_f_mean, strat_d_mean)
            print(f"  Strategy main effect: t = {t_s:.4f}, p = {p_s:.4f}")
            for c_i, comp in enumerate(components):
                comp_vals = coef_matrix[:, :, c_i].mean(axis=1)
                print(f"  Component {comp}: mean coef = {comp_vals.mean():.4f} ± {comp_vals.std():.4f}")
    else:
        print("  'coefficients' key not found; Test 3 skipped.")

    out = RESULTS_DIR / "h3_rq3_results.json"
    with open(out, "w") as f:
        json.dump({"test1_anova": anova_res, "test1_posthoc": posthoc_res,
                   "decision": decision_h3,
                   "test2_paired_t": {"t": float(t2), "p": float(p2), "cohens_d": float(d2)}},
                  f, indent=2, default=float)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()