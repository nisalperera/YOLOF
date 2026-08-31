"""
Descriptive-only per-component loss landscape geometry (LMC barriers +
Hessian traces). NOT a formal hypothesis in the final thesis draft.

This is the renamed successor to h2_rq2_lmc_barrier_anova.py (content
provided by the user, unchanged computationally). Ch.4's Preliminary Note
states explicitly: "the per-component loss landscape geometry analysis...
formerly H2... in the earlier draft, [is] exclusively maintained as
descriptive context" and is "no longer defined as [a] formal hypothesis."
Ch.1 Sec.1.5 confirms the same. Chapter 3 Sec.3.5 still labels this
analysis "the statistical test for RQ2 and H2" with a formal Bonferroni-
style decision -- that description is stale and should be rewritten to
match this descriptive framing (flagged in STATISTICS_UPDATE_NOTES.md).

h2_rq2_lmc_barrier_anova.py is left in the repo unchanged for now;
recommend deleting it once this file is verified.

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
Only the framing: the RM-ANOVA, post-hoc contrasts, and the cls-vs-reg
paired t-test are still computed (this sampling unit -- 15 ingredient-
model pairs -- is legitimate, unrelated to the per-class-AP
pseudo-replication problem). What changed is that the "REJECT H02" /
"FAIL TO REJECT H02" formal decision label is REMOVED, since this is not
a formal hypothesis in the final draft. Results are reported descriptively,
matching Chapter 4's own reporting style ("a moderate effect size that
falls short of conventional significance").

ADDITIONAL CAVEAT (not in the original script)
------------------------------------------------
The 15 pairs are not fully independent: each of the N=6 ingredient models
contributes to 5 different pairs. This should be reported as a
limitation if this analysis is discussed inferentially in Chapter 5.

Expected barrier JSON schema:
{
  "0": {
    "pair_0102": {
      "backbone_encoder": ...,
      "cls_head": ...,
      "reg_head": ...,
      "shared": ...,
      "full_model": ...
    },
    ... 15 pairs
  },
  "1": { ... }, ..., "5": { ... }
}
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy import stats

RESULTS_DIR = pathlib.Path("results")
BARRIER_FILE = RESULTS_DIR / "phase4_barrier_results.json"
RAW_AUDIT_FILE = RESULTS_DIR / "descriptive_lmc_barrier_raw_90rows.json"
PAIR_AVG_FILE = RESULTS_DIR / "descriptive_lmc_barrier_pair_averaged_15rows.json"
SUMMARY_FILE = RESULTS_DIR / "descriptive_lmc_barrier_geometry_results.json"

COMPONENTS = ["backbone_encoder", "cls_head", "reg_head", "objectness_module"]
FULL_MODEL_KEY = "full_model"
EXPECTED_BASES = 6
EXPECTED_PAIRS = 15


def normalize_pair_record(rec: dict) -> dict:
    obj_val = rec.get("objectness_module", rec.get("shared"))
    if obj_val is None:
        raise KeyError("Expected 'objectness_module' or 'shared' in pair record.")
    return {
        "backbone_encoder": float(rec["backbone_encoder"]),
        "cls_head": float(rec["cls_head"]),
        "reg_head": float(rec["reg_head"]),
        "objectness_module": float(obj_val),
        "full_model": float(rec["full_model"]),
    }


def load_raw_barrier_data(path: pathlib.Path):
    with open(path) as f:
        data = json.load(f)

    if len(data) != EXPECTED_BASES:
        raise ValueError(f"Expected {EXPECTED_BASES} base-model groups, found {len(data)}.")

    raw_rows = []
    pair_to_rows = defaultdict(list)

    for base_key, pair_dict in data.items():
        if len(pair_dict) != EXPECTED_PAIRS:
            raise ValueError(f"Base {base_key} should contain {EXPECTED_PAIRS} pairs, found {len(pair_dict)}.")

        for pair_name, rec in pair_dict.items():
            norm = normalize_pair_record(rec)
            row = {"base_model": str(base_key), "pair": pair_name, **norm}
            raw_rows.append(row)
            pair_to_rows[pair_name].append(row)

    if len(raw_rows) != EXPECTED_BASES * EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_BASES * EXPECTED_PAIRS} raw rows, found {len(raw_rows)}.")
    if len(pair_to_rows) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} unique pairs, found {len(pair_to_rows)}.")

    return raw_rows, pair_to_rows


def average_pairs_across_bases(pair_to_rows):
    pair_avg_rows = []
    for pair_name, rows in sorted(pair_to_rows.items()):
        if len(rows) != EXPECTED_BASES:
            raise ValueError(f"{pair_name} should have {EXPECTED_BASES} base-conditioned rows.")
        avg_row = {"pair": pair_name, "n_bases": len(rows)}
        for key in COMPONENTS + [FULL_MODEL_KEY]:
            avg_row[key] = float(np.mean([r[key] for r in rows]))
        pair_avg_rows.append(avg_row)
    return pair_avg_rows


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


def rm_anova_manual(data: np.ndarray):
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

    return {
        "F": float(F), "p": float(p), "p_gg": float(p_gg), "eta_sq": float(eta_sq),
        "epsilon": float(epsilon), "df_between": float(df_between), "df_error": float(df_error),
        "df_gg_between": float(df_gg_between), "df_gg_error": float(df_gg_error),
    }


def posthoc_pairwise_paired(data: np.ndarray, labels: list[str]):
    from itertools import combinations

    pairs = list(combinations(range(len(labels)), 2))
    out = []
    m = len(pairs)
    for i, j in pairs:
        diff = data[:, j] - data[:, i]
        t, p = stats.ttest_rel(data[:, j], data[:, i])
        out.append({
            "A": labels[i], "B": labels[j], "t": float(t), "p": float(p),
            "p_bonferroni": float(min(p * m, 1.0)), "mean_diff": float(diff.mean()),
        })
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows, pair_to_rows = load_raw_barrier_data(BARRIER_FILE)
    pair_avg_rows = average_pairs_across_bases(pair_to_rows)

    with open(RAW_AUDIT_FILE, "w") as f:
        json.dump(raw_rows, f, indent=2)
    with open(PAIR_AVG_FILE, "w") as f:
        json.dump(pair_avg_rows, f, indent=2)

    barrier_matrix = np.array([[row[c] for c in COMPONENTS] for row in pair_avg_rows])

    anova_res = rm_anova_manual(barrier_matrix)
    posthoc_res = posthoc_pairwise_paired(barrier_matrix, COMPONENTS)

    cls_barriers = barrier_matrix[:, COMPONENTS.index("cls_head")]
    reg_barriers = barrier_matrix[:, COMPONENTS.index("reg_head")]
    diff_cls_reg = cls_barriers - reg_barriers
    t_cls_reg, p_cls_reg = stats.ttest_rel(cls_barriers, reg_barriers)
    d_cls_reg = diff_cls_reg.mean() / diff_cls_reg.std(ddof=1)

    full_model_vals = np.array([row["full_model"] for row in pair_avg_rows])
    full_model_summary = {
        "mean": float(full_model_vals.mean()), "std": float(full_model_vals.std(ddof=1)),
        "min": float(full_model_vals.min()), "max": float(full_model_vals.max()),
    }

    print("\n" + "=" * 70)
    print("DESCRIPTIVE per-component loss landscape geometry (NOT a formal hypothesis)")
    print("=" * 70)
    print(f"  RM-ANOVA (descriptive): F = {anova_res['F']:.4f}, p = {anova_res['p']:.4f}, p_GG = {anova_res['p_gg']:.4f}, eta^2 = {anova_res['eta_sq']:.4f}")
    print(f"  cls_head vs reg_head barrier (descriptive): mean diff = {diff_cls_reg.mean():+.4f}, t = {t_cls_reg:.4f}, p = {p_cls_reg:.4f}, d = {d_cls_reg:.4f}")
    print(
        "  These figures are reported descriptively per Ch.1 Sec.1.5 / Ch.4 Preliminary "
        "Note (this analysis is not a formal hypothesis). No alpha=0.05 accept/reject "
        "decision is applied."
    )
    print(
        "  CAVEAT: the 15 pairs are not fully independent -- each of the N=6 ingredient "
        "models contributes to 5 pairs. Treat as a limitation if interpreted further in "
        "Chapter 5."
    )

    summary = {
        "status": "descriptive_only",
        "rationale": (
            "Per Ch.1 Sec.1.5 and Ch.4 Preliminary Note, this analysis (formerly labeled "
            "H2 in an earlier draft) is retained exclusively as descriptive context and is "
            "not a formal hypothesis in the final thesis. Chapter 3 Sec.3.5 still describes "
            "it as 'the statistical test for RQ2 and H2' with a formal decision; that text "
            "is stale and should be revised to match this descriptive framing."
        ),
        "caveat_pair_independence": (
            "The 15 ingredient-model pairs are not fully independent: each of the N=6 "
            "models appears in 5 different pairs."
        ),
        "input_shape": {
            "n_raw_rows": len(raw_rows), "n_unique_pairs": len(pair_avg_rows), "n_bases_per_pair": EXPECTED_BASES,
        },
        "descriptive_barrier_rm_anova": anova_res,
        "descriptive_posthoc": posthoc_res,
        "descriptive_cls_vs_reg": {
            "t": float(t_cls_reg), "p": float(p_cls_reg), "cohens_d": float(d_cls_reg),
            "mean_diff_cls_minus_reg": float(diff_cls_reg.mean()),
        },
        "full_model_descriptive_only": full_model_summary,
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved raw audit table: {RAW_AUDIT_FILE}")
    print(f"Saved pair-averaged table: {PAIR_AVG_FILE}")
    print(f"Saved descriptive summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
