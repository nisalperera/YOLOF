"""
Orchestration entry point — runs all statistical tests (RQ1–RQ4) in sequence
and writes a consolidated summary to results/statistical_summary.json.

Usage:
    python yolof_soup/statistics/run_all_stats.py

Pre-requisites (all produced by earlier pipeline phases):
    results/phase1_ingredient_results.json
    results/phase3_soup_results.json
    results/phase4_barrier_results.json
    results/phase4_finetune_results.json

Optional (speeds up imports):
    pip install pingouin   # for proper RM-ANOVA with GG correction + post-hoc
"""

import importlib
import json
import pathlib
import sys
import traceback

RESULTS_DIR = pathlib.Path("results")

# Ordered list of (module_path, display_name)
MODULES = [
    ("yolof_soup.statistics.h1_rq1_component_vs_global",    "RQ1/H1 — Component vs Global"),
    ("yolof_soup.statistics.h2_rq2_lmc_barrier_anova",      "RQ2/H2 — LMC Barrier ANOVA"),
    ("yolof_soup.statistics.h3_rm_anova_conditions2to5",    "RQ3/H3 — Coefficient Strategy ANOVA"),
    ("yolof_soup.statistics.h4a_paired_ttest_M6_vs_M5",     "RQ4/H4a — M6 vs M5"),
    ("yolof_soup.statistics.h4b_beta_onesample_ttest",      "RQ4/H4b — β one-sample t-tests"),
    ("yolof_soup.statistics.h4c_finetune_gain",             "RQ4/H4c — Fine-tuning gain"),
]


def run_module(module_path: str, display_name: str) -> dict:
    sep = "="*70
    print(f"\n{sep}")
    print(f"  {display_name}")
    print(sep)
    status = "OK"
    error_msg = None
    try:
        mod = importlib.import_module(module_path)
        mod.main()
    except FileNotFoundError as e:
        status = "SKIPPED"
        error_msg = f"Missing input file: {e}"
        print(f"  [SKIP] {error_msg}")
    except Exception:  # noqa: BLE001
        status = "ERROR"
        error_msg = traceback.format_exc()
        print(f"  [ERROR]\n{error_msg}")
    return {"module": module_path, "name": display_name,
            "status": status, "error": error_msg}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for module_path, display_name in MODULES:
        result = run_module(module_path, display_name)
        summary.append(result)

    # Collect individual result files
    consolidated: dict = {"run_status": summary}
    for fname in ["h1_rq1_results.json", "h2_rq2_results.json",
                  "h3_rq3_results.json", "h4a_results.json",
                  "h4b_results.json",    "h4c_results.json"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                consolidated[fname.replace(".json", "")] = json.load(f)

    out = RESULTS_DIR / "statistical_summary.json"
    with open(out, "w") as f:
        json.dump(consolidated, f, indent=2, default=float)

    print("\n" + "="*70)
    print("STATISTICAL TESTING COMPLETE")
    print("="*70)
    ok = sum(1 for r in summary if r["status"] == "OK")
    skipped = sum(1 for r in summary if r["status"] == "SKIPPED")
    errors = sum(1 for r in summary if r["status"] == "ERROR")
    print(f"  OK: {ok}  |  Skipped (missing data): {skipped}  |  Errors: {errors}")
    print(f"  Consolidated report → {out}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
