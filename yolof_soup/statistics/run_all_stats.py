"""
Orchestration entry point -- runs all statistical analyses (RQ1-RQ4 plus
two descriptive-only geometric/calibration analyses) in sequence and writes
a consolidated summary to results/statistical_summary.json.

Updated for the FINAL thesis numbering (Ch.1 Sec.1.5 / Ch.4 Preliminary
Note): H1/RQ1 static learning, H2/RQ2 strategy equivalence, H3/RQ3 M6 vs
M5, H4/RQ4 full pipeline superiority. The former h2_rq2_lmc_barrier_anova.py
and h4b_beta_onesample_ttest.py (LMC geometry, beta calibration) are
descriptive-only and carry no RQ/H label in the final draft.

Usage:
    python -m yolof_soup.statistics.run_all_stats

Pre-requisites:
  One-time setup (before running H1/H2/H3/H4):
    python -m yolof_soup.statistics.generate_coco_bootstrap_manifest \\
      --ground-truth results/ground_truth_heldout.json \\
      --predictions results/predictions/*.json \\
      --iterations 5000 --seed 42 \\
      --output-dir results/bootstrap_manifest

  Config files (see each script's docstring for exact keys):
    configs/h1_predictions.json
    configs/h2_predictions.json
    configs/h3_predictions.json
    configs/h4_predictions.json

  Descriptive-only analyses also need:
    results/phase4_barrier_results.json  (LMC geometry)
    results/phase3_soup_results.json     (beta calibration, coefficient magnitudes)

Optional:
    pip install pycocotools numpy scipy
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import traceback

RESULTS_DIR = pathlib.Path("results")
BOOTSTRAP_IMAGE_IDS_FILE = RESULTS_DIR / "bootstrap_manifest" / "bootstrap_image_ids.npy"

# Ordered list of (module_path, display_name, is_formal_hypothesis)
MODULES = [
    ("yolof_soup.statistics.h1_rq1_static_learning_effect", "RQ1/H1 -- Static Learning Effect (paired image-level bootstrap)", True),
    ("yolof_soup.statistics.h2_rq2_strategy_equivalence", "RQ2/H2 -- Strategy Equivalence, Conditions 2-6 (paired image-level bootstrap)", True),
    ("yolof_soup.statistics.h3_rq3_m6_vs_m5", "RQ3/H3 -- M6 vs M5 (paired image-level bootstrap)", True),
    ("yolof_soup.statistics.h4_rq4_full_pipeline_superiority", "RQ4/H4 -- Full Pipeline Superiority (paired image-level bootstrap)", True),
    ("yolof_soup.statistics.descriptive_lmc_barrier_geometry", "Descriptive -- LMC Barrier Geometry (formerly H2, no longer a formal hypothesis)", False),
    ("yolof_soup.statistics.descriptive_beta_calibration", "Descriptive -- Beta Temperature Calibration (formerly H4b, no longer a formal hypothesis)", False),
]

RESULT_FILES = [
    "h1_rq1_results.json",
    "h2_rq2_results.json",
    "h3_rq3_results.json",
    "h4_rq4_results.json",
    "descriptive_lmc_barrier_geometry_results.json",
    "descriptive_beta_calibration_results.json",
]


def check_bootstrap_manifest() -> None:
    print("=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)
    if BOOTSTRAP_IMAGE_IDS_FILE.is_file():
        print(f"  OK: found bootstrap manifest at {BOOTSTRAP_IMAGE_IDS_FILE}")
    else:
        print(
            f"  WARNING: {BOOTSTRAP_IMAGE_IDS_FILE} not found. H1, H2, H3, and H4 "
            "scripts require this file (generated once via "
            "generate_coco_bootstrap_manifest.py, seed=42) and will be SKIPPED "
            "below with a FileNotFoundError until it exists. The two descriptive-"
            "only analyses (LMC geometry, beta calibration) do not need it."
        )
    print()


def run_module(module_path: str, display_name: str) -> dict:
    sep = "=" * 70
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
    return {"module": module_path, "name": display_name, "status": status, "error": error_msg}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    check_bootstrap_manifest()

    summary = []
    for module_path, display_name, _ in MODULES:
        result = run_module(module_path, display_name)
        summary.append(result)

    consolidated: dict = {"run_status": summary}
    for fname in RESULT_FILES:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                consolidated[fname.replace(".json", "")] = json.load(f)

    out = RESULTS_DIR / "statistical_summary.json"
    with open(out, "w") as f:
        json.dump(consolidated, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("STATISTICAL TESTING COMPLETE")
    print("=" * 70)
    ok = sum(1 for r in summary if r["status"] == "OK")
    skipped = sum(1 for r in summary if r["status"] == "SKIPPED")
    errors = sum(1 for r in summary if r["status"] == "ERROR")
    print(f"  OK: {ok}  |  Skipped (missing data): {skipped}  |  Errors: {errors}")
    print(f"  Consolidated report -> {out}")
    print(
        "\n  NOTE: h1_rq1_component_vs_global.py, h3_rm_anova_conditions2to5.py, "
        "h4a_paired_ttest_M6_vs_M5.py, h4c_finetune_gain.py, h2_rq2_lmc_barrier_anova.py, "
        "and h4b_beta_onesample_ttest.py are SUPERSEDED by the modules run above and are "
        "not invoked by this orchestrator. Recommend deleting them once you've verified "
        "the renamed versions produce expected output."
    )

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
