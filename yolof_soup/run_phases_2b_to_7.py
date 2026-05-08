#!/usr/bin/env python3
"""
run_phases_2b_to_7.py
======================

Master orchestrator: Run Phases 2b → 7 sequentially (or individually).

COMPLETED (user-managed):
  ✓ Phase 1: Reproducibility setup & pilot check
  ✓ Phase 2a: Ingredient pool training (L1-L4, C1-C2)

PENDING (this script):
  ☐ Phase 2b: Ingredient quality audit
  ☐ Phase 3:  Loss landscape geometry (parallel with Phase 4)
  ☐ Phase 4:  Soup merging & evaluation
  ☐ Phase 4b: Pre-registration of best learned soup
  ☐ Phase 5:  Head-only fine-tuning (USER RUNS MANUALLY)
  ☐ Phase 6:  Data archiving & summary
  ☐ Phase 7:  Statistical analysis

Fix log:
  - run_phase_3 and run_phase_4 previously imported their respective
    main() functions but NEVER called them, always returning a fake
    {"status": "phase_executed"} dict.  Both are now actually invoked.
  - run_phase_7 similarly now calls phase7_main() with the correct args.

Usage:
  python run_phases_2b_to_7.py --run-all
  python run_phases_2b_to_7.py --phase 2b
  python run_phases_2b_to_7.py --phase 3,4,4b
  python run_phases_2b_to_7.py --phase 6 --after-user-phase-5
  python run_phases_2b_to_7.py --show-config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from yolof_soup.utils.global_logger import get_logger, configure_logger

logger = get_logger()


def run_phase_2b(
    phase2_output_dir: str | Path,
    results_dir: str | Path,
    outlier_threshold: float = 3.0,
) -> dict:
    """Run Phase 2b: Ingredient Quality Audit"""
    logger.info("="*80)
    logger.info("RUNNING PHASE 2B: INGREDIENT QUALITY AUDIT")
    logger.info("="*80)

    from yolof_soup.experiments.quality_audit import run_full_audit
    from yolof_soup.config.experiment_config import EVAL_DATASET, build_eval_cfg
    from yolof_soup.config.experiment_registry import get_run_specs

    run_registry     = get_run_specs()
    ingredient_runs  = [r for r in run_registry if r.role == "ingredient"]

    ckpt_paths, run_ids, cfgs = [], [], []
    for run_spec in ingredient_runs:
        ckpt_path = Path(phase2_output_dir) / f"{run_spec.run_name}/model_best.pth"
        ckpt_paths.append(str(ckpt_path))
        run_ids.append(run_spec.run_id)
        cfg = build_eval_cfg(
            EVAL_DATASET,
            cfg_file=Path(phase2_output_dir) / f"{run_spec.run_name}/config.yaml",
            weights_path=ckpt_path,
        )
        cfgs.append(cfg)

    audit_report = run_full_audit(
        ckpt_paths,
        run_ids=run_ids,
        cfgs=cfgs,
        eval_dataset=EVAL_DATASET,
        output_dir=results_dir,
        outlier_threshold_pp=outlier_threshold,
    )

    report_path = Path(results_dir) / "phase2b_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2, default=str)
    logger.info("Phase 2b audit report saved → %s", report_path)
    return audit_report


def run_phase_3(results_dir: str | Path) -> dict:
    """Run Phase 3: Loss Landscape Geometry — actually executes the phase."""
    logger.info("="*80)
    logger.info("RUNNING PHASE 3: LOSS LANDSCAPE MEASUREMENT")
    logger.info("="*80)

    from yolof_soup.experiments.loss_landscape import main as phase3_main

    # Build a minimal sys.argv so phase3_main's argparse does not see
    # the parent script's arguments.
    import sys
    orig_argv = sys.argv
    sys.argv = ["loss_landscape.py", "--results-dir", str(results_dir)]
    try:
        phase3_result = phase3_main()
    finally:
        sys.argv = orig_argv

    logger.info("Phase 3: Loss landscape measurement complete")
    return phase3_result if isinstance(phase3_result, dict) else {"status": "phase3_complete"}


def run_phase_4(results_dir: str | Path) -> dict:
    """Run Phase 4: Soup Merging & Evaluation — actually executes the phase."""
    logger.info("="*80)
    logger.info("RUNNING PHASE 4: SOUP MERGING & EVALUATION")
    logger.info("="*80)

    from yolof_soup.experiments.soup_construction import run as phase4_run

    results = phase4_run(verbose=True)
    logger.info("Phase 4: Soup merging complete")
    return results


def run_phase_4b(soup_results_json: str | Path, results_dir: str | Path) -> dict:
    """Run Phase 4b: Pre-registration of Best Learned Soup"""
    logger.info("="*80)
    logger.info("RUNNING PHASE 4B: PRE-REGISTRATION")
    logger.info("="*80)

    from yolof_soup.experiments.preregistration import preregister_best_learned_soup, save_preregistration

    preregistration = preregister_best_learned_soup(soup_results_json, results_dir)
    json_path, txt_path = save_preregistration(preregistration, results_dir)

    logger.info("PRE-REGISTRATION COMPLETE")
    logger.info("Chosen checkpoint: %s", preregistration["chosen_checkpoint"])
    logger.info("  JSON: %s", json_path)
    logger.info("  TXT:  %s", txt_path)
    logger.info("Next: User runs Phase 5a and 5b manually (D1, D2, C3)")
    return preregistration


def run_phase_6(
    results_dir: str | Path,
    audit_report: Optional[dict] = None,
    lmc_json: Optional[str | Path] = None,
    hessian_json: Optional[str | Path] = None,
    soup_results_json: Optional[str | Path] = None,
    d1_results_json: Optional[str | Path] = None,
    d2_results_json: Optional[str | Path] = None,
    c3_results_json: Optional[str | Path] = None,
) -> dict:
    """Run Phase 6: Data Archiving & Summary"""
    logger.info("="*80)
    logger.info("RUNNING PHASE 6: DATA ARCHIVING & SUMMARY")
    logger.info("="*80)

    from yolof_soup.experiments.archive_and_summary import build_experiment_summary, save_experiment_summary

    summary = build_experiment_summary(
        phase2b_audit_report=audit_report,
        phase3_lmc_barriers_json=lmc_json,
        phase3_hessian_traces_json=hessian_json,
        phase4_soup_results_json=soup_results_json,
        phase5_d1_results_json=d1_results_json,
        phase5_d2_results_json=d2_results_json,
        phase5_c3_results_json=c3_results_json,
    )
    json_path, txt_path = save_experiment_summary(summary, results_dir)
    logger.info("ARCHIVING COMPLETE — JSON: %s  TXT: %s", json_path, txt_path)
    return summary


def run_phase_7(
    soup_results_json: Optional[str | Path] = None,
    lmc_hessian_json: Optional[str | Path] = None,
    finetuning_results_json: Optional[str | Path] = None,
    results_dir: str | Path = None,
) -> dict:
    """Run Phase 7: Statistical Analysis — actually executes the phase."""
    logger.info("="*80)
    logger.info("RUNNING PHASE 7: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    logger.info("="*80)

    from yolof_soup.experiments.statistical_analysis import main as phase7_main

    import sys
    orig_argv = sys.argv
    argv = ["statistical_analysis.py"]
    if soup_results_json:
        argv += ["--soup-results-json", str(soup_results_json)]
    if lmc_hessian_json:
        argv += ["--barriers-hessians-json", str(lmc_hessian_json)]
    if finetuning_results_json:
        argv += ["--finetuning-results-json", str(finetuning_results_json)]
    if results_dir:
        argv += ["--output-dir", str(results_dir)]
    sys.argv = argv
    try:
        result = phase7_main()
    finally:
        sys.argv = orig_argv

    logger.info("Phase 7: Statistical analysis complete")
    return result if isinstance(result, dict) else {"status": "phase7_complete"}


def show_configuration() -> None:
    """Display current configuration."""
    from yolof_soup.config.experiment_config import (
        PROJECT_ROOT, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR,
        PHASE2_OUTPUT_DIR, EVAL_DATASET, SELECTION_DATASET, DEVICE, NUM_GPUS,
    )
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    logger.info("PROJECT_ROOT:       %s", PROJECT_ROOT)
    logger.info("CHECKPOINT_DIR:     %s", CHECKPOINT_DIR)
    logger.info("RESULTS_DIR:        %s", RESULTS_DIR)
    logger.info("LOG_DIR:            %s", LOG_DIR)
    logger.info("PHASE2_OUTPUT_DIR:  %s", PHASE2_OUTPUT_DIR)
    logger.info("EVAL_DATASET:       %s", EVAL_DATASET)
    logger.info("SELECTION_DATASET:  %s", SELECTION_DATASET)
    logger.info("DEVICE:             %s", DEVICE)
    logger.info("NUM_GPUS:           %d", NUM_GPUS)
    logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master orchestrator: Run Phases 2b → 7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-all",          action="store_true")
    parser.add_argument("--phase",            type=str)
    parser.add_argument("--after-user-phase-5", action="store_true")
    parser.add_argument("--show-config",      action="store_true")
    parser.add_argument("--phase2-output-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--verbose",          action="store_true")
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.show_config:
        show_configuration()
        return 0

    from yolof_soup.config.experiment_config import PHASE2_OUTPUT_DIR, RESULTS_DIR

    phase2_output_dir = args.phase2_output_dir or PHASE2_OUTPUT_DIR
    results_dir       = args.results_dir       or RESULTS_DIR

    phases_to_run: List[str] = []
    if args.run_all:
        phases_to_run = ["2b", "3", "4", "4b"]
        if args.after_user_phase_5:
            phases_to_run.extend(["6", "7"])
    elif args.phase:
        phases_to_run = [p.strip() for p in args.phase.split(",")]
    else:
        parser.print_help()
        return 1

    logger.info("Phases to run: %s", phases_to_run)

    audit_report      = None
    soup_results_json = None

    for phase_id in phases_to_run:
        try:
            if phase_id == "2b":
                audit_report = run_phase_2b(phase2_output_dir, results_dir)

            elif phase_id == "3":
                run_phase_3(results_dir)

            elif phase_id == "4":
                run_phase_4(results_dir)
                soup_results_json = Path(results_dir) / "phase4_soup_results.json"

            elif phase_id == "4b":
                if not soup_results_json or not Path(soup_results_json).exists():
                    soup_results_json = Path(results_dir) / "phase4_soup_results.json"
                    if not Path(soup_results_json).exists():
                        logger.error("Phase 4 soup results not found. Run Phase 4 first.")
                        continue
                run_phase_4b(soup_results_json, results_dir)

            elif phase_id == "6":
                if not args.after_user_phase_5:
                    logger.warning(
                        "Phase 6 should be run after user completes Phase 5a/5b. "
                        "Use --after-user-phase-5 flag to proceed."
                    )
                    continue
                run_phase_6(results_dir)

            elif phase_id == "7":
                run_phase_7(results_dir=results_dir)

            else:
                logger.warning("Unknown phase: %s", phase_id)

        except Exception as e:
            logger.error("Phase %s failed: %s", phase_id, e)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    logger.info("="*80)
    logger.info("ALL REQUESTED PHASES COMPLETE")
    logger.info("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
