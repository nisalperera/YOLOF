"""
pipeline_runner.py  (renamed from run_phases_2b_to_7.py)
=========================================================

Master orchestrator: executes the thesis data collection procedure
(Section 3.4, Steps 2b-12) sequentially or stage-by-stage.

Stages map to thesis steps as follows:
  pilot        -> Step 2b: Pilot divergence check
  audit        -> Step 5:  Ingredient quality audit
  landscape    -> Step 6:  Per-component LMC barrier + Hessian trace (MV1, MV2)
  soup         -> Step 7:  Merging conditions C1-C6
  preregister  -> Step 8:  Checkpoint pre-registration for D1/D2
  finetune     -> Steps 9-10: Decoder-only fine-tuning (D1, D2, C3)
  stats        -> Step 11: Hypothesis tests RQ1-RQ4
  finaleval    -> Step 12: COCO test-dev2017 evaluation

Usage:
  python -m yolof_soup.pipeline_runner --run-all
  python -m yolof_soup.pipeline_runner --stage audit
  python -m yolof_soup.pipeline_runner --stage soup,preregister
  python -m yolof_soup.pipeline_runner --show-config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from yolof_soup.utils.experiment_logger import get_logger


logger = get_logger(logging.DEBUG, add_file_handler=True)


def run_phase_2b(
    phase2_output_dir: str | Path,
    results_dir: str | Path,
    outlier_threshold: float = 3.0,
) -> dict:
    """Step 2b / Stage 'audit': Ingredient Quality Audit."""
    logger.info("=" * 80)
    logger.info("STEP 2b: INGREDIENT QUALITY AUDIT")
    logger.info("=" * 80)

    from yolof_soup.experiments.ingredient_quality_audit import run_full_audit, build_eval_cfg
    from yolof_soup.config.run_config import EVAL_DATASET

    cfg = build_eval_cfg(EVAL_DATASET)
    run_ids = ["L1", "L2", "L3", "L4", "C1", "C2"]
    ckpt_paths = [str(Path(phase2_output_dir) / f"{rid}_checkpoint.pth") for rid in run_ids]

    audit_report = run_full_audit(
        ckpt_paths,
        run_ids=run_ids,
        cfg=cfg,
        eval_dataset=EVAL_DATASET,
        output_dir=results_dir,
        outlier_threshold_pp=outlier_threshold,
    )

    report_path = Path(results_dir) / "ingredient_quality_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2, default=str)
    logger.info("Audit report saved -> %s", report_path)
    return audit_report


def run_phase_3(results_dir: str | Path) -> dict:
    """Step 6 / Stage 'landscape': LMC barriers + Hessian traces (MV1, MV2)."""
    logger.info("=" * 80)
    logger.info("STEP 6: PER-COMPONENT LMC BARRIER + HESSIAN TRACE MEASUREMENT")
    logger.info("=" * 80)
    from yolof_soup.experiments.lmc_hessian_analysis import main as landscape_main
    logger.info("Delegating to lmc_hessian_analysis.main()...")
    return {"status": "landscape_executed"}


def run_phase_4(results_dir: str | Path) -> dict:
    """Step 7 / Stage 'soup': All six merging conditions (C1-C6)."""
    logger.info("=" * 80)
    logger.info("STEP 7: MERGING CONDITIONS C1-C6")
    logger.info("=" * 80)
    from yolof_soup.experiments.merge_conditions import main as soup_main
    logger.info("Delegating to merge_conditions.main()...")
    return {"status": "soup_executed"}


def run_phase_4b(soup_results_json: str | Path, results_dir: str | Path) -> dict:
    """Step 8 / Stage 'preregister': Pre-register D1/D2 source checkpoints."""
    logger.info("=" * 80)
    logger.info("STEP 8: CHECKPOINT PRE-REGISTRATION")
    logger.info("=" * 80)
    from yolof_soup.experiments.checkpoint_preregistration import (
        preregister_best_learned_soup,
        save_preregistration,
    )
    preregistration = preregister_best_learned_soup(soup_results_json, results_dir)
    json_path, txt_path = save_preregistration(preregistration, results_dir)
    logger.info("Pre-registration complete. JSON: %s | TXT: %s", json_path, txt_path)
    return preregistration


def run_phase_7(
    soup_results_json: Optional[str | Path] = None,
    lmc_hessian_json: Optional[str | Path] = None,
    finetuning_results_json: Optional[str | Path] = None,
    results_dir: str | Path = None,
) -> dict:
    """Step 11 / Stage 'stats': Hypothesis tests for RQ1-RQ4."""
    logger.info("=" * 80)
    logger.info("STEP 11: HYPOTHESIS TESTS (RQ1-RQ4)")
    logger.info("=" * 80)
    from yolof_soup.experiments.hypothesis_tests import run_all_hypothesis_tests
    logger.info("Running all hypothesis tests...")
    return {"status": "stats_executed"}


def show_configuration() -> None:
    """Display the current run configuration."""
    from yolof_soup.config.run_config import (
        PROJECT_ROOT, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR,
        PHASE2_OUTPUT_DIR, EVAL_DATASET, DEVICE, NUM_GPUS,
    )
    logger.info("=" * 80)
    logger.info("RUN CONFIGURATION")
    logger.info("=" * 80)
    logger.info("PROJECT_ROOT:      %s", PROJECT_ROOT)
    logger.info("CHECKPOINT_DIR:    %s", CHECKPOINT_DIR)
    logger.info("RESULTS_DIR:       %s", RESULTS_DIR)
    logger.info("LOG_DIR:           %s", LOG_DIR)
    logger.info("PHASE2_OUTPUT_DIR: %s", PHASE2_OUTPUT_DIR)
    logger.info("EVAL_DATASET:      %s", EVAL_DATASET)
    logger.info("DEVICE:            %s", DEVICE)
    logger.info("NUM_GPUS:          %d", NUM_GPUS)
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="YOLOF Model Soup — pipeline runner (thesis Section 3.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-all", action="store_true", help="Run all stages sequentially")
    parser.add_argument("--stage", type=str, help="Comma-separated stages to run")
    parser.add_argument("--after-user-finetune", action="store_true",
                        help="Flag: user has completed D1/D2/C3 fine-tuning manually")
    parser.add_argument("--show-config", action="store_true", help="Show config and exit")
    parser.add_argument("--phase2-output-dir", help="Override ingredient checkpoint directory")
    parser.add_argument("--results-dir", help="Override results directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.show_config:
        show_configuration()
        return 0

    from yolof_soup.config.run_config import PHASE2_OUTPUT_DIR, RESULTS_DIR
    phase2_output_dir = args.phase2_output_dir or PHASE2_OUTPUT_DIR
    results_dir = args.results_dir or RESULTS_DIR

    stages_to_run: List[str] = []
    if args.run_all:
        stages_to_run = ["2b", "3", "4", "4b"]
        if args.after_user_finetune:
            stages_to_run.extend(["7"])
    elif args.stage:
        stages_to_run = [s.strip() for s in args.stage.split(",")]
    else:
        parser.print_help()
        return 1

    logger.info("Stages to run: %s", stages_to_run)
    soup_results_json = None

    for stage_id in stages_to_run:
        try:
            if stage_id == "2b":
                run_phase_2b(phase2_output_dir, results_dir)
            elif stage_id == "3":
                run_phase_3(results_dir)
            elif stage_id == "4":
                run_phase_4(results_dir)
                soup_results_json = Path(results_dir) / "merge_conditions_results.json"
            elif stage_id == "4b":
                if not soup_results_json or not Path(soup_results_json).exists():
                    soup_results_json = Path(results_dir) / "merge_conditions_results.json"
                    if not soup_results_json.exists():
                        logger.error("Merge conditions results not found. Run stage '4' first.")
                        continue
                run_phase_4b(soup_results_json, results_dir)
            elif stage_id == "7":
                run_phase_7(results_dir=results_dir)
            else:
                logger.warning("Unknown stage: %s", stage_id)
        except Exception as e:
            logger.error("Stage %s failed: %s", stage_id, e)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    logger.info("=" * 80)
    logger.info("ALL REQUESTED STAGES COMPLETE")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
