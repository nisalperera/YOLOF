"""
quality_audit.py
================

Phase 2b: Ingredient Quality Audit

After Phase 2a (pool training L1-L4, C1-C2), audit all 6 ingredient checkpoints:
  1. Load each ingredient checkpoint
  2. Evaluate on COCO val2017 (held-out evaluation split)
  3. Extract mAP50:95, mAP50, AR@100, per-class AP
  4. Compare against pool maximum
  5. Flag any checkpoint > 3 percentage points below max for review
  6. Save audit report: phase2b_audit_report.json

Output:
  - phase2b_audit_report.json: per-run metrics + pass/fail status
  - phase2b_audit_summary.txt: human-readable summary

Run: python -m yolof_soup.experiments.quality_audit
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from detectron2.modeling import build_model

from yolof_soup.config.experiment_config import RESULTS_DIR, EVAL_DATASET, build_eval_cfg
from yolof_soup.config.experiment_registry import get_run_specs
from yolof_soup.utils.checkpoint_utils import load_state
from yolof_soup.utils.eval_utils import compute_coco_map, extract_per_class_ap
from yolof_soup.utils.inference import InferenceWrapper
from yolof_soup.utils.logging_utils import setup_logging

logger = setup_logging(level=logging.INFO, filename="phase2b_quality_audit.log", use_stdout=True)

# ─────────────────────────────────────────────────────────────────────────────
# Ingredient Audit: Load, Evaluate, Report
# ─────────────────────────────────────────────────────────────────────────────

def audit_single_ingredient(
    run_id: str,
    checkpoint_path: str,
    cfg,
    eval_dataset: str = EVAL_DATASET,
    output_dir: str | Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Evaluate one ingredient checkpoint; return metrics dict.

    Args:
        run_id:            Run identifier (e.g., "L1", "C2")
        checkpoint_path:   Full path to checkpoint file
        cfg:               Detectron2 CfgNode (merged, ready to build model)
        eval_dataset:      Detectron2 dataset name for evaluation
        output_dir:        Directory for intermediate results

    Returns:
        Dict with keys: run_id, checkpoint_path, mAP50:95, mAP50, AR@100,
                       per_class_ap (list of 80), status (pass/fail/warning)
    """

    logger.info("[%s] Evaluating ingredient checkpoint: %s", run_id, checkpoint_path)

    # Load checkpoint state-dict
    if not os.path.exists(checkpoint_path):
        logger.error("[%s] Checkpoint not found: %s", run_id, checkpoint_path)
        return {
            "run_id": run_id,
            "checkpoint_path": str(checkpoint_path),
            "mAP50:95": None,
            "mAP50": None,
            "AR@100": None,
            "per_class_ap": {},
            "status": "MISSING",
            "error": "Checkpoint file not found",
        }
        
    predictor = InferenceWrapper(cfg)  # Rebuild model to ensure correct device, etc.

    # Evaluate
    try:
        tag = f"audit_{run_id}"
        results = compute_coco_map(predictor, cfg, eval_dataset, output_dir, tag=tag)
        map_50_95 = results.get("AP", None)
        map_50 = results.get("AP50", None)
        ar_100 = results.get("AR-maxDets=100", None)

        logger.info(
            "[%s] Evaluation complete — mAP50:95=%.4f  mAP50=%.4f  AR@100=%.4f",
            run_id, map_50_95 or 0.0, map_50 or 0.0, ar_100 or 0.0,
        )

        per_class_ap = extract_per_class_ap(results)

        return {
            "run_id": run_id,
            "checkpoint_path": str(checkpoint_path),
            "mAP50:95": map_50_95,
            "mAP50": map_50,
            "AR@100": ar_100,
            "per_class_ap": per_class_ap,
            "status": "OK",
            "error": None,
        }
    except Exception as e:
        logger.error("[%s] Evaluation failed: %s", run_id, e, exc_info=True)
        return {
            "run_id": run_id,
            "checkpoint_path": str(checkpoint_path),
            "mAP50:95": None,
            "mAP50": None,
            "AR@100": None,
            "per_class_ap": {},
            "status": "EVAL_ERROR",
            "error": str(e),
        }


def run_full_audit(
    ingredient_checkpoint_paths: List[str],
    run_ids: Optional[List[str]] = None,
    cfgs: Optional[List] = None,
    eval_dataset: str = EVAL_DATASET,
    output_dir: str | Path = RESULTS_DIR,
    outlier_threshold_pp: float = 3.0,  # percentage points below max
) -> Dict[str, Any]:
    """
    Audit all ingredients; return aggregated report with pass/fail flags.

    Args:
        ingredient_checkpoint_paths: List of checkpoint file paths (length=N_INGREDIENTS)
        run_ids:                     Run identifiers (e.g., ["L1", "L2", ..., "C2"])
        cfgs:                         List of Detectron2 CfgNodes
        eval_dataset:                Dataset name for evaluation
        output_dir:                  Results directory
        outlier_threshold_pp:        mAP threshold (pp below max) for flagging

    Returns:
        Dict with "results" (list of per-run audits), "summary" (aggregates),
        "pool_max_map", "passed_count", "flagged_count", etc.
    """
    if cfgs is None:
        cfgs = [build_eval_cfg(EVAL_DATASET)] * len(ingredient_checkpoint_paths)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_ids = run_ids or [f"R{i}" for i in range(len(ingredient_checkpoint_paths))]
    if len(run_ids) != len(ingredient_checkpoint_paths):
        raise ValueError("run_ids and ingredient_checkpoint_paths must have same length")

    logger.info("="*80)
    logger.info("PHASE 2B: INGREDIENT QUALITY AUDIT")
    logger.info("="*80)
    logger.info("Auditing %d ingredients...", len(ingredient_checkpoint_paths))

    # Audit each ingredient
    audit_results = []
    for run_id, cfg, ckpt_path in zip(run_ids, cfgs, ingredient_checkpoint_paths):
        result = audit_single_ingredient(run_id, ckpt_path, cfg, eval_dataset, output_dir)
        audit_results.append(result)

    # Extract valid mAP scores (skip errors)
    valid_maps = [
        r["mAP50:95"] for r in audit_results
        if r["status"] == "OK" and r["mAP50:95"] is not None
    ]

    if not valid_maps:
        logger.warning("No valid ingredients found!")
        pool_max = None
        passed_ids = []
        flagged_ids = []
    else:
        pool_max = max(valid_maps)

        # Flag outliers
        passed_ids = [r["run_id"] for r in audit_results if r["status"] == "OK"]
        flagged_ids = [
            r["run_id"] for r in audit_results
            if r["status"] == "OK" and pool_max - r["mAP50:95"] >= outlier_threshold_pp
        ]

    # Aggregate summary
    summary = {
        "pool_max_map50_95": pool_max,
        "outlier_threshold_pp": outlier_threshold_pp,
        "total_ingredients": len(ingredient_checkpoint_paths),
        "passed_count": len(passed_ids),
        "flagged_count": len(flagged_ids),
        "error_count": len([r for r in audit_results if r["status"] != "OK"]),
        "passed_run_ids": passed_ids,
        "flagged_run_ids": flagged_ids,
    }

    # Log summary
    logger.info("="*80)
    logger.info("AUDIT SUMMARY")
    logger.info("="*80)
    logger.info("Pool max mAP50:95: %.4f", pool_max or 0.0)
    logger.info("Passed: %d  Flagged: %d  Errors: %d", summary["passed_count"],
                summary["flagged_count"], summary["error_count"])
    for r in audit_results:
        status_str = f"[{r['status']}]"
        if r["status"] == "OK":
            logger.info("  %-4s %s mAP50:95=%.4f", r["run_id"], status_str, r["mAP50:95"] or 0.0)
        else:
            logger.info("  %-4s %s %s", r["run_id"], status_str, r["error"] or "Unknown error")

    return {"results": audit_results, "summary": summary}


def main():
    """
    Entry point: run full audit from Phase 2a outputs.
    
    Expected Phase 2a outputs:
      - L1, L2, L3, L4: full finetune runs (RTX 5070 Ti)
      - C1, C2: full finetune runs (RTX 5090)
      - All saved to PHASE2_OUTPUT_DIR
    """
    import argparse
    from yolof_soup.config.experiment_config import PHASE2_OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Phase 2b: Ingredient Quality Audit")
    parser.add_argument(
        "--phase2-output-dir",
        default=PHASE2_OUTPUT_DIR,
        help="Directory containing Phase 2a ingredient checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory to save audit results",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.0,
        help="mAP threshold (pp) below pool max for flagging outliers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Build config
    # cfg = build_eval_cfg(EVAL_DATASET)

    # Construct paths for all 6 ingredients (L1-L4, C1-C2)
    # Adjust this based on your actual checkpoint naming/locations
    run_registry = get_run_specs()
    ingredient_runs = [r for r in run_registry if r.role == "ingredient"]

    ckpt_paths = []
    run_ids = []
    cfgs = []
    for run_spec in ingredient_runs:
        ckpt_path = Path(args.phase2_output_dir) / f"{run_spec.run_name}/model_best.pth"
        ckpt_paths.append(str(ckpt_path))
        run_ids.append(run_spec.run_id)
        cfg = build_eval_cfg(EVAL_DATASET, cfg_file=Path(args.phase2_output_dir) / f"{run_spec.run_name}/config.yaml", weights_path=ckpt_path)
        cfgs.append(cfg)
        logger.info("Will audit: %s → %s", run_spec.run_id, ckpt_path)

    # Run audit
    audit_report = run_full_audit(
        ckpt_paths,
        run_ids=run_ids,
        cfgs=cfgs,
        eval_dataset=EVAL_DATASET,
        output_dir=args.results_dir,
        outlier_threshold_pp=args.outlier_threshold,
    )

    # Save JSON report
    report_path = Path(args.results_dir) / "phase2b_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2, default=str)
    logger.info("Audit report saved → %s", report_path)

    # Save text summary
    summary_path = Path(args.results_dir) / "phase2b_audit_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PHASE 2B: INGREDIENT QUALITY AUDIT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        summary = audit_report["summary"]
        f.write(f"Pool max mAP50:95:  {summary['pool_max_map50_95']:.4f}\n")
        f.write(f"Outlier threshold:  {summary['outlier_threshold_pp']:.1f} pp\n")
        f.write(f"Total ingredients:  {summary['total_ingredients']}\n")
        f.write(f"Passed:             {summary['passed_count']}\n")
        f.write(f"Flagged:            {summary['flagged_count']}\n")
        f.write(f"Errors:             {summary['error_count']}\n\n")
        f.write("Passed Run IDs:\n")
        for rid in summary["passed_run_ids"]:
            f.write(f"  - {rid}\n")
        if summary["flagged_run_ids"]:
            f.write("\nFlagged Run IDs (review before proceeding):\n")
            for rid in summary["flagged_run_ids"]:
                f.write(f"  - {rid}\n")
    logger.info("Summary saved → %s", summary_path)

    return audit_report


if __name__ == "__main__":
    main()
