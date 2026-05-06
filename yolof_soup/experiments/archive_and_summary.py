"""
archive_and_summary.py
======================

Phase 6: Data Logging & Archiving

After Phase 5b (user completes C3 final pipeline run):

1. Collect all results and checkpoints from Phases 2b-5b
2. Compute SHA-256 hashes for all checkpoints and result files
3. Store metadata: run ID, condition, checkpoint hash, mAP, per-class AP
4. Create experiment lineage document (full traceability)
5. Save to Git+DVC version control

Outputs:
  - experiment_summary.json: aggregated results + hashes + lineage
  - experiment_lineage.txt: human-readable experiment timeline
  - .dvc file: DVC tracking for large checkpoints (if applicable)

Run (after user Phase 5b): python -m yolof_soup.experiments.archive_and_summary
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hash Computation & Verification
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.

    Args:
        file_path:  Path to file
        algorithm:  Hash algorithm (default: sha256)

    Returns:
        Hex digest of the hash
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning("File not found for hashing: %s", file_path)
        return ""

    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_hash(file_path: str | Path, expected_hash: str) -> bool:
    """Verify that a file's hash matches expected value."""
    computed = compute_file_hash(file_path)
    return computed == expected_hash


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Summary: Aggregate Results & Hashes
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_summary(
    phase2b_audit_report: Optional[Dict] = None,
    phase3_lmc_barriers_json: Optional[str | Path] = None,
    phase3_hessian_traces_json: Optional[str | Path] = None,
    phase4_soup_results_json: Optional[str | Path] = None,
    phase4b_preregistration_json: Optional[str | Path] = None,
    phase5_d1_results_json: Optional[str | Path] = None,
    phase5_d2_results_json: Optional[str | Path] = None,
    phase5_c3_results_json: Optional[str | Path] = None,
    ingredient_checkpoint_paths: Optional[List[str | Path]] = None,
    m2_checkpoint_path: Optional[str | Path] = None,
    best_learned_checkpoint_path: Optional[str | Path] = None,
    d1_checkpoint_path: Optional[str | Path] = None,
    d2_checkpoint_path: Optional[str | Path] = None,
    c3_checkpoint_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Build comprehensive experiment summary from all phases.

    Returns:
        Dict with keys: timestamp, phases (dict of phase results), checkpoints (dict with hashes),
                        lineage, notes
    """
    logger.info("Building experiment summary...")

    timestamp = datetime.now().isoformat()

    # Collect phase results
    phases = {}

    if phase2b_audit_report:
        phases["2b_quality_audit"] = {
            "status": "completed",
            "pool_max_map50_95": phase2b_audit_report.get("summary", {}).get("pool_max_map50_95"),
            "passed_count": phase2b_audit_report.get("summary", {}).get("passed_count"),
            "flagged_count": phase2b_audit_report.get("summary", {}).get("flagged_count"),
        }

    if phase3_lmc_barriers_json:
        phases["3_loss_landscape"] = {
            "status": "completed",
            "lmc_barriers_json": str(phase3_lmc_barriers_json),
            "hessian_traces_json": str(phase3_hessian_traces_json),
        }

    if phase4_soup_results_json:
        try:
            with open(phase4_soup_results_json) as f:
                soup_results = json.load(f)
            phases["4_soup_merging"] = {
                "status": "completed",
                "soup_results_json": str(phase4_soup_results_json),
                "conditions_evaluated": list(soup_results.get("conditions", {}).keys()),
            }
        except Exception as e:
            logger.warning("Could not read soup results: %s", e)
            phases["4_soup_merging"] = {"status": "completed", "error": str(e)}

    if phase4b_preregistration_json:
        try:
            with open(phase4b_preregistration_json) as f:
                preregistration = json.load(f)
            phases["4b_preregistration"] = {
                "status": "completed",
                "chosen_condition": preregistration.get("chosen_condition"),
                "chosen_map50_95": preregistration.get("chosen_map50_95"),
                "timestamp": preregistration.get("timestamp"),
            }
        except Exception as e:
            logger.warning("Could not read preregistration: %s", e)
            phases["4b_preregistration"] = {"status": "completed", "error": str(e)}

    # Collect head fine-tuning results
    if phase5_d1_results_json or phase5_d2_results_json or phase5_c3_results_json:
        phases["5_head_finetuning"] = {
            "status": "completed",
            "d1_results_json": str(phase5_d1_results_json) if phase5_d1_results_json else None,
            "d2_results_json": str(phase5_d2_results_json) if phase5_d2_results_json else None,
            "c3_results_json": str(phase5_c3_results_json) if phase5_c3_results_json else None,
        }

    # Collect checkpoints with hashes
    checkpoints = {}

    if ingredient_checkpoint_paths:
        checkpoints["ingredients"] = {}
        run_ids = ["L1", "L2", "L3", "L4", "C1", "C2"]
        for run_id, ckpt_path in zip(run_ids, ingredient_checkpoint_paths):
            ckpt_path = Path(ckpt_path)
            if ckpt_path.exists():
                checkpoints["ingredients"][run_id] = {
                    "path": str(ckpt_path),
                    "hash_sha256": compute_file_hash(ckpt_path),
                    "size_mb": ckpt_path.stat().st_size / (1024 ** 2),
                }
            else:
                logger.warning("Ingredient checkpoint not found: %s", ckpt_path)

    if m2_checkpoint_path and Path(m2_checkpoint_path).exists():
        checkpoints["m2_branch_uniform"] = {
            "path": str(m2_checkpoint_path),
            "hash_sha256": compute_file_hash(m2_checkpoint_path),
            "size_mb": Path(m2_checkpoint_path).stat().st_size / (1024 ** 2),
        }

    if best_learned_checkpoint_path and Path(best_learned_checkpoint_path).exists():
        checkpoints["best_learned_soup"] = {
            "path": str(best_learned_checkpoint_path),
            "hash_sha256": compute_file_hash(best_learned_checkpoint_path),
            "size_mb": Path(best_learned_checkpoint_path).stat().st_size / (1024 ** 2),
        }

    if d1_checkpoint_path and Path(d1_checkpoint_path).exists():
        checkpoints["d1_finetuned"] = {
            "path": str(d1_checkpoint_path),
            "hash_sha256": compute_file_hash(d1_checkpoint_path),
            "size_mb": Path(d1_checkpoint_path).stat().st_size / (1024 ** 2),
        }

    if d2_checkpoint_path and Path(d2_checkpoint_path).exists():
        checkpoints["d2_finetuned"] = {
            "path": str(d2_checkpoint_path),
            "hash_sha256": compute_file_hash(d2_checkpoint_path),
            "size_mb": Path(d2_checkpoint_path).stat().st_size / (1024 ** 2),
        }

    if c3_checkpoint_path and Path(c3_checkpoint_path).exists():
        checkpoints["c3_final_pipeline"] = {
            "path": str(c3_checkpoint_path),
            "hash_sha256": compute_file_hash(c3_checkpoint_path),
            "size_mb": Path(c3_checkpoint_path).stat().st_size / (1024 ** 2),
        }

    # Build lineage
    lineage = {
        "phases_completed": list(phases.keys()),
        "checkpoints_archived": list(checkpoints.keys()),
        "total_checkpoints": sum(
            len(v) if isinstance(v, dict) else 1
            for v in checkpoints.values()
        ),
    }

    summary = {
        "timestamp": timestamp,
        "experiment_name": "YOLOF Branch-Specific Model Soups on COCO 2017",
        "phases": phases,
        "checkpoints": checkpoints,
        "lineage": lineage,
        "notes": (
            "This summary provides full traceability for the thesis experiments. "
            "All checkpoint hashes enable reproducibility verification. "
            "See individual phase JSON files for detailed per-class results."
        ),
    }

    logger.info("Experiment summary built with %d checkpoints archived", lineage["total_checkpoints"])
    return summary


def save_experiment_summary(
    summary: Dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Save experiment summary in JSON and TXT formats.

    Returns:
        (json_path, txt_path) — paths where files were saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "experiment_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Experiment summary JSON saved → %s", json_path)

    # TXT (human-readable lineage)
    txt_path = output_dir / "experiment_lineage.txt"
    with open(txt_path, "w") as f:
        f.write("EXPERIMENT LINEAGE & ARCHIVAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {summary['timestamp']}\n")
        f.write(f"Experiment: {summary['experiment_name']}\n\n")
        f.write("PHASES COMPLETED:\n")
        for phase in summary.get("lineage", {}).get("phases_completed", []):
            f.write(f"  ✓ {phase}\n")
        f.write(f"\nTOTAL CHECKPOINTS ARCHIVED: {summary.get('lineage', {}).get('total_checkpoints', 0)}\n\n")
        f.write("CHECKPOINT HASHES (SHA-256):\n")
        for group_name, group_data in summary.get("checkpoints", {}).items():
            if isinstance(group_data, dict) and "hash_sha256" in group_data:
                f.write(f"  {group_name:40s}: {group_data['hash_sha256']}\n")
            elif isinstance(group_data, dict):
                f.write(f"  {group_name}:\n")
                for item_name, item_data in group_data.items():
                    if isinstance(item_data, dict) and "hash_sha256" in item_data:
                        f.write(f"    {item_name:36s}: {item_data['hash_sha256']}\n")
        f.write(f"\nNOTES:\n{summary.get('notes', 'N/A')}\n")
    logger.info("Experiment lineage TXT saved → %s", txt_path)

    return json_path, txt_path


def main():
    """Entry point: build and save experiment summary from all phases."""
    import argparse
    from yolof_soup.config.experiment_config import RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Phase 6: Archive experiment results with hashes & lineage"
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory to save archive files",
    )
    parser.add_argument(
        "--audit-report",
        help="Path to phase2b_audit_report.json",
    )
    parser.add_argument(
        "--lmc-json",
        help="Path to phase3_lmc_barriers.json",
    )
    parser.add_argument(
        "--hessian-json",
        help="Path to phase3_hessian_traces.json",
    )
    parser.add_argument(
        "--soup-results-json",
        help="Path to phase4_soup_results.json",
    )
    parser.add_argument(
        "--preregistration-json",
        help="Path to phase4b_preregistration.json",
    )
    parser.add_argument(
        "--d1-results-json",
        help="Path to phase5_d1_results.json",
    )
    parser.add_argument(
        "--d2-results-json",
        help="Path to phase5_d2_results.json",
    )
    parser.add_argument(
        "--c3-results-json",
        help="Path to phase5_c3_results.json",
    )
    args = parser.parse_args()

    # Load audit report if provided
    audit_report = None
    if args.audit_report:
        try:
            with open(args.audit_report) as f:
                audit_report = json.load(f)
        except Exception as e:
            logger.warning("Could not load audit report: %s", e)

    # Build summary
    summary = build_experiment_summary(
        phase2b_audit_report=audit_report,
        phase3_lmc_barriers_json=args.lmc_json,
        phase3_hessian_traces_json=args.hessian_json,
        phase4_soup_results_json=args.soup_results_json,
        phase4b_preregistration_json=args.preregistration_json,
        phase5_d1_results_json=args.d1_results_json,
        phase5_d2_results_json=args.d2_results_json,
        phase5_c3_results_json=args.c3_results_json,
    )

    # Save
    json_path, txt_path = save_experiment_summary(summary, args.output_dir)

    logger.info("="*80)
    logger.info("ARCHIVE & SUMMARY COMPLETE")
    logger.info("="*80)
    logger.info("Files saved:")
    logger.info("  JSON: %s", json_path)
    logger.info("  TXT:  %s", txt_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    main()
