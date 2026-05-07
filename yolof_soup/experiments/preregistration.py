"""
preregistration.py
==================

Phase 4b: Source Checkpoint Pre-registration

After Phase 4 (soup merging M1-M4), before Phase 5 (head fine-tuning D1-D2-C3):

1. Identify best learned soup (M3 vs M4 by val mAP)
2. Lock the choice in a version-controlled log
3. Document rationale (mAP metrics, component analysis)
4. Save pre-registration: phase4b_preregistration.json

This prevents post-hoc selection bias in comparing D1 (M2 init) vs D2 (best learned init).

Output:
  - phase4b_preregistration.json: locked selection + timestamp
  - phase4b_preregistration.txt: human-readable record

Run: python -m yolof_soup.experiments.preregistration
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from yolof_soup.utils.logging_utils import setup_logging

# ─────────────────────────────────────────────────────────────────────────────
# Pre-registration: Lock Best Learned Soup Before Phase 5
# ─────────────────────────────────────────────────────────────────────────────

def preregister_best_learned_soup(
    soup_results_json: str | Path,
    output_dir: str | Path = None,
) -> Dict[str, Any]:
    """
    Read Phase 4 soup results, identify best learned condition (M3 or M4),
    and pre-register the choice.

    Args:
        soup_results_json:   Path to phase4_soup_results.json from Phase 4
        output_dir:          Directory to save pre-registration (defaults to parent of JSON)

    Returns:
        Pre-registration dict with keys: timestamp, chosen_condition, chosen_checkpoint,
                                         rationale, m1_map, m2_map, m3_map, m4_map
    """
    soup_results_json = Path(soup_results_json)
    if not soup_results_json.exists():
        raise FileNotFoundError(f"Soup results JSON not found: {soup_results_json}")

    # Load Phase 4 soup results
    logging.info("Loading Phase 4 soup results: %s", soup_results_json)
    with open(soup_results_json) as f:
        soup_results = json.load(f)

    # Extract mAP scores for each condition
    m1_map = None
    m2_map = None
    m3_map = None
    m4_map = None
    m3_ckpt = None
    m4_ckpt = None

    # Soup results structure depends on implementation; adjust keys as needed
    conditions = soup_results.get("conditions", {})
    
    if "m1" in conditions:
        m1_map = conditions["m1"].get("mAP50:95", None)
    if "m2" in conditions:
        m2_map = conditions["m2"].get("mAP50:95", None)
    if "m3" in conditions:
        m3_map = conditions["m3"].get("mAP50:95", None)
        m3_ckpt = conditions["m3"].get("checkpoint", None)
    if "m4" in conditions:
        m4_map = conditions["m4"].get("mAP50:95", None)
        m4_ckpt = conditions["m4"].get("checkpoint", None)

    # Determine best learned condition (M3 vs M4)
    learned_maps = {}
    if m3_map is not None:
        learned_maps["m3"] = m3_map
    if m4_map is not None:
        learned_maps["m4"] = m4_map

    if not learned_maps:
        logging.error("No valid learned soup results found in %s", soup_results_json)
        raise ValueError("Cannot identify best learned soup from results")

    best_condition = max(learned_maps, key=learned_maps.get)
    best_map = learned_maps[best_condition]
    best_ckpt = m3_ckpt if best_condition == "m3" else m4_ckpt

    logging.info("Best learned soup identified: %s with mAP50:95=%.4f", best_condition, best_map)
    logging.info("  M3 mAP: %.4f", m3_map or 0.0)
    logging.info("  M4 mAP: %.4f", m4_map or 0.0)

    # Create pre-registration record
    preregistration = {
        "timestamp": datetime.now().isoformat(),
        "phase": "4b",
        "purpose": "Lock best learned soup before Phase 5 head fine-tuning",
        "chosen_condition": best_condition.upper(),
        "chosen_checkpoint": str(best_ckpt),
        "chosen_map50_95": best_map,
        "reference_maps": {
            "m1_global_uniform": m1_map,
            "m2_branch_uniform": m2_map,
            "m3_dirichlet_search": m3_map,
            "m4_fisher_weighted": m4_map,
        },
        "rationale": (
            f"Best-performing learned soup is {best_condition.upper()} "
            f"(mAP50:95={best_map:.4f}). This checkpoint will be used to initialize "
            f"D2 (Phase 5a) and C3 (Phase 5b) to test whether superior merge quality "
            f"yields larger head fine-tuning gains."
        ),
        "soup_results_source": str(soup_results_json),
    }

    logging.info("Pre-registration created:")
    logging.info("  Chosen:  %s", preregistration["chosen_condition"])
    logging.info("  Map:     %.4f", preregistration["chosen_map50_95"])
    logging.info("  Ckpt:    %s", preregistration["chosen_checkpoint"])

    return preregistration


def save_preregistration(
    preregistration: Dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Save pre-registration record in JSON and TXT formats.

    Returns:
        (json_path, txt_path) — paths where files were saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "phase4b_preregistration.json"
    with open(json_path, "w") as f:
        json.dump(preregistration, f, indent=2)
    logging.info("Pre-registration JSON saved → %s", json_path)

    # TXT
    txt_path = output_dir / "phase4b_preregistration.txt"
    with open(txt_path, "w") as f:
        f.write("PHASE 4B: SOURCE CHECKPOINT PRE-REGISTRATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp:      {preregistration['timestamp']}\n")
        f.write(f"Purpose:        {preregistration['purpose']}\n\n")
        f.write("CHOSEN CONDITION FOR PHASE 5:\n")
        f.write(f"  Condition:    {preregistration['chosen_condition']}\n")
        f.write(f"  Checkpoint:   {preregistration['chosen_checkpoint']}\n")
        f.write(f"  mAP50:95:     {preregistration['chosen_map50_95']:.4f}\n\n")
        f.write("REFERENCE METRICS (all learned soups):\n")
        for cond_name, map_val in preregistration["reference_maps"].items():
            if map_val is not None:
                f.write(f"  {cond_name:30s}: {map_val:8.4f}\n")
            else:
                f.write(f"  {cond_name:30s}: N/A\n")
        f.write(f"\nRATIONALE:\n{preregistration['rationale']}\n\n")
        f.write("USAGE:\n")
        f.write("  This checkpoint MUST be used to initialize:\n")
        f.write("    - D2 (head fine-tuning from best learned soup)\n")
        f.write("    - C3 (final pipeline run)\n")
        f.write("  Do NOT change this selection after this point.\n")
    logging.info("Pre-registration TXT saved → %s", txt_path)

    return json_path, txt_path


def main():
    """Entry point: load Phase 4 results, pre-register best learned soup."""
    import argparse
    from yolof_soup.config.experiment_config import RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Phase 4b: Pre-register best learned soup before Phase 5"
    )
    parser.add_argument(
        "--soup-results-json",
        required=True,
        help="Path to phase4_soup_results.json from Phase 4",
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory to save pre-registration files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO, filename="phase4b_preregistration.log", use_stdout=True)

    # Pre-register
    preregistration = preregister_best_learned_soup(args.soup_results_json)

    # Save
    json_path, txt_path = save_preregistration(preregistration, args.output_dir)

    logging.info("="*80)
    logging.info("PRE-REGISTRATION COMPLETE")
    logging.info("="*80)
    logging.info("Files saved:")
    logging.info("  JSON: %s", json_path)
    logging.info("  TXT:  %s", txt_path)
    logging.info("Next step: Use chosen checkpoint for D2 and C3 in Phase 5")

    return preregistration


if __name__ == "__main__":
    main()
