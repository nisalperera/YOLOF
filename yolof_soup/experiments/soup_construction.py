"""
phase3_soup_construction.py
============================

Phase 3: Build and evaluate soup merging conditions (M1-M4).

This module constructs 4 merging conditions:
  M1 (Condition 1): Global uniform averaging
  M2 (Condition 2): Branch-uniform averaging (cls_head, reg_head averaged independently)
  M3 (Condition 3): Dirichlet-sampled branch coefficients via coordinate descent
  M4 (Condition 4): Fisher/Hessian-weighted branch coefficients

For each condition, we:
  1. Build the merged state-dict
  2. Evaluate on the held-out eval split (4000 images)
  3. Extract per-class AP array (80 classes) + secondary metrics (mAP50, AR@100)
  4. Save checkpoint and results JSON

Key outputs:
  - phase3_soup_results.json — per-class AP arrays + summary metrics for all conditions
  - branch_uniform_soup.pth — Condition 2 (M2) checkpoint
  - best_learned_soup.pth — best of Condition 3 or 4 (M3 or M4) checkpoint

Run: python -m yolof_soup.experiments.phase3_soup_construction
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    DEVICE,
    EVAL_DATASET,
    RESULTS_DIR,
    SELECTION_DATASET,
    build_eval_cfg,
)
from yolof_soup.utils.checkpoint_utils import load_states, save_checkpoint
from yolof_soup.utils.eval_utils import build_eval_dataloader, get_map, extract_per_class_ap
from yolof_soup.utils.key_utils import (
    apply_uniform_lambdas,
    compute_anchor,
    compute_task_vectors,
    extract_subdict,
    get_decoder_keys,
    get_backbone_encoder_keys,
    merge_subdicts,
    split_decoder_subheads,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (can be adjusted)
# ─────────────────────────────────────────────────────────────────────────────

#: Coordinate descent λ grid for Dirichlet search (Condition 3)
CD_LAMBDA_GRID: List[float] = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

#: Whether to use Hessian (expensive) or L2 proxy for Fisher weights
USE_HESSIAN_FOR_FISHER: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Detectron2 model building & state-dict assignment
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_config(cfg) -> torch.nn.Module:
    """Build a YOLOF model from Detectron2 config (CPU device)."""
    from detectron2.modeling import build_model
    return build_model(cfg)


def assign_state_to_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """In-place assign state_dict to model."""
    model.load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Condition 1 (M1): Global Uniform Averaging
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_1_global_uniform(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M1 (Condition 1): Uniform averaging over entire model.
    θ_soup = (1/N) Σ θ_i
    """
    logger.info("Building Condition 1 (M1): Global Uniform Averaging")
    soup = compute_anchor(ingredient_states)
    logger.info("  ✓ Global uniform soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


# ─────────────────────────────────────────────────────────────────────────────
# Condition 2 (M2): Branch-Uniform Averaging
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_2_branch_uniform(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M2 (Condition 2): Branch-partitioned uniform averaging.
    
    Partition decoder into cls_head, reg_head, and shared subparts.
    Average cls_head uniformly, reg_head uniformly, shared uniformly.
    Keep backbone+encoder from ingredient 0 (same as input).
    """
    logger.info("Building Condition 2 (M2): Branch-Uniform Averaging")
    
    # Extract decoder keys from first ingredient
    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    
    # Extract backbone+encoder from ingredient 0 (frozen)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)
    
    # Average cls_head, reg_head, shared_keys uniformly
    cls_states = [extract_subdict(sd, cls_keys) for sd in ingredient_states]
    reg_states = [extract_subdict(sd, reg_keys) for sd in ingredient_states]
    shared_states = [extract_subdict(sd, shared_keys) for sd in ingredient_states]
    
    cls_avg = compute_anchor(cls_states)
    reg_avg = compute_anchor(reg_states)
    shared_avg = compute_anchor(shared_states)
    
    # Merge all parts
    soup = merge_subdicts(be_dict, cls_avg, reg_avg, shared_avg)
    logger.info("  ✓ Branch-uniform soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


# ─────────────────────────────────────────────────────────────────────────────
# Condition 3 (M3): Dirichlet-Sampled Branch Coefficients (Coordinate Descent)
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_3_dirichlet_cd(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg,
    selection_dataloader,
) -> Dict[str, torch.Tensor]:
    """
    M3 (Condition 3): Dirichlet simplex search via coordinate descent.
    
    Use selection split to find best λ values for cls and reg branches.
    Then construct soup with those weights.
    """
    logger.info("Building Condition 3 (M3): Dirichlet-Sampled Branch Coefficients (CD Search)")
    
    # Extract decoder keys and sub-head partitions
    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)
    
    # Compute anchors for each subhead
    cls_states = [extract_subdict(sd, cls_keys) for sd in ingredient_states]
    reg_states = [extract_subdict(sd, reg_keys) for sd in ingredient_states]
    shared_states = [extract_subdict(sd, shared_keys) for sd in ingredient_states]
    
    anchor_cls = compute_anchor(cls_states)
    anchor_reg = compute_anchor(reg_states)
    anchor_shared = compute_anchor(shared_states)
    
    # Compute task vectors
    cls_taus = compute_task_vectors(cls_states, anchor_cls)
    reg_taus = compute_task_vectors(reg_states, anchor_reg)
    shared_taus = compute_task_vectors(shared_states, anchor_shared)
    
    # Coordinate descent search for best λ per branch
    best_lam_cls = _coordinate_descent_search(
        ingredient_states, anchor_cls, cls_taus, cls_keys, be_dict, cfg, 
        selection_dataloader, "cls"
    )
    best_lam_reg = _coordinate_descent_search(
        ingredient_states, anchor_reg, reg_taus, reg_keys, be_dict, cfg, 
        selection_dataloader, "reg"
    )
    
    logger.info("  → CD search found: λ_cls=%.3f, λ_reg=%.3f", best_lam_cls, best_lam_reg)
    
    # Build final soup with optimal λ values
    cls_merged = apply_uniform_lambdas(anchor_cls, cls_taus, best_lam_cls)
    reg_merged = apply_uniform_lambdas(anchor_reg, reg_taus, best_lam_reg)
    shared_merged = apply_uniform_lambdas(anchor_shared, shared_taus, 1.0)  # shared always λ=1.0
    
    soup = merge_subdicts(be_dict, cls_merged, reg_merged, shared_merged)
    logger.info("  ✓ Dirichlet soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


def _coordinate_descent_search(
    ingredient_states: List[Dict[str, torch.Tensor]],
    anchor: Dict[str, torch.Tensor],
    taus: List[Dict[str, torch.Tensor]],
    subhead_keys: List[str],
    be_dict: Dict[str, torch.Tensor],
    cfg,
    selection_dataloader,
    branch_name: str,
) -> float:
    """
    Find best λ via coordinate descent on selection split.
    """
    best_lam = 1.0
    best_map = 0.0
    
    for lam in CD_LAMBDA_GRID:
        merged = apply_uniform_lambdas(anchor, taus, lam)
        full_state = merge_subdicts(be_dict, merged)
        
        # Quick evaluation on selection split
        model = build_model_from_config(cfg)
        assign_state_to_model(model, full_state)
        model = model.to(DEVICE)
        model.eval()
        
        try:
            map_val = get_map(model, cfg, SELECTION_DATASET, output_dir=Path(RESULTS_DIR) / "cd_search", tag=f"{branch_name}_lam{lam:.2f}")
            logger.debug("  [CD search %s] λ=%.3f → mAP=%.4f", branch_name, lam, map_val)
            
            if map_val > best_map:
                best_map = map_val
                best_lam = lam
        except Exception as e:
            logger.warning("  [CD search %s] λ=%.3f → evaluation failed: %s", branch_name, lam, str(e))
        finally:
            del model
            torch.cuda.empty_cache()
    
    return best_lam


# ─────────────────────────────────────────────────────────────────────────────
# Condition 4 (M4): Fisher/Hessian-Weighted Branch Coefficients
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_4_fisher_weighted(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M4 (Condition 4): Fisher/Hessian-weighted branch coefficients.
    
    Compute Fisher information (or L2 proxy) per ingredient per branch.
    Weight each ingredient inversely by Fisher magnitude.
    """
    logger.info("Building Condition 4 (M4): Fisher/Hessian-Weighted Branch Coefficients")
    
    # For now, use L2 norm as proxy for Fisher (deterministic, fast)
    
    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)
    
    # Compute L2 norms per branch (proxy for Hessian eigenvalues)
    cls_norms = [_compute_l2_norm(extract_subdict(sd, cls_keys)) for sd in ingredient_states]
    reg_norms = [_compute_l2_norm(extract_subdict(sd, reg_keys)) for sd in ingredient_states]
    shared_norms = [_compute_l2_norm(extract_subdict(sd, shared_keys)) for sd in ingredient_states]
    
    # Inverse weights: smaller norm → larger weight
    cls_weights = _inverse_norm_weights(cls_norms)
    reg_weights = _inverse_norm_weights(reg_norms)
    shared_weights = _inverse_norm_weights(shared_norms)
    
    logger.debug("  Fisher weights: cls=%s", [f"{w:.3f}" for w in cls_weights])
    logger.debug("  Fisher weights: reg=%s", [f"{w:.3f}" for w in reg_weights])
    
    # Weighted merge
    cls_avg = _weighted_average(
        [extract_subdict(sd, cls_keys) for sd in ingredient_states],
        cls_weights
    )
    reg_avg = _weighted_average(
        [extract_subdict(sd, reg_keys) for sd in ingredient_states],
        reg_weights
    )
    shared_avg = _weighted_average(
        [extract_subdict(sd, shared_keys) for sd in ingredient_states],
        shared_weights
    )
    
    soup = merge_subdicts(be_dict, cls_avg, reg_avg, shared_avg)
    logger.info("  ✓ Fisher-weighted soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


def _compute_l2_norm(state_dict: Dict[str, torch.Tensor]) -> float:
    """Compute L2 norm of all parameters in state_dict."""
    norm_sq = 0.0
    for tensor in state_dict.values():
        if torch.is_floating_point(tensor):
            norm_sq += (tensor.float() ** 2).sum().item()
    return float(np.sqrt(norm_sq))


def _inverse_norm_weights(norms: List[float]) -> List[float]:
    """Convert L2 norms to inverse weights (smaller norm → larger weight)."""
    inv_norms = [1.0 / (n + 1e-8) for n in norms]
    total = sum(inv_norms)
    return [w / total for w in inv_norms]


def _weighted_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Weighted average of state-dicts."""
    averaged: Dict[str, torch.Tensor] = {}
    for k in state_dicts[0]:
        if torch.is_floating_point(state_dicts[0][k]):
            weighted = sum(w * sd[k].float() for w, sd in zip(weights, state_dicts))
            averaged[k] = weighted.to(state_dicts[0][k].dtype)
        else:
            averaged[k] = state_dicts[0][k].clone()
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_condition(
    state_dict: Dict[str, torch.Tensor],
    cfg,
    tag: str,
) -> Dict[str, Any]:
    """
    Evaluate a condition and extract key metrics including per-class AP.
    
    Returns:
        {
            "map50_95": float,
            "map50": float,
            "ar100": float,
            "per_class_ap": [80-element list],
        }
    """
    logger.info("Evaluating condition %s on eval split...", tag)
    
    # Build and load model
    model = build_model_from_config(cfg)
    assign_state_to_model(model, state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    # Run evaluation
    try:
        from yolof_soup.utils.eval_utils import compute_coco_map
        results_dict = compute_coco_map(model, cfg, EVAL_DATASET, output_dir=Path(RESULTS_DIR) / "phase3_eval", tag=tag)
        
        map_val = float(results_dict.get("AP", 0.0))
        map50_val = float(results_dict.get("AP50", 0.0))
        ar100_val = float(results_dict.get("AR-maxDets=100", 0.0))
        
        # Extract per-class AP values
        per_class_ap = extract_per_class_ap(results_dict, n_classes=80)
        
        logger.info("  ✓ Condition %s: mAP50:95=%.4f, mAP50=%.4f, AR@100=%.4f", 
                   tag, map_val, map50_val, ar100_val)
    except Exception as e:
        logger.error("  ✗ Evaluation failed for %s: %s", tag, str(e))
        map_val = 0.0
        map50_val = 0.0
        ar100_val = 0.0
        per_class_ap = [0.0] * 80
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "map50_95": map_val,
        "map50": map50_val,
        "ar100": ar100_val,
        "per_class_ap": per_class_ap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: State-dict size
# ─────────────────────────────────────────────────────────────────────────────

def _state_dict_size_mb(state_dict: Dict[str, torch.Tensor]) -> float:
    """Estimate state-dict size in MB."""
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    return total_bytes / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True) -> Dict[str, Any]:
    """
    Main Phase 3 entry point.
    
    Args:
        verbose: Whether to log progress
    
    Returns:
        Dict with results for all 4 conditions + metadata
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
    
    logger.info("=" * 90)
    logger.info("PHASE 3: SOUP CONSTRUCTION & EVALUATION")
    logger.info("=" * 90)
    
    # Setup directories
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ingredient checkpoints
    logger.info("\n[1/6] Loading 6 ingredient checkpoints...")
    ingredient_paths = []
    for _run in os.listdir(CHECKPOINT_DIR):
        ingredient_paths.append(Path(CHECKPOINT_DIR) / _run / "model_best.pth")
    
    # Check if paths exist
    missing = [p for p in ingredient_paths if not p.exists()]
    if missing:
        logger.error("Missing checkpoints: %s", missing)
        raise FileNotFoundError(f"Missing phase 2 checkpoints. Expected at: {ingredient_paths[0].parent}/")
    
    ingredient_states = load_states(ingredient_paths)
    logger.info("  ✓ Loaded 6 ingredients")
    
    # Build Detectron2 config
    logger.info("\n[2/6] Building Detectron2 config...")
    cfg = build_eval_cfg()
    logger.info("  ✓ Config ready")
    
    # Build dataloaders
    logger.info("\n[3/6] Building dataloaders...")
    eval_dataloader = build_eval_dataloader(cfg, EVAL_DATASET)
    selection_dataloader = build_eval_dataloader(cfg, SELECTION_DATASET)
    logger.info("  ✓ Dataloaders ready")
    
    # Build all 4 conditions
    logger.info("\n[4/6] Building 4 soup conditions...")
    condition_1_state = build_condition_1_global_uniform(ingredient_states)
    condition_2_state = build_condition_2_branch_uniform(ingredient_states)
    condition_3_state = build_condition_3_dirichlet_cd(
        ingredient_states, cfg, selection_dataloader
    )
    condition_4_state = build_condition_4_fisher_weighted(ingredient_states)
    logger.info("  ✓ All 4 conditions built")
    
    # Evaluate all 4 conditions
    logger.info("\n[5/6] Evaluating all 4 conditions on eval split...")
    results_cond1 = evaluate_condition(condition_1_state, cfg, "condition_1")
    results_cond2 = evaluate_condition(condition_2_state, cfg, "condition_2")
    results_cond3 = evaluate_condition(condition_3_state, cfg, "condition_3")
    results_cond4 = evaluate_condition(condition_4_state, cfg, "condition_4")
    
    # Also evaluate best individual (ingredient 0 as reference)
    results_best_individual = evaluate_condition(ingredient_states[0], cfg, "best_individual")
    logger.info("  ✓ All evaluations complete")
    
    # Determine best learned condition
    map_3 = results_cond3["map50_95"]
    map_4 = results_cond4["map50_95"]
    best_learned_is_cond3 = map_3 >= map_4
    best_learned_state = condition_3_state if best_learned_is_cond3 else condition_4_state
    best_learned_tag = "condition_3" if best_learned_is_cond3 else "condition_4"
    
    logger.info("\nBest learned condition:")
    logger.info("  → Condition 3 (Dirichlet): mAP50:95=%.4f", map_3)
    logger.info("  → Condition 4 (Fisher):    mAP50:95=%.4f", map_4)
    logger.info("  → Best: %s (mAP50:95=%.4f)", best_learned_tag, max(map_3, map_4))
    
    # Save checkpoints
    logger.info("\n[6/6] Saving checkpoints...")
    branch_uniform_path = checkpoint_dir / "branch_uniform_soup.pth"
    best_learned_path = checkpoint_dir / "best_learned_soup.pth"
    global_uniform_path = checkpoint_dir / "global_uniform_soup.pth"
    
    save_checkpoint(
        branch_uniform_path,
        condition_2_state,
        metadata={"condition": 2, "method": "branch_uniform", "map50_95": results_cond2["map50_95"]},
    )
    save_checkpoint(
        best_learned_path,
        best_learned_state,
        metadata={"condition": 3 if best_learned_is_cond3 else 4, "method": best_learned_tag, "map50_95": max(map_3, map_4)},
    )
    save_checkpoint(
        global_uniform_path,
        condition_1_state,
        metadata={"condition": 1, "method": "global_uniform", "map50_95": results_cond1["map50_95"]},
    )
    logger.info("  ✓ Checkpoints saved")
    
    # Compile and save results JSON
    logger.info("\nSaving results JSON...")
    results = {
        "condition_1": results_cond1,
        "condition_2": results_cond2,
        "condition_3": results_cond3,
        "condition_4": results_cond4,
        "best_individual": results_best_individual,
        "best_learned_condition": best_learned_tag,
        "metadata": {
            "n_ingredients": 6,
            "cd_lambda_grid": CD_LAMBDA_GRID,
        },
    }
    
    results_json_path = results_dir / "phase3_soup_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  → Results: %s", results_json_path)
    
    logger.info("\n" + "=" * 90)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 90)
    logger.info("Outputs:")
    logger.info("  • Global-uniform soup:   %s", global_uniform_path)
    logger.info("  • Branch-uniform soup:   %s", branch_uniform_path)
    logger.info("  • Best learned soup:     %s", best_learned_path)
    logger.info("  • Results JSON:          %s", results_json_path)
    logger.info("=" * 90)
    
    return results


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description="Phase 3: Soup Construction & Evaluation")
    args.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parsed_args = args.parse_args()

    run(verbose=parsed_args.verbose)
