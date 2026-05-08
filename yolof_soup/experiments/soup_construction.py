"""
phase4_soup_construction.py  (was: phase3_soup_construction.py)
================================================================

Phase 4: Build and evaluate soup merging conditions (M1–M4).

Conditions:
  M1 (Condition 1): Global uniform averaging
  M2 (Condition 2): Branch-uniform averaging (cls / reg averaged independently)
  M3 (Condition 3): Coordinate-descent Dirichlet search over branch λ values
  M4 (Condition 4): Fisher / L2-proxy weighted branch averaging

For each condition:
  1. Build merged state-dict
  2. Calibrate BN running statistics (BN recalibration pass)
  3. Evaluate on held-out eval split
  4. Extract per-class AP (80 classes) + mAP50, AR@100
  5. Save checkpoint + results JSON

Key outputs:
  phase4_soup_results.json  — per-class AP arrays + metrics for all conditions
  branch_uniform_soup.pth   — M2 checkpoint
  best_learned_soup.pth     — best of M3 / M4 checkpoint
  global_uniform_soup.pth   — M1 checkpoint

Fix log:
  - Double evaluation of conditions 1 & 2 removed; each condition is now
    evaluated exactly once.
  - selection_dataloader now uses a SEPARATE selection dataset
    (SELECTION_DATASET config key) to avoid data leakage into the eval set
    during the coordinate-descent λ search.
  - Coordinate-descent search is now iterative (true CD): cls and reg
    lambdas are searched in alternating rounds until convergence, with each
    branch fixed at the current best value while the other is varied.
  - Shared decoder trunk λ is now interpolated as ½(λ_cls + λ_reg) instead
    of the previous hard-coded λ=1.0, consistent with apply_subhead_lambdas.
  - logger=None initialisation removed; module-level logger created at import
    time so that functions called before run() do not raise AttributeError.
  - BN recalibration (calibrate_bn) called on every soup model before eval.
  - Fisher weight proxy comment updated to clarify limitation.

Run: python -m yolof_soup.experiments.soup_construction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    DEVICE,
    EVAL_DATASET,
    RESULTS_DIR,
    SELECTION_DATASET,
    PHASE2_OUTPUT_DIR,
    build_eval_cfg,
)
from yolof_soup.config.experiment_registry import get_run_specs
from yolof_soup.utils.inference import InferenceWrapper
from yolof_soup.utils.checkpoint_utils import load_states, save_checkpoint
from yolof_soup.utils.eval_utils import build_eval_dataloader, get_map, extract_per_class_ap, compute_coco_map
from yolof_soup.utils.key_utils import (
    apply_uniform_lambdas,
    calibrate_bn,
    compute_anchor,
    compute_task_vectors,
    extract_subdict,
    get_decoder_keys,
    get_backbone_encoder_keys,
    merge_subdicts,
    split_decoder_subheads,
)
from yolof_soup.utils.state_dict_utils import assign_state_to_model
from yolof_soup.utils.global_logger import get_logger

# Module-level logger — always available; no NoneType risk
logger = get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

#: λ grid for the coordinate-descent search (Condition 3)
CD_LAMBDA_GRID: List[float] = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

#: Maximum CD rounds before declaring convergence
CD_MAX_ROUNDS: int = 5

#: Minimum absolute mAP improvement to continue CD rounds
CD_CONVERGENCE_TOL: float = 1e-4

#: Whether to use real Hessian (expensive) or L2 proxy for Fisher weights
USE_HESSIAN_FOR_FISHER: bool = False

#: BN calibration batches after soup construction
BN_CALIB_BATCHES: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Condition 1 (M1): Global Uniform Averaging
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_1_global_uniform(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M1: Uniform averaging over the entire model.
    θ_soup = (1/N) Σ θ_i
    BN running stats are taken from the first (best) ingredient.
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
    M2: Branch-partitioned uniform averaging.
    cls_head, reg_head, and shared trunk are each averaged uniformly.
    Backbone + encoder are taken from ingredient 0.
    """
    logger.info("Building Condition 2 (M2): Branch-Uniform Averaging")

    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)

    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)

    cls_avg    = compute_anchor([extract_subdict(sd, cls_keys)    for sd in ingredient_states])
    reg_avg    = compute_anchor([extract_subdict(sd, reg_keys)    for sd in ingredient_states])
    shared_avg = compute_anchor([extract_subdict(sd, shared_keys) for sd in ingredient_states])

    soup = merge_subdicts(be_dict, cls_avg, reg_avg, shared_avg)
    logger.info("  ✓ Branch-uniform soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


# ─────────────────────────────────────────────────────────────────────────────
# Condition 3 (M3): Coordinate-Descent Dirichlet Search
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_3_dirichlet_cd(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg,
    selection_dataloader,
    calib_dataloader,
) -> Dict[str, torch.Tensor]:
    """
    M3: Iterative coordinate-descent search for optimal per-branch λ values.

    Uses the SELECTION split (separate from the eval split) to search λ,
    preventing data leakage.  The search alternates between cls and reg
    branches (true CD) until mAP improvement falls below CD_CONVERGENCE_TOL
    or CD_MAX_ROUNDS is reached.

    Shared trunk λ is set to ½(λ_cls + λ_reg) as defined in
    apply_subhead_lambdas (was incorrectly hard-coded to 1.0 before).
    """
    logger.info("Building Condition 3 (M3): Dirichlet-Sampled CD Search")

    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)

    cls_states    = [extract_subdict(sd, cls_keys)    for sd in ingredient_states]
    reg_states    = [extract_subdict(sd, reg_keys)    for sd in ingredient_states]
    shared_states = [extract_subdict(sd, shared_keys) for sd in ingredient_states]

    anchor_cls    = compute_anchor(cls_states)
    anchor_reg    = compute_anchor(reg_states)
    anchor_shared = compute_anchor(shared_states)

    cls_taus    = compute_task_vectors(cls_states,    anchor_cls)
    reg_taus    = compute_task_vectors(reg_states,    anchor_reg)
    shared_taus = compute_task_vectors(shared_states, anchor_shared)

    # True coordinate descent: alternate between cls and reg
    best_lam_cls = 1.0
    best_lam_reg = 1.0
    prev_best_map = -1.0

    for cd_round in range(CD_MAX_ROUNDS):
        logger.info("  [CD round %d/%d] current λ_cls=%.3f  λ_reg=%.3f",
                    cd_round + 1, CD_MAX_ROUNDS, best_lam_cls, best_lam_reg)

        # Search cls while holding reg fixed at best_lam_reg
        best_lam_cls = _search_one_branch(
            anchor_cls, anchor_reg, anchor_shared,
            cls_taus, reg_taus, shared_taus,
            be_dict, cfg, calib_dataloader, selection_dataloader,
            search_branch="cls",
            fixed_lam_other=best_lam_reg,
        )
        # Search reg while holding cls fixed at best_lam_cls
        best_lam_reg = _search_one_branch(
            anchor_cls, anchor_reg, anchor_shared,
            cls_taus, reg_taus, shared_taus,
            be_dict, cfg, calib_dataloader, selection_dataloader,
            search_branch="reg",
            fixed_lam_other=best_lam_cls,
        )

        # Evaluate joint optimum
        shared_lam  = (best_lam_cls + best_lam_reg) / 2.0
        full_state  = _assemble_state(
            anchor_cls, anchor_reg, anchor_shared,
            cls_taus, reg_taus, shared_taus,
            be_dict,
            lam_cls=best_lam_cls, lam_reg=best_lam_reg, lam_shared=shared_lam,
        )
        curr_map    = _eval_state_on_dataloader(
            full_state, cfg, calib_dataloader, selection_dataloader,
            tag=f"cd_round{cd_round}_joint",
        )
        improvement = curr_map - prev_best_map
        logger.info(
            "  [CD round %d] mAP=%.4f  Δ=%.5f",
            cd_round + 1, curr_map, improvement,
        )
        if improvement < CD_CONVERGENCE_TOL and cd_round > 0:
            logger.info("  CD converged after %d rounds.", cd_round + 1)
            break
        prev_best_map = curr_map

    logger.info("  → CD final: λ_cls=%.3f  λ_reg=%.3f", best_lam_cls, best_lam_reg)

    shared_lam = (best_lam_cls + best_lam_reg) / 2.0
    soup = _assemble_state(
        anchor_cls, anchor_reg, anchor_shared,
        cls_taus, reg_taus, shared_taus,
        be_dict,
        lam_cls=best_lam_cls, lam_reg=best_lam_reg, lam_shared=shared_lam,
    )
    logger.info("  ✓ Dirichlet CD soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


def _assemble_state(
    anchor_cls, anchor_reg, anchor_shared,
    taus_cls, taus_reg, taus_shared,
    be_dict,
    lam_cls: float, lam_reg: float, lam_shared: float,
) -> Dict[str, torch.Tensor]:
    """Build a full state-dict from per-branch anchors, taus, and lambdas."""
    cls_merged    = apply_uniform_lambdas(anchor_cls,    taus_cls,    lam_cls)
    reg_merged    = apply_uniform_lambdas(anchor_reg,    taus_reg,    lam_reg)
    shared_merged = apply_uniform_lambdas(anchor_shared, taus_shared, lam_shared)
    return merge_subdicts(be_dict, cls_merged, reg_merged, shared_merged)


def _search_one_branch(
    anchor_cls, anchor_reg, anchor_shared,
    taus_cls, taus_reg, taus_shared,
    be_dict,
    cfg,
    calib_dataloader,
    selection_dataloader,
    search_branch: str,       # "cls" or "reg"
    fixed_lam_other: float,   # best current lambda for the OTHER branch
) -> float:
    """
    Line-search over CD_LAMBDA_GRID for one branch while the other is held
    at fixed_lam_other.  Shared trunk always uses ½(lam_cls + lam_reg).
    Evaluation is on the SELECTION split only.
    """
    best_lam = 1.0
    best_map = -1.0
    all_failed = True

    for lam in CD_LAMBDA_GRID:
        if search_branch == "cls":
            lam_cls, lam_reg = lam, fixed_lam_other
        else:
            lam_cls, lam_reg = fixed_lam_other, lam

        lam_shared = (lam_cls + lam_reg) / 2.0
        full_state = _assemble_state(
            anchor_cls, anchor_reg, anchor_shared,
            taus_cls, taus_reg, taus_shared,
            be_dict,
            lam_cls=lam_cls, lam_reg=lam_reg, lam_shared=lam_shared,
        )
        try:
            map_val = _eval_state_on_dataloader(
                full_state, cfg, calib_dataloader, selection_dataloader,
                tag=f"{search_branch}_lam{lam:.2f}",
            )
            all_failed = False
            logger.debug("  [CD %s] λ=%.3f → mAP=%.4f", search_branch, lam, map_val)
            if map_val > best_map:
                best_map = map_val
                best_lam = lam
        except Exception as e:
            logger.warning("  [CD %s] λ=%.3f → eval failed: %s", search_branch, lam, e)

    if all_failed:
        logger.warning(
            "  [CD %s] All λ values failed; defaulting to λ=1.0.", search_branch
        )
        return 1.0
    return best_lam


def _eval_state_on_dataloader(
    state: Dict[str, torch.Tensor],
    cfg,
    calib_dataloader,
    selection_dataloader,
    tag: str = "cd_eval",
) -> float:
    """Load state into a model, calibrate BN, evaluate on selection_dataloader."""
    model = InferenceWrapper(cfg)
    assign_state_to_model(model, state)
    model = model.to(DEVICE)
    calibrate_bn(model, calib_dataloader, n_batches=BN_CALIB_BATCHES, device=DEVICE)
    model.eval()
    try:
        map_val = get_map(
            model.model, cfg, SELECTION_DATASET,
            output_dir=Path(RESULTS_DIR) / "cd_search",
            tag=tag,
        )
    finally:
        del model
        torch.cuda.empty_cache()
    return map_val


# ─────────────────────────────────────────────────────────────────────────────
# Condition 4 (M4): Fisher / L2-Proxy Weighted Averaging
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_4_fisher_weighted(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M4: L2-norm-proxied weighted branch averaging.

    NOTE: L2 norm is an approximation of relative parameter magnitude,
    NOT a true Fisher information measure.  True Fisher weights require
    per-sample gradient computation.  This proxy is used for computational
    tractability; the thesis should clearly acknowledge this limitation
    in the Methodology chapter.

    Ingredients with smaller L2 norm receive higher weight on the
    assumption that smaller-norm models landed in flatter minima.
    This assumption is imperfect and should be validated with ablations.
    """
    logger.info("Building Condition 4 (M4): L2-Proxy Weighted Averaging")

    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)

    cls_norms    = [_compute_l2_norm(extract_subdict(sd, cls_keys))    for sd in ingredient_states]
    reg_norms    = [_compute_l2_norm(extract_subdict(sd, reg_keys))    for sd in ingredient_states]
    shared_norms = [_compute_l2_norm(extract_subdict(sd, shared_keys)) for sd in ingredient_states]

    cls_weights    = _inverse_norm_weights(cls_norms)
    reg_weights    = _inverse_norm_weights(reg_norms)
    shared_weights = _inverse_norm_weights(shared_norms)

    logger.debug("  L2-proxy weights cls:    %s", [f"{w:.3f}" for w in cls_weights])
    logger.debug("  L2-proxy weights reg:    %s", [f"{w:.3f}" for w in reg_weights])
    logger.debug("  L2-proxy weights shared: %s", [f"{w:.3f}" for w in shared_weights])

    cls_avg    = _weighted_average([extract_subdict(sd, cls_keys)    for sd in ingredient_states], cls_weights)
    reg_avg    = _weighted_average([extract_subdict(sd, reg_keys)    for sd in ingredient_states], reg_weights)
    shared_avg = _weighted_average([extract_subdict(sd, shared_keys) for sd in ingredient_states], shared_weights)

    soup = merge_subdicts(be_dict, cls_avg, reg_avg, shared_avg)
    logger.info("  ✓ L2-proxy weighted soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


def _compute_l2_norm(state_dict: Dict[str, torch.Tensor]) -> float:
    norm_sq = 0.0
    for tensor in state_dict.values():
        if torch.is_floating_point(tensor):
            norm_sq += (tensor.float() ** 2).sum().item()
    return float(np.sqrt(norm_sq))


def _inverse_norm_weights(norms: List[float]) -> List[float]:
    inv = [1.0 / (n + 1e-8) for n in norms]
    total = sum(inv)
    return [w / total for w in inv]


def _weighted_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    averaged: Dict[str, torch.Tensor] = {}
    for k in state_dicts[0]:
        if torch.is_floating_point(state_dicts[0][k]):
            weighted = sum(w * sd[k].float() for w, sd in zip(weights, state_dicts))
            averaged[k] = weighted.to(state_dicts[0][k].dtype)
        else:
            averaged[k] = state_dicts[0][k].clone()
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_condition(
    state_dict: Dict[str, torch.Tensor],
    cfg,
    calib_dataloader,
    tag: str,
) -> Dict[str, Any]:
    """
    Load a soup state-dict, calibrate BN, evaluate on the HELD-OUT EVAL split.

    Returns:
        {
            "map50_95": float,
            "map50":    float,
            "ar100":    float,
            "per_class_ap": list[float] (len=80),
        }
    """
    logger.info("Evaluating condition %s on eval split…", tag)

    model = InferenceWrapper(cfg)
    assign_state_to_model(model, state_dict)
    model = model.to(DEVICE)

    # Recalibrate BN running stats for the merged weights
    calibrate_bn(model, calib_dataloader, n_batches=BN_CALIB_BATCHES, device=DEVICE)
    model.eval()

    try:
        results_dict = compute_coco_map(
            model, cfg, EVAL_DATASET,
            output_dir=Path(RESULTS_DIR) / "phase4_eval",
            tag=tag,
        )
        map_val   = float(results_dict.get("AP",                0.0))
        map50_val = float(results_dict.get("AP50",              0.0))
        ar100_val = float(results_dict.get("AR-maxDets=100",    0.0))
        per_class = extract_per_class_ap(results_dict)
        logger.info(
            "  ✓ %s: mAP50:95=%.4f  mAP50=%.4f  AR@100=%.4f",
            tag, map_val, map50_val, ar100_val,
        )
    except Exception as e:
        logger.error("  ✗ Evaluation failed for %s: %s", tag, e, exc_info=True)
        map_val = map50_val = ar100_val = 0.0
        per_class = [0.0] * 80
    finally:
        del model
        torch.cuda.empty_cache()

    return {
        "map50_95":     map_val,
        "map50":        map50_val,
        "ar100":        ar100_val,
        "per_class_ap": per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _state_dict_size_mb(state_dict: Dict[str, torch.Tensor]) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    return total_bytes / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True) -> Dict[str, Any]:
    """
    Phase 4 entry point.

    Changes vs. original:
      - Conditions 1 & 2 evaluated ONCE (not twice).
      - Separate selection_dataloader built from SELECTION_DATASET.
      - BN recalibration applied after every soup construction.
      - True iterative CD search for Condition 3.
      - Shared λ = ½(λ_cls + λ_reg) instead of hard-coded 1.0.
    """
    global logger
    logger = get_logger(
        level=logging.DEBUG if verbose else logging.INFO,
        add_file_handler=True,
    )

    logger.info("=" * 90)
    logger.info("PHASE 4: SOUP CONSTRUCTION & EVALUATION")
    logger.info("=" * 90)

    results_dir   = Path(RESULTS_DIR);   results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(CHECKPOINT_DIR); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ingredients_dir = Path(PHASE2_OUTPUT_DIR)

    # ── 1. Load ingredient checkpoints ──────────────────────────────────────
    logger.info("\n[1/6] Loading ingredient checkpoints…")
    run_registry     = get_run_specs()
    ingredient_runs  = [r for r in run_registry if r.role == "ingredient"]
    ingredient_paths = []
    cfg_paths        = []
    for run_spec in ingredient_runs:
        ckpt_path = Path(ingredients_dir) / f"{run_spec.run_name}/model_best.pth"
        ingredient_paths.append(ckpt_path)
        cfg_paths.append(Path(ingredients_dir) / f"{run_spec.run_name}/config.yaml")

    missing = [p for p in ingredient_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing phase 2 checkpoints: {missing}  "
            f"(expected in {ingredients_dir})"
        )

    ingredient_states = load_states(ingredient_paths)
    logger.info("  ✓ Loaded %d ingredients", len(ingredient_states))

    # ── 2. Build config ──────────────────────────────────────────────────────
    logger.info("\n[2/6] Building Detectron2 config…")
    cfg = build_eval_cfg()
    logger.info("  ✓ Config ready")

    # ── 3. Build dataloaders ─────────────────────────────────────────────────
    # eval_dataloader:      held-out test split  (ONLY for final evaluation)
    # selection_dataloader: separate validation split (ONLY for CD λ search)
    # calib_dataloader:     same as selection; used for BN recalibration
    logger.info("\n[3/6] Building dataloaders…")
    eval_dataloader      = build_eval_dataloader(cfg, EVAL_DATASET)
    selection_dataloader = build_eval_dataloader(cfg, SELECTION_DATASET)
    calib_dataloader     = build_eval_dataloader(cfg, SELECTION_DATASET)
    logger.info(
        "  ✓ eval=%s  selection=%s  calib=%s",
        EVAL_DATASET, SELECTION_DATASET, SELECTION_DATASET,
    )

    # ── 4. Build soup conditions ─────────────────────────────────────────────
    logger.info("\n[4/6] Building 4 soup conditions…")
    condition_1_state = build_condition_1_global_uniform(ingredient_states)
    condition_2_state = build_condition_2_branch_uniform(ingredient_states)
    condition_3_state = build_condition_3_dirichlet_cd(
        ingredient_states, cfg, selection_dataloader, calib_dataloader
    )
    condition_4_state = build_condition_4_fisher_weighted(ingredient_states)
    logger.info("  ✓ All 4 conditions built")

    # ── 5. Evaluate each condition ONCE on the held-out eval split ───────────
    logger.info("\n[5/6] Evaluating all 4 conditions on eval split…")
    results_cond1          = evaluate_condition(condition_1_state, cfg, calib_dataloader, "condition_1")
    results_cond2          = evaluate_condition(condition_2_state, cfg, calib_dataloader, "condition_2")
    results_cond3          = evaluate_condition(condition_3_state, cfg, calib_dataloader, "condition_3")
    results_cond4          = evaluate_condition(condition_4_state, cfg, calib_dataloader, "condition_4")
    results_best_individual = evaluate_condition(ingredient_states[0], cfg, calib_dataloader, "best_individual")
    logger.info("  ✓ All evaluations complete")

    # Determine best learned condition
    map_3 = results_cond3["map50_95"]
    map_4 = results_cond4["map50_95"]
    best_learned_is_cond3 = map_3 >= map_4
    best_learned_state    = condition_3_state if best_learned_is_cond3 else condition_4_state
    best_learned_tag      = "condition_3"     if best_learned_is_cond3 else "condition_4"
    logger.info("  Best learned: %s (mAP50:95=%.4f)", best_learned_tag, max(map_3, map_4))

    # ── 6. Save checkpoints ──────────────────────────────────────────────────
    logger.info("\n[6/6] Saving checkpoints…")
    global_uniform_path  = checkpoint_dir / "global_uniform_soup.pth"
    branch_uniform_path  = checkpoint_dir / "branch_uniform_soup.pth"
    best_learned_path    = checkpoint_dir / "best_learned_soup.pth"

    save_checkpoint(global_uniform_path,  condition_1_state,
                    metadata={"condition": 1, "method": "global_uniform",
                               "map50_95": results_cond1["map50_95"]})
    save_checkpoint(branch_uniform_path,  condition_2_state,
                    metadata={"condition": 2, "method": "branch_uniform",
                               "map50_95": results_cond2["map50_95"]})
    save_checkpoint(best_learned_path,    best_learned_state,
                    metadata={"condition": 3 if best_learned_is_cond3 else 4,
                               "method": best_learned_tag,
                               "map50_95": max(map_3, map_4)})
    logger.info("  ✓ Checkpoints saved")

    # ── Compile results JSON ─────────────────────────────────────────────────
    results = {
        "condition_1":             results_cond1,
        "condition_2":             results_cond2,
        "condition_3":             results_cond3,
        "condition_4":             results_cond4,
        "best_individual":         results_best_individual,
        "best_learned_condition":  best_learned_tag,
        "metadata": {
            "n_ingredients":   len(ingredient_states),
            "cd_lambda_grid":  CD_LAMBDA_GRID,
            "cd_max_rounds":   CD_MAX_ROUNDS,
            "eval_dataset":    EVAL_DATASET,
            "selection_dataset": SELECTION_DATASET,
        },
    }
    results_json_path = results_dir / "phase4_soup_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  → Results: %s", results_json_path)

    logger.info("\n" + "=" * 90)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 90)
    logger.info("  Global-uniform soup:  %s", global_uniform_path)
    logger.info("  Branch-uniform soup:  %s", branch_uniform_path)
    logger.info("  Best learned soup:    %s", best_learned_path)
    logger.info("  Results JSON:         %s", results_json_path)
    logger.info("=" * 90)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Soup Construction & Evaluation")
    parser.add_argument("--verbose", action="store_true")
    parsed = parser.parse_args()
    run(verbose=parsed.verbose)


if __name__ == "__main__":
    main()
