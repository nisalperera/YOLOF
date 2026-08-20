"""
phase3_soup_construction.py
============================

Phase 3: Build and evaluate soup merging conditions (M1-M5).

This module constructs 5 merging conditions:
  M1 (Condition 1): Global uniform averaging
  M2 (Condition 2): Branch-uniform averaging (cls_head, reg_head averaged independently)
  M3 (Condition 3): Dirichlet-sampled branch coefficients via coordinate descent
  M4 (Condition 4): Fisher/Hessian-weighted branch coefficients
  M5 (Condition 5): Learned soup with optimized mixing coefficients α and temperature β
                    per Wortsman et al. (2022) eq. (2)

For each condition, we:
  1. Build the merged state-dict
  2. Evaluate on the held-out eval split (4000 images)
  3. Extract per-class AP array (80 classes) + secondary metrics (mAP50, AR@100)
  4. Save checkpoint and results JSON

Key outputs:
  - phase3_soup_results.json — per-class AP arrays + summary metrics for all conditions
  - branch_uniform_soup.pth — Condition 2 (M2) checkpoint
  - best_learned_soup.pth — best of Conditions 3, 4, or 5 (M3, M4, or M5) checkpoint

Run: python -m yolof_soup.experiments.phase3_soup_construction
"""

from __future__ import annotations

from datetime import datetime
import os
import copy
import json
import time
import logging
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch.multiprocessing as mp

import torch

from tabulate import tabulate

from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog

from yolof.utils import _format_duration
from yolof.modeling.yolof import permute_to_N_HWA_K

from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    DEVICE,
    TRAIN_DATASET,
    RESULTS_DIR,
    SELECTION_DATASET,
    EVAL_DATASET,
    PHASE2_OUTPUT_DIR,
    _register_datasets,
    build_eval_cfg,
)
from yolof_soup.config.experiment_registry import get_run_specs
from yolof_soup.utils.inference import EvaluateModel
from yolof_soup.utils.checkpoint_utils import load_states, load_state, save_checkpoint
from yolof_soup.utils.eval_utils import (
    build_eval_dataloader,
    build_train_dataloader,
    get_map,
    extract_per_class_ap
)
from yolof_soup.utils.key_utils import (
    apply_uniform_lambdas,
    compute_anchor,
    compute_task_vectors,
    extract_subdict,
    get_decoder_keys,
    get_backbone_encoder_keys,
    merge_subdicts,
    split_decoder_subheads,
    calibrate_bn
)
from yolof_soup.utils.state_dict_utils import assign_state_to_model
from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor
from yolof_soup.utils.global_logger import get_logger


logger = None

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (can be adjusted)
# ─────────────────────────────────────────────────────────────────────────────

#: Coordinate descent λ grid for Dirichlet search (Condition 3)
CD_LAMBDA_GRID: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

#: Whether to use Hessian (expensive) or L2 proxy for Fisher weights
USE_HESSIAN_FOR_FISHER: bool = False

#: Learned Soup hyperparameters (Condition 5)
LEARNED_SOUP_LR: float = 0.01
LEARNED_SOUP_EPOCHS: int = 20
LEARNED_SOUP_PATIENCE: int = 5
LEARNED_SOUP_BATCH_SIZE: int = 16

# NEW: Regularization hyperparameters (academic literature)
LEARNED_SOUP_ENTROPY_WEIGHT: float = 0.1      # KL penalty toward uniform (prevent collapse)
LEARNED_SOUP_TEMP_MIN: float = 0.5            # Min temperature β (Guo et al., 2017)
LEARNED_SOUP_TEMP_MAX: float = 2.0            # Max temperature β  
LEARNED_SOUP_GRAD_CLIP: float = 1.0           # Gradient clipping to prevent divergence
LEARNED_SOUP_ALPHA_THRESHOLD: float = 0.05    # Warn if ingredient weight < 5% (ingredient filtering)

CORE_GROUPS = [
    list(range(0, 4)),    # Model 0 → cores 0-3
    list(range(4, 8)),    # Model 1 → cores 4-7
    list(range(8, 12)),   # Model 2 → cores 8-11
    list(range(12, 16)),  # Model 3 → cores 12-15
    list(range(16, 20)),  # Model 4 → cores 16-19
    list(range(20, 24)),  # Model 5 → cores 20-23
]


# ─────────────────────────────────────────────────────────────────────────────
# Condition 1 (M1): Global Uniform Averaging
# ─────────────────────────────────────────────────────────────────────────────

def build_global_uniform_souped_model(
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

def build_branch_uniform(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M2 (Condition 2): Branch-partitioned uniform averaging.

    Partition decoder into cls_head, reg_head, and shared subparts.
    Average cls_head uniformly, reg_head uniformly, shared uniformly.
    Keep backbone+encoder from ingredient 0 (same as input).
    """
    logger.info("Building Condition 2 (M2): Branch-Uniform Averaging")

    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)

    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)

    cls_states = [extract_subdict(sd, cls_keys)    for sd in ingredient_states]
    reg_states = [extract_subdict(sd, reg_keys)    for sd in ingredient_states]
    shared_states = [extract_subdict(sd, shared_keys) for sd in ingredient_states]

    cls_avg = compute_anchor(cls_states)
    reg_avg = compute_anchor(reg_states)
    shared_avg = compute_anchor(shared_states)

    soup = merge_subdicts(be_dict, cls_avg, reg_avg, shared_avg)
    logger.info("  ✓ Branch-uniform soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


# ─────────────────────────────────────────────────────────────────────────────
# Condition 3 (M3): Dirichlet-Sampled Branch Coefficients (Coordinate Descent)
# ─────────────────────────────────────────────────────────────────────────────

def build_dirichlet_cd(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
) -> Dict[str, torch.Tensor]:
    """
    M3 (Condition 3): Dirichlet simplex search via coordinate descent.

    Use selection split to find best λ values for cls and reg branches.
    Then construct soup with those weights.
    """
    logger.info("Building Condition 3 (M3): Dirichlet-Sampled Branch Coefficients (CD Search)")

    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)

    cls_states = [extract_subdict(sd, cls_keys)    for sd in ingredient_states]
    reg_states = [extract_subdict(sd, reg_keys)    for sd in ingredient_states]
    shared_states = [extract_subdict(sd, shared_keys) for sd in ingredient_states]

    anchor_cls = compute_anchor(cls_states)
    anchor_reg = compute_anchor(reg_states)
    anchor_shared = compute_anchor(shared_states)

    cls_taus = compute_task_vectors(cls_states,    anchor_cls)
    reg_taus = compute_task_vectors(reg_states,    anchor_reg)
    shared_taus = compute_task_vectors(shared_states, anchor_shared)

    best_lam_cls = _coordinate_descent_search(
        anchor_cls, anchor_reg, anchor_shared,
        cls_taus, reg_taus, shared_taus,
        be_dict, cfg, "cls"
    )
    best_lam_reg = _coordinate_descent_search(
        anchor_cls, anchor_reg, anchor_shared,
        cls_taus, reg_taus, shared_taus,
        be_dict, cfg, "reg"
    )

    logger.info("  → CD search found: λ_cls=%.3f, λ_reg=%.3f", best_lam_cls, best_lam_reg)

    cls_merged = apply_uniform_lambdas(anchor_cls,    cls_taus,    best_lam_cls)
    reg_merged = apply_uniform_lambdas(anchor_reg,    reg_taus,    best_lam_reg)
    shared_merged = apply_uniform_lambdas(anchor_shared, shared_taus, 1.0)

    soup = merge_subdicts(be_dict, cls_merged, reg_merged, shared_merged)
    logger.info("  ✓ Dirichlet soup built. Size: %.2f MB", _state_dict_size_mb(soup))
    return soup


def _search(
        search_branch: str,
        lam: float,
        anchor_cls: Dict[str, torch.Tensor],
        taus_cls: List[Dict[str, torch.Tensor]],
        anchor_reg: Dict[str, torch.Tensor],
        taus_reg: List[Dict[str, torch.Tensor]],
        anchor_shared: Dict[str, torch.Tensor],
        taus_shared: List[Dict[str, torch.Tensor]],
        be_dict: Dict[str, torch.Tensor],
        cfg: CfgNode
    ) -> tuple[float, float]:

    os.sched_setaffinity(0, CORE_GROUPS[CD_LAMBDA_GRID.index(lam) % len(CORE_GROUPS)])
    torch.set_num_threads(1)

    _register_datasets()
    model = None

    try:
        logger.info("  [CD search %s] Evaluating λ=%.3f...", search_branch, lam)
        if search_branch == "cls":
            cls_merged = apply_uniform_lambdas(anchor_cls,    taus_cls,    lam)
            reg_merged = apply_uniform_lambdas(anchor_reg,    taus_reg,    1.0)
            shared_merged = apply_uniform_lambdas(anchor_shared, taus_shared, 1.0)
        elif search_branch == "reg":
            cls_merged = apply_uniform_lambdas(anchor_cls,    taus_cls,    1.0)
            reg_merged = apply_uniform_lambdas(anchor_reg,    taus_reg,    lam)
            shared_merged = apply_uniform_lambdas(anchor_shared, taus_shared, 1.0)
        else:
            raise ValueError(f"Unknown branch: {search_branch}")

        full_state = merge_subdicts(be_dict, cls_merged, reg_merged, shared_merged)
        model = EvaluateModel(cfg, state_dict=full_state)
        map_val = get_map(model, cfg, SELECTION_DATASET)
        logger.info("  [CD search %s] λ=%.3f → mAP=%.4f", search_branch, lam, map_val)
        return map_val, lam

    except Exception as e:
        logger.error("  [CD search %s] λ=%.3f failed: %s", search_branch, lam, str(e), exc_info=True)
        return float("nan"), lam

    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        torch.set_num_threads(torch.get_num_threads.__doc__ and 1 or 1)
        try:
            all_cores = set(range(os.cpu_count()))
            os.sched_setaffinity(0, all_cores)
        except OSError as affinity_err:
            logger.error("  [CD search %s] Could not reset CPU affinity: %s", search_branch, affinity_err)


def _worker_initializer(verbose: bool):
    global logger
    worker_id = f"worker-{os.getpid()}"
    file_name = os.path.basename(__file__).split(".")[0]
    logger = get_logger(
        level=logging.DEBUG if verbose else logging.INFO,
        add_file_handler=True,
        log_file=f"{file_name}_{worker_id}.log"
    )
    logger.propagate = False


def _coordinate_descent_search(
    anchor_cls: Dict[str, torch.Tensor],
    anchor_reg: Dict[str, torch.Tensor],
    anchor_shared: Dict[str, torch.Tensor],
    taus_cls: List[Dict[str, torch.Tensor]],
    taus_reg: List[Dict[str, torch.Tensor]],
    taus_shared: List[Dict[str, torch.Tensor]],
    be_dict: Dict[str, torch.Tensor],
    cfg: CfgNode,
    search_branch: str,
) -> float:
    """Find best λ via coordinate descent over one branch while others stay at λ=1.0."""

    best_lam = 1.0
    best_map = -1.0
    nan_count = 0

    gpu_utils = GPUMemoryMonitor(verbose=parsed_args.verbose)
    gpu_utils.start()

    ctx = mp.get_context("spawn")
    semaphore = ctx.Semaphore(len(CORE_GROUPS))
    results = []
    errors = []

    def on_success(result):
        results.append(result)
        semaphore.release()

    def error_callback(idx):
        def on_error(exc):
            logger.error("Job %d failed with error: %s: %s", idx, type(exc).__name__, exc, exc_info=exc)
            errors.append((idx, exc))
            semaphore.release()
        return on_error

    pool = ctx.Pool(
        processes=len(CORE_GROUPS),
        initializer=_worker_initializer,
        initargs=(parsed_args.verbose,)
    )
    try:
        for lam in CD_LAMBDA_GRID:
            semaphore.acquire()
            pool.apply_async(
                _search,
                (search_branch, lam, anchor_cls, taus_cls, anchor_reg, taus_reg,
                 anchor_shared, taus_shared, be_dict, cfg),
                callback=on_success,
                error_callback=error_callback(CD_LAMBDA_GRID.index(lam))
            )
        pool.close()
        pool.join()
    except Exception:
        pool.terminate()
        raise
    finally:
        pool.join()
        gpu_utils.stop()

    for map_val, lam in results:
        if np.isnan(map_val):
            nan_count += 1
        elif map_val > best_map:
            best_map = map_val
            best_lam = lam

    if nan_count == len(CD_LAMBDA_GRID):
        logger.warning("  [CD search %s] All λ values returned NaN. Defaulting to λ=1.0", search_branch)
        best_lam = 1.0

    return best_lam


# ─────────────────────────────────────────────────────────────────────────────
# Condition 4 (M4): Fisher/Hessian-Weighted Branch Coefficients
# ─────────────────────────────────────────────────────────────────────────────

def build_fisher_weighted(
    ingredient_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    M4 (Condition 4): Fisher/Hessian-weighted branch coefficients.

    Compute Fisher information (or L2 proxy) per ingredient per branch.
    Weight each ingredient inversely by Fisher magnitude.
    """
    logger.info("Building Condition 4 (M4): Fisher/Hessian-Weighted Branch Coefficients")

    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    be_dict = extract_subdict(ingredient_states[0], be_keys)

    cls_norms = [_compute_l2_norm(extract_subdict(sd, cls_keys))    for sd in ingredient_states]
    reg_norms = [_compute_l2_norm(extract_subdict(sd, reg_keys))    for sd in ingredient_states]
    shared_norms = [_compute_l2_norm(extract_subdict(sd, shared_keys)) for sd in ingredient_states]

    cls_weights = _inverse_norm_weights(cls_norms)
    reg_weights = _inverse_norm_weights(reg_norms)
    shared_weights = _inverse_norm_weights(shared_norms)

    logger.debug("  Fisher weights: cls=%s",    [f"{w:.3f}" for w in cls_weights])
    logger.debug("  Fisher weights: reg=%s",    [f"{w:.3f}" for w in reg_weights])
    logger.debug("  Fisher weights: shared=%s", [f"{w:.3f}" for w in shared_weights])

    cls_avg = _weighted_average([extract_subdict(sd, cls_keys)    for sd in ingredient_states], cls_weights)
    reg_avg = _weighted_average([extract_subdict(sd, reg_keys)    for sd in ingredient_states], reg_weights)
    shared_avg = _weighted_average([extract_subdict(sd, shared_keys) for sd in ingredient_states], shared_weights)

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
# Condition 5 (M5): Learned Soup — Wortsman et al. (2022) eq. (2)
#
# Objective (from the paper):
#   arg min_{α ∈ ℝᵏ, β ∈ ℝ}  Σⱼ ℓ( β · f(xⱼ, Σᵢ αᵢθᵢ), yⱼ )
#
# α is parameterised via softmax so that αᵢ ≥ 0 and Σαᵢ = 1.
# β multiplies the merged model's output (logits) directly:
#   β > 1 → sharpens predictions (higher confidence)
#   β < 1 → softens predictions  (lower confidence)
#   β = 1 → no change
# β is parameterised as exp(log_β) to ensure β > 0 at all times.
#
# Scalar-detach gradient strategy (memory-efficient):
#   • Run each ingredient model under torch.no_grad() — no activations retained.
#   • Collect scalar losses L̄ᵢ via .item().
#   • Move each model back to CPU before loading the next one.
#   • Reconstruct a minimal autograd graph:
#         scaled_loss = β · Σᵢ αᵢ · L̄ᵢ
#     where only alpha_raw and log_beta are leaf tensors.
#   • backward() is near-instantaneous (graph has k+1 nodes).
#   Peak GPU = 1 model + 1 batch at any time.
#
# Mixing strategy (decoder sub-heads use learned α, rest uniform):
#   Decoder sub-heads  (cls_subnet, cls_score, bbox_subnet, bbox_pred):
#       θ_key = Σᵢ αᵢ · θᵢ_key       ← learned per-ingredient weights
#   All other parameters (backbone, encoder, shared decoder trunk):
#       θ_key = (1/k) Σᵢ θᵢ_key      ← fixed uniform average
#
# At merge time β is baked into cls_score weight and bias:
#   W_cls ← β · W_cls,  b_cls ← β · b_cls
# This gives zero-cost temperature scaling at inference.
# bbox_pred (regression) is intentionally excluded — temperature scaling
# is semantically meaningful for classification logits only.
# ─────────────────────────────────────────────────────────────────────────────

# Keys whose names contain any of these substrings use learned α mixing.
# Everything else (backbone, encoder, shared decoder trunk) is uniformly averaged.
LEARNED_MIXING_SUBSTRINGS: tuple[str, ...] = (
    "cls_subnet",
    "cls_score",
    "bbox_subnet",
    "bbox_pred",
)


def _is_decoder_key(key: str) -> bool:
    """
    Return True if *key* belongs to a decoder sub-head and should use
    learned α mixing.

    Matches the four YOLOF decoder sub-head groups:
        decoder.cls_subnet.*   — classification conv layers
        decoder.cls_score.*    — classification prediction layer
        decoder.bbox_subnet.*  — box regression conv layers
        decoder.bbox_pred.*    — box regression prediction layer

    Everything else (backbone, encoder, shared decoder trunk) is uniformly
    averaged, exactly as in the original Model Soups paper.
    """
    return any(sub in key for sub in LEARNED_MIXING_SUBSTRINGS)


def _mix_states_selective(
    ingredient_states: List[Dict[str, torch.Tensor]],
    alpha: torch.Tensor,  # normalised (sum=1), detached, shape (k,)
) -> Dict[str, torch.Tensor]:
    """
    Produce the final merged state dict using two mixing strategies,
    following Wortsman et al. (2022) eq. (2).

    Decoder sub-head keys  (cls_subnet, cls_score, bbox_subnet, bbox_pred):
        θ_key = Σᵢ αᵢ · θᵢ_key        ← learned per-ingredient weights

    All other keys (backbone, encoder, shared decoder trunk):
        θ_key = (1/k) Σᵢ θᵢ_key       ← fixed uniform average

    Args:
        ingredient_states: List of k state dicts (fine-tuned checkpoints).
        alpha:             Optimised, normalised weight vector, shape (k,).
                           MUST be detached (.detach()) before calling.

    Returns:
        Merged state dict in the original dtype of each parameter.
    """
    assert abs(alpha.sum().item() - 1.0) < 1e-4, (
        f"alpha must sum to 1.0, got {alpha.sum().item():.6f}"
    )
    assert (alpha >= 0).all(), (
        "alpha must be non-negative — apply softmax before calling _mix_states_selective"
    )

    n              = len(ingredient_states)
    uniform_weight = 1.0 / n
    mixed_state: Dict[str, torch.Tensor] = {}
    decoder_keys: List[str] = []
    uniform_keys: List[str] = []

    for key in ingredient_states[0].keys():
        use_learned = _is_decoder_key(key)
        (decoder_keys if use_learned else uniform_keys).append(key)

        mixed: Optional[torch.Tensor] = None
        for i, state in enumerate(ingredient_states):
            if key not in state:
                continue
            param = state[key].float()
            weight = alpha[i].item() if use_learned else uniform_weight
            mixed = weight * param if mixed is None else mixed + weight * param

        if mixed is not None:
            mixed_state[key] = mixed.to(ingredient_states[0][key].dtype)
        else:
            mixed_state[key] = ingredient_states[0][key].clone()

    logger.info(
        "  → Selective mix: %d decoder sub-head keys (learned α=[%s]), "
        "%d other keys (uniform 1/%d)",
        len(decoder_keys),
        ", ".join(f"{a:.4f}" for a in alpha.tolist()),
        len(uniform_keys),
        n,
    )
    logger.debug(
        "  → Decoder sub-head keys mixed with α (%d total):\n    %s",
        len(decoder_keys),
        "\n    ".join(decoder_keys),
    )
    logger.debug(
        "  → Uniform-mixed key breakdown: backbone=%d  encoder=%d  other=%d",
        sum(1 for k in uniform_keys if k.startswith("backbone.")),
        sum(1 for k in uniform_keys if k.startswith("encoder.")),
        sum(1 for k in uniform_keys
            if not k.startswith("backbone.") and not k.startswith("encoder.")),
    )
    return mixed_state


def _apply_temperature_to_cls_score(
    state_dict: Dict[str, torch.Tensor],
    beta: float,
) -> Dict[str, torch.Tensor]:
    """
    Bake learned temperature β into cls_score weight and bias.

    Per Wortsman et al. (2022) eq. (2), β multiplies the merged model's
    output directly:
        logit_scaled = β · f(x, θ_soup)
                     ≡ (β · W_cls) @ x + (β · b_cls)

    Absorbing β into cls_score.weight and cls_score.bias gives zero-cost
    temperature scaling at inference — no extra operations needed.

        β > 1 → sharpens predictions (higher confidence)
        β < 1 → softens predictions  (lower confidence)
        β = 1 → no change

    Only cls_score keys are modified. bbox_pred (regression) is intentionally
    excluded — temperature scaling is meaningful for classification logits only,
    not continuous box delta predictions.

    Args:
        state_dict: Merged model state dict (post α-mixing).
        beta:       Learned temperature scalar β > 0.

    Returns:
        New state dict with cls_score.weight and cls_score.bias scaled by β.
    """
    if abs(beta - 1.0) < 1e-6:
        logger.info("  → β ≈ 1.0 — temperature scaling is a no-op, skipping.")
        return state_dict

    scaled = {k: v.clone() for k, v in state_dict.items()}
    scaled_keys: List[str] = []

    for key in scaled:
        if "cls_score" in key and (key.endswith(".weight") or key.endswith(".bias")):
            scaled[key] = scaled[key].float().mul(beta).to(state_dict[key].dtype)
            scaled_keys.append(key)

    if not scaled_keys:
        logger.warning(
            "  _apply_temperature_to_cls_score: no cls_score.weight / cls_score.bias "
            "found in state_dict — temperature not applied. "
            "Verify that the YOLOF decoder uses 'cls_score' as the attribute name."
        )
    else:
        logger.info(
            "  → Temperature β=%.4f baked into %d cls_score key(s): %s",
            beta, len(scaled_keys), scaled_keys,
        )

    return scaled


def _get_temperature_beta(log_beta_raw_val):
    """Convert unconstrained log_beta to constrained β ∈ [0.5, 2.0]."""
    beta_unclamped = torch.sigmoid(log_beta_raw_val) * (LEARNED_SOUP_TEMP_MAX - LEARNED_SOUP_TEMP_MIN) + LEARNED_SOUP_TEMP_MIN
    return beta_unclamped


def _inference(state: Dict[str, torch.Tensor], batch: List[Dict[str, Any]], cfg: CfgNode, device: torch.device, batch_idx: int, i: int) -> float:
    os.sched_setaffinity(0, CORE_GROUPS[i % len(CORE_GROUPS)])
    torch.set_num_threads(1)

    model_i = EvaluateModel(cfg, state_dict=state)
    model_i.to(device)
    model_i.eval()

    _register_datasets()

    # Keep BN in eval mode — running stats must not drift
    for mod in model_i.model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
            mod.eval()

    try:
        with torch.no_grad():
            outputs_i = model_i(batch)

        raw_loss_i = _compute_raw_loss(outputs_i, device)
        # .item() severs any remaining graph references
        return raw_loss_i.item()

    except Exception as e:
        logger.error(
            "  Model %d, batch %d forward error: %s",
            i, batch_idx, str(e),
        )
        raise

    finally:
        model_i.to("cpu")
        del model_i
        torch.cuda.empty_cache()


def build_learned_soup(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataloader,
) -> Dict[str, torch.Tensor]:
    """
    M5 (Condition 5): Learned Soup — Wortsman et al. (2022) eq. (2).

    Jointly optimises mixing coefficients α ∈ ℝᵏ and temperature β ∈ ℝ via:

        arg min_{α, β}  Σⱼ ℓ( β · f(xⱼ, Σᵢ αᵢθᵢ), yⱼ )
        s.t.  αᵢ ≥ 0,  Σαᵢ = 1   (enforced via softmax)
              β > 0                (enforced via exp parameterisation)

    Scalar-detach gradient strategy (memory-efficient)
    --------------------------------------------------
    Because models run one at a time under torch.no_grad(), only scalar
    loss values L̄ᵢ = ℓ(f(x, θᵢ), y) are retained (not activations).
    The scalar proxy used here is:

        scaled_loss = β · Σᵢ αᵢ · L̄ᵢ

    Gradients:
        ∂/∂αᵢ  = β · L̄ᵢ            → prefers low-loss ingredients
        ∂/∂β   = Σᵢ αᵢ · L̄ᵢ        → β grows when weighted loss is large
                                       (sharpens when model is more confident)

    Peak GPU = 1 model + 1 batch at any time.

    Mixing strategy
    ---------------
    Decoder sub-heads  (cls_subnet, cls_score, bbox_subnet, bbox_pred):
        θ_key = Σᵢ αᵢ · θᵢ_key       ← learned per-ingredient weights

    All other parameters (backbone, encoder, shared decoder trunk):
        θ_key = (1/k) Σᵢ θᵢ_key      ← fixed uniform average

    Temperature at merge time
    -------------------------
    β is baked into cls_score.weight and cls_score.bias as W ← β·W, b ← β·b,
    giving zero-overhead temperature scaling at inference.
    bbox_pred is intentionally excluded (meaningful for classification only).

    Args:
        ingredient_states:    List of k fine-tuned model state dicts.
        cfg:                  Detectron2 / CfgNode model configuration.
        selection_dataloader: Held-out validation dataloader for optimisation.

    Returns:
        Merged state dict — decoder sub-heads mixed with learned α, all other
        parameters uniformly averaged, cls_score scaled by β.
    """
    logger.info("Building Condition 5 (M5): Learned Soup (Wortsman et al., 2022 eq. 2)")
    logger.info(
        "  → Decoder keys %s → learned α (softmax simplex)",
        LEARNED_MIXING_SUBSTRINGS,
    )
    logger.info(
        "  → All other keys (backbone, encoder, shared trunk) → uniform 1/%d",
        len(ingredient_states),
    )
    logger.info(
        "  → β = temperature scalar (β · logits per paper); "
        "baked into cls_score at merge time"
    )

    eval_log_freq = 100
    n_ingredients = len(ingredient_states)
    device = DEVICE

    # NEW: Initialize α with small random noise to break symmetry and encourage diversity
    # (standard practice in ensemble learning — Henderson et al., 2018)
    # Note: Create leaf tensor first, scale it, THEN enable gradients (avoid non-leaf tensor issue)
    alpha_raw = torch.randn(
        n_ingredients, device=device, dtype=torch.float32
    ) * 0.1
    alpha_raw.requires_grad_(True)

    # NEW: Parameterize temperature as constrained: β = sigmoid(log_beta_raw) * (β_max - β_min) + β_min
    # This enforces 0.5 ≤ β ≤ 2.0, preventing pathological extreme values
    # (Guo et al., 2017 — Beyond Simple Accuracy: Instance-level Calibration for Deep Learning)
    log_beta_raw = torch.tensor(
        0.0, device=device, dtype=torch.float32, requires_grad=True
    )

    optimizer = torch.optim.AdamW(
        [alpha_raw, log_beta_raw], lr=LEARNED_SOUP_LR  # Changed: log_beta_raw instead of log_beta
    )

    best_loss: float = float("inf")
    best_alpha_normalized: Optional[torch.Tensor] = None
    best_log_beta: Optional[torch.Tensor] = None
    patience_counter: int = 0

    logger.info(
        "  → Initial α (softmax of zeros): %s",
        [f"{a:.4f}" for a in torch.softmax(alpha_raw.detach(), dim=0).tolist()],
    )
    logger.info("  → Initial β (sigmoid→constrained): %.4f", _get_temperature_beta(log_beta_raw).item())

    ctx = mp.get_context("spawn")
    semaphore = ctx.Semaphore(len(CORE_GROUPS))
    errors = []

    def on_success_callback(per_model_loss_scalars):
        def on_success(result):
            per_model_loss_scalars.append(result)
            semaphore.release()
        return on_success

    def error_callback(idx):
        def on_error(exc):
            logger.error("Job %d failed with error: %s: %s", idx, type(exc).__name__, exc, exc_info=exc)
            errors.append((idx, exc))
            semaphore.release()
        return on_error

    try:
        for epoch in range(LEARNED_SOUP_EPOCHS):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(selection_dataloader):

                # ── Step 1: collect per-model loss SCALARS (no graph retained) ─
                # Each model loaded to GPU, run under no_grad, moved back to CPU.
                # Peak GPU = 1 model + 1 batch at any time.
                per_model_loss_scalars: List[float] = []

                pool = ctx.Pool(
                    processes=len(CORE_GROUPS),
                    initializer=_worker_initializer,
                    initargs=(parsed_args.verbose,)
                )
                try:
                    for i, state in enumerate(ingredient_states):
                        semaphore.acquire()
                        pool.apply_async(
                            _inference,
                            (state, batch, cfg, device, batch_idx, i),
                            callback=on_success_callback(per_model_loss_scalars),
                            error_callback=error_callback(i)
                        )
                    pool.close()
                    pool.join()
                except Exception:
                    pool.terminate()
                    raise
                finally:
                    pool.join()

                # ── Step 2: build the α / β graph over SCALAR losses ──────────
                # per_model_loss_scalars are plain Python floats here.
                # Only alpha_raw and log_beta_raw are autograd leaves.
                optimizer.zero_grad(set_to_none=True)

                alpha_normalized = torch.softmax(alpha_raw, dim=0)   # (k,)  Σ=1
                beta = _get_temperature_beta(log_beta_raw)            # scalar ∈ [0.5, 2.0]

                # Convert scalar losses to a no-grad device tensor
                loss_tensor = torch.tensor(
                    per_model_loss_scalars,
                    device=device,
                    dtype=torch.float32,
                )  # shape: (k,), no autograd history

                # ─── PRIMARY OBJECTIVE: weighted detection loss ──────────────────
                weighted_loss = (alpha_normalized * loss_tensor)
                detection_loss = beta * weighted_loss.sum()

                # ─── NEW: REGULARIZATION 1 — Entropy (encourages diversity) ─────
                # From ensemble literature: prevent α from collapsing to one ingredient
                # H(α) = -Σᵢ αᵢ log(αᵢ + ε)  [Shannon entropy]
                # We MAXIMIZE this, so add negative to loss
                eps = 1e-8
                entropy = -(alpha_normalized * torch.log(alpha_normalized + eps)).sum()
                
                # ─── NEW: REGULARIZATION 2 — KL divergence (per Wortsman et al.) ──
                # Penalize deviation from uniform distribution
                # KL(α || uniform) = Σᵢ αᵢ log(α_i / (1/k))
                #                  = Σᵢ αᵢ log(k · αᵢ)
                uniform_dist = torch.ones(n_ingredients, device=device, dtype=torch.float32) / n_ingredients
                kl_divergence = torch.sum(alpha_normalized * (torch.log(alpha_normalized + eps) - torch.log(uniform_dist + eps)))
                
                # ─── NEW: REGULARIZATION 3 — Ingredient filtering penalty ───────
                # Warn and penalize if any ingredient becomes too small (< 5%)
                # This promotes balanced ensemble use
                min_alpha = alpha_normalized.min()
                ingredient_filter_penalty = 0.0
                if min_alpha < LEARNED_SOUP_ALPHA_THRESHOLD:
                    n_weak = (alpha_normalized < LEARNED_SOUP_ALPHA_THRESHOLD).sum().item()
                    ingredient_filter_penalty = 0.05 * (LEARNED_SOUP_ALPHA_THRESHOLD - min_alpha)
                
                # ─── COMBINED LOSS ──────────────────────────────────────────────
                # L_total = L_detection + λ_entropy·H(α) + λ_kl·KL(α||U) + λ_filter·L_filter
                # All regularization terms encourage diversity & prevent mode-seeking
                total_loss = (
                    detection_loss 
                    - LEARNED_SOUP_ENTROPY_WEIGHT * entropy        # Maximize entropy (negative sign)
                    + LEARNED_SOUP_ENTROPY_WEIGHT * 0.5 * kl_divergence  # KL penalty
                    + ingredient_filter_penalty
                )
                
                # Use detection_loss for logging, but total_loss for backward()
                scaled_loss = total_loss

                if not scaled_loss.isfinite():
                    logger.warning(
                        "  Non-finite loss at epoch %d batch %d — skipping",
                        epoch, batch_idx,
                    )
                    del scaled_loss, weighted_loss, loss_tensor, detection_loss, entropy, kl_divergence
                    continue

                # NEW: Backward with gradient clipping to prevent divergence
                scaled_loss.backward()
                
                # Gradient clipping (standard practice in deep learning)
                torch.nn.utils.clip_grad_norm_(
                    [alpha_raw, log_beta_raw], 
                    max_norm=LEARNED_SOUP_GRAD_CLIP
                )
                
                optimizer.step()

                epoch_loss += detection_loss.item()  # Log detection loss, not regularized loss
                epoch_batches += 1
                
                # NEW: Log regularization breakdown for debugging
                if batch_idx > 0 and batch_idx % eval_log_freq == 0:
                    # NEW: Compute and log ingredient weights with warning for weak ingredients
                    alpha_list = alpha_normalized.detach().tolist()
                    weak_ingredients = [(i, w) for i, w in enumerate(alpha_list) if w < LEARNED_SOUP_ALPHA_THRESHOLD]
                    weak_warning = "" if not weak_ingredients else f" ⚠ Weak: {weak_ingredients}"
                    
                    logger.debug(
                        "  Epoch %d batch %d: L_det=%.4f  avg=%.4f  "
                        "α=%s  β=%.4f%s  entropy=%.4f  kl=%.4f",
                        epoch + 1,
                        batch_idx,
                        detection_loss.item(),
                        epoch_loss / max(epoch_batches, 1),
                        [f"{a:.4f}" for a in alpha_list],
                        beta.item(),
                        weak_warning,
                        entropy.item(),
                        kl_divergence.item(),
                    )

                del scaled_loss, weighted_loss, loss_tensor, detection_loss, entropy, kl_divergence

                if epoch_loss / max(epoch_batches, 1) > 100.0:
                    logger.warning(
                        "  Loss diverging at epoch %d — stopping optimisation early",
                        epoch,
                    )
                    break

            avg_loss = epoch_loss / max(epoch_batches, 1)

            # ── Early stopping ────────────────────────────────────────────────
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_alpha_normalized = torch.softmax(alpha_raw, dim=0).detach().clone()
                best_log_beta = log_beta_raw.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            # NEW: Compute epoch-level metrics for better diagnostics
            alpha_epoch = torch.softmax(alpha_raw.detach(), dim=0)
            beta_epoch = _get_temperature_beta(log_beta_raw).item()
            entropy_epoch = -(alpha_epoch * torch.log(alpha_epoch + 1e-8)).sum().item()
            max_alpha = alpha_epoch.max().item()
            min_alpha = alpha_epoch.min().item()
            
            # NEW: Alert if α is becoming too concentrated (mode-seeking behavior)
            concentration_warning = ""
            if max_alpha > 0.7:
                concentration_warning = f" ⚠ α concentration HIGH ({max_alpha:.1%} on ingredient {alpha_epoch.argmax().item()})"
            
            logger.info(
                "  Epoch %d/%d — L_det=%.4f  best=%.4f  "
                "α_range=[%.2f, %.2f]  β=%.4f  H(α)=%.4f  patience=%d/%d%s",
                epoch + 1, LEARNED_SOUP_EPOCHS,
                avg_loss, best_loss,
                min_alpha, max_alpha,
                beta_epoch,
                entropy_epoch,
                patience_counter, LEARNED_SOUP_PATIENCE,
                concentration_warning,
            )

            if patience_counter >= LEARNED_SOUP_PATIENCE:
                logger.info(
                    "  → Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1, LEARNED_SOUP_PATIENCE,
                )
                break

    except Exception:
        logger.exception("Learned soup optimisation failed.")
        raise

    # ── Fallback: uniform α and β=1.0 if no epoch improved ──────────────────
    if best_alpha_normalized is None:
        logger.warning(
            "  Optimisation produced no improvement. "
            "Falling back to uniform α and β=1.0."
        )
        best_alpha_normalized = torch.full(
            (n_ingredients,), 1.0 / n_ingredients, dtype=torch.float32
        )
        best_log_beta = torch.tensor(0.0, dtype=torch.float32)

    best_beta: float = _get_temperature_beta(best_log_beta).item()  # Apply constraint function
    best_alpha_list = best_alpha_normalized.tolist()
    best_entropy = -(best_alpha_normalized * torch.log(best_alpha_normalized + 1e-8)).sum().item()
    
    logger.info(
        "  ✓ Optimisation complete — best α=%s  best β=%.4f  H(α)=%.4f",
        [f"{a:.4f}" for a in best_alpha_list],
        best_beta,
        best_entropy,
    )
    
    # NEW: Diagnostic check for mode-seeking behavior
    max_alpha = max(best_alpha_list)
    if max_alpha > 0.7:
        logger.warning(
            "  ⚠ WARNING: Learned soup is mode-seeking (α concentration = %.1f%%). "
            "Consider increasing LEARNED_SOUP_ENTROPY_WEIGHT or using uniform averaging instead.",
            max_alpha * 100,
        )

    # ── Merge: α-mix decoder sub-heads, uniform-mix everything else ──────────
    merged_state = _mix_states_selective(ingredient_states, best_alpha_normalized)

    # ── Bake β into cls_score weight and bias (zero inference overhead) ───────
    # W_cls ← β·W_cls, b_cls ← β·b_cls, per Wortsman et al. eq. (2): β·f(x, θ)
    merged_state = _apply_temperature_to_cls_score(merged_state, best_beta)

    logger.info(
        "  ✓ Learned soup built. Size: %.2f MB",
        _state_dict_size_mb(merged_state),
    )
    return merged_state


# ─────────────────────────────────────────────────────────────────────────────
# Kept for reference — full learned α mix (all keys)
# ─────────────────────────────────────────────────────────────────────────────

def _mix_states_with_alpha(
    ingredient_states: List[Dict[str, torch.Tensor]],
    alpha: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Merge ALL parameters with learned α.

        θ_soup = Σᵢ αᵢ θᵢ

    NOTE: .item() is intentional — called only after optimisation with
    a detached α. Gradients are not required here.
    """
    mixed_state = {}
    for key in ingredient_states[0].keys():
        mixed = None
        for i, state in enumerate(ingredient_states):
            if key in state:
                param  = state[key].float()
                weight = alpha[i].item()
                mixed  = weight * param if mixed is None else mixed + weight * param
        if mixed is not None:
            mixed_state[key] = mixed.to(ingredient_states[0][key].dtype)
        else:
            mixed_state[key] = ingredient_states[0][key].clone()
    return mixed_state


# ─────────────────────────────────────────────────────────────────────────────
# Raw (un-scaled) loss from a single model's forward output.
# Called under torch.no_grad() — returns a graph-free scalar tensor.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_raw_loss(
    outputs,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract and sum detection losses from one model's forward output.

    Called under torch.no_grad(), so the returned tensor has no autograd
    history. The caller converts it to a Python float via .item() before
    it is used in the α/β graph.

    Supported output formats
    ------------------------
    Format A (preferred — Detectron2 / YOLOF style):
        outputs = [
            {
                "instances": <Instances>,
                "losses": {
                    "loss_cls":     Tensor,
                    "loss_box_reg": Tensor,
                    ...
                }
            },
            ...
        ]

    Format B (fallback — tuple/list per image):
        outputs = [
            (predictions, loss_cls_tensor, loss_box_reg_tensor, ...),
            ...
        ]

    Returns:
        Scalar loss tensor (no autograd graph).

    Raises:
        ValueError: If the output format is unrecognised.
    """
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)

    if not (isinstance(outputs, list) and len(outputs) > 0):
        raise ValueError(
            f"_compute_raw_loss: expected a non-empty list, got {type(outputs)}"
        )

    batch_loss = zero.clone()

    for item in outputs:
        if isinstance(item, dict) and "losses" in item:
            for loss_val in item["losses"].values():
                if isinstance(loss_val, torch.Tensor):
                    batch_loss = batch_loss + loss_val

        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            batch_loss = batch_loss + item[1] + item[2]

        else:
            raise ValueError(
                f"_compute_raw_loss: unrecognised item format {type(item)}. "
                "Expected dict with 'losses' key, or tuple/list of length >= 3."
            )

    return batch_loss / max(len(outputs), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Condition 6 (M6): Tri-Head Learned Soup — Decoder-Only Activation Graph
#
# Three independent (α, β) pairs — one per decoder prediction sub-head:
#
#   (alpha_cls,  beta_cls)   ← cls_subnet  + cls_score   (classification)
#   (alpha_bbox, beta_bbox)  ← bbox_subnet + bbox_pred   (box regression)
#   (alpha_obj,  beta_obj)   ← object_pred               (objectness)
#
# Objective (Wortsman et al., 2022 eq. (2), exact for decoder parameters):
#
#   arg min_{α_cls, β_cls, α_bbox, β_bbox, α_obj, β_obj}
#       Σⱼ ℓ( β_cls  · f_cls (zⱼ, Σᵢ α_cls_i  · θᵢ_cls ),  yⱼ )
#     + Σⱼ ℓ( β_bbox · f_bbox(zⱼ, Σᵢ α_bbox_i · θᵢ_bbox), yⱼ )
#     + Σⱼ ℓ( β_obj  · f_obj (zⱼ, Σᵢ α_obj_i  · θᵢ_obj ),  yⱼ )
#
#   where zⱼ = backbone_encoder(xⱼ)  [detached — no graph retained]
#
# Temperature semantics:
#   beta_cls  > 1 → sharper cls  logits   (classification confidence)
#   beta_bbox > 1 → larger  bbox deltas   (regression scale)
#   beta_obj  > 1 → sharper obj  logits   (foreground/background)
#   All three are baked into static weights at merge time — zero inference cost.
#
# Backbone / encoder / shared decoder trunk → fixed uniform 1/k (no_grad).
# Peak GPU = 1 model skeleton + 1 batch + decoder activations only.
# ─────────────────────────────────────────────────────────────────────────────

from torch.func import functional_call   # requires PyTorch >= 2.0


# ── Sub-head substring groups ─────────────────────────────────────────────────
CLS_HEAD_SUBSTRINGS:  tuple[str, ...] = ("cls_subnet",  "cls_score")
BBOX_HEAD_SUBSTRINGS: tuple[str, ...] = ("bbox_subnet", "bbox_pred")
OBJ_HEAD_SUBSTRINGS:  tuple[str, ...] = ("object_pred",)


# ── Key-classification helpers ────────────────────────────────────────────────

def _is_cls_head_key(key: str) -> bool:
    """Return True if *key* belongs to the classification decoder sub-head."""
    return any(sub in key for sub in CLS_HEAD_SUBSTRINGS)


def _is_bbox_head_key(key: str) -> bool:
    """Return True if *key* belongs to the bbox regression decoder sub-head."""
    return any(sub in key for sub in BBOX_HEAD_SUBSTRINGS)


def _is_obj_head_key(key: str) -> bool:
    """Return True if *key* belongs to the objectness decoder sub-head."""
    return any(sub in key for sub in OBJ_HEAD_SUBSTRINGS)


def _partition_keys(
    state_dict: Dict[str, torch.Tensor],
) -> tuple[List[str], List[str], List[str], List[str]]:
    """
    Partition all keys in *state_dict* into four disjoint lists:

        cls_keys   — classification sub-head  (alpha_cls  mixing)
        bbox_keys  — regression sub-head      (alpha_bbox mixing)
        obj_keys   — objectness sub-head      (alpha_obj  mixing)
        other_keys — backbone, encoder, shared trunk (uniform average)

    Priority order (to avoid double-assignment if key matches multiple groups):
        cls > bbox > obj > other
    """
    cls_keys, bbox_keys, obj_keys, other_keys = [], [], [], []
    for key in state_dict:
        if _is_cls_head_key(key):
            cls_keys.append(key)
        elif _is_bbox_head_key(key):
            bbox_keys.append(key)
        elif _is_obj_head_key(key):
            obj_keys.append(key)
        else:
            other_keys.append(key)
    return cls_keys, bbox_keys, obj_keys, other_keys


# ── Uniform backbone+encoder state (fixed, built once before the loop) ────────

def _build_uniform_backbone_encoder(
    ingredient_states: List[Dict[str, torch.Tensor]],
    other_keys: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Uniform average of backbone, encoder, and shared trunk keys.
    This dict is loaded once into the model and never updated during
    the optimisation loop.
    """
    n = len(ingredient_states)
    averaged: Dict[str, torch.Tensor] = {}
    for key in other_keys:
        mixed: Optional[torch.Tensor] = None
        for state in ingredient_states:
            if key not in state:
                continue
            p = state[key].float()
            mixed = p / n if mixed is None else mixed + p / n
        if mixed is not None:
            averaged[key] = mixed.to(ingredient_states[0][key].dtype)
        else:
            averaged[key] = ingredient_states[0][key].clone()
    return averaged


# ── In-graph blended decoder parameter dict ──────────────────────────────────

def _blend_decoder_params(
    ingredient_cls_params: List[Dict[str, torch.Tensor]],
    ingredient_bbox_params: List[Dict[str, torch.Tensor]],
    ingredient_obj_params: List[Dict[str, torch.Tensor]],
    cls_keys: List[str],
    bbox_keys: List[str],
    obj_keys: List[str],
    alpha_cls: torch.Tensor,   # softmax-normalised, shape (k,) — IN graph
    alpha_bbox: torch.Tensor,   # softmax-normalised, shape (k,) — IN graph
    alpha_obj: torch.Tensor,   # softmax-normalised, shape (k,) — IN graph
    beta_cls: torch.Tensor,   # exp-constrained scalar — IN graph
    beta_bbox: torch.Tensor,   # exp-constrained scalar — IN graph
    beta_obj: torch.Tensor,   # exp-constrained scalar — IN graph
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build the blended decoder parameter dict that remains in the autograd graph.

    Each group uses its own α vector for the convex combination:
        cls  head: θ_blend = Σᵢ alpha_cls_i  · θᵢ  (+ inline beta_cls  on cls_score)
        bbox head: θ_blend = Σᵢ alpha_bbox_i · θᵢ  (+ inline beta_bbox on bbox_pred)
        obj  head: θ_blend = Σᵢ alpha_obj_i  · θᵢ  (+ inline beta_obj  on object_pred)

    β values are applied inline to prediction-layer weights/biases so that
    gradients flow from the loss through β → log_beta → autograd graph.

    Args:
        ingredient_cls_params: CPU-pinned cls  param dicts per ingredient.
        ingredient_bbox_params: CPU-pinned bbox param dicts per ingredient.
        ingredient_obj_params: CPU-pinned obj  param dicts per ingredient.
        cls_keys / bbox_keys / obj_keys: Pre-partitioned key lists.
        alpha_cls / alpha_bbox / alpha_obj: In-graph normalised weight vectors.
        beta_cls  / beta_bbox  / beta_obj:  In-graph temperature scalars.
        device: Target CUDA device.

    Returns:
        Dict mapping state-dict key → blended float32 tensor (in autograd graph).
    """
    blended: Dict[str, torch.Tensor] = {}

    # ── Classification head ───────────────────────────────────────────────────
    for key in cls_keys:
        mixed: Optional[torch.Tensor] = None
        for i, params in enumerate(ingredient_cls_params):
            if key not in params:
                continue
            p = params[key].to(device=device, dtype=torch.float32)
            mixed = alpha_cls[i] * p if mixed is None else mixed + alpha_cls[i] * p
        if mixed is None:
            mixed = ingredient_cls_params[0].get(key, torch.zeros(1)).to(
                device=device, dtype=torch.float32)
        # Inline β_cls on cls_score prediction layer only
        if "cls_score" in key:
            mixed = beta_cls * mixed
        blended[key] = mixed

    # ── Bounding-box regression head ──────────────────────────────────────────
    for key in bbox_keys:
        mixed = None
        for i, params in enumerate(ingredient_bbox_params):
            if key not in params:
                continue
            p = params[key].to(device=device, dtype=torch.float32)
            mixed = alpha_bbox[i] * p if mixed is None else mixed + alpha_bbox[i] * p
        if mixed is None:
            mixed = ingredient_bbox_params[0].get(key, torch.zeros(1)).to(
                device=device, dtype=torch.float32)
        # Inline β_bbox on bbox_pred prediction layer only
        if "bbox_pred" in key:
            mixed = beta_bbox * mixed
        blended[key] = mixed

    # ── Objectness head ───────────────────────────────────────────────────────
    for key in obj_keys:
        mixed = None
        for i, params in enumerate(ingredient_obj_params):
            if key not in params:
                continue
            p = params[key].to(device=device, dtype=torch.float32)
            mixed = alpha_obj[i] * p if mixed is None else mixed + alpha_obj[i] * p
        if mixed is None:
            mixed = ingredient_obj_params[0].get(key, torch.zeros(1)).to(
                device=device, dtype=torch.float32)
        # Inline β_obj on object_pred prediction layer only
        if "object_pred" in key:
            mixed = beta_obj * mixed
        blended[key] = mixed

    return blended


# ── Encoder-feature extraction (no_grad, no activations retained) ─────────────

def _extract_encoder_features(
    model,
    batch: List[Dict[str, Any]],
) -> Any:
    """
    Run backbone and encoder under torch.no_grad().
    Returns a fully detached encoder feature tensor. No activations retained.
    """
    with torch.no_grad():
        m = model.model
        images = m.preprocess_image(batch)
        features = m.backbone(images.tensor)
        features = [features[m.backbone_level]]
        encoder_out = m.encoder(features[0])

    if isinstance(encoder_out, torch.Tensor):
        return encoder_out.detach()
    if isinstance(encoder_out, dict):
        return {k: v.detach() for k, v in encoder_out.items()}
    return [v.detach() if isinstance(v, torch.Tensor) else v for v in encoder_out]


def _split_params_and_buffers(
    module: torch.nn.Module,
    local_params: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Split a flat local-key dict into two dicts that functional_call accepts:

        params  — keys that are registered nn.Parameters in *module*
                  (conv weights, biases, BN affine weight/bias)
        buffers — keys that are registered buffers in *module*
                  (BN running_mean, running_var, num_batches_tracked, etc.)

    Any key not found in either registry is placed into params (safe default).

    BN buffers must NOT carry requires_grad — cudnn_batch_norm raises
    RuntimeError if they do.  This function strips the grad by calling
    .detach() on every buffer tensor.

    Args:
        module:       The nn.Module whose parameter/buffer registry is used
                      (typically model.decoder).
        local_params: Flat dict keyed by local attribute path (no module prefix).

    Returns:
        (params_dict, buffers_dict) — safe to pass directly to functional_call.
    """
    # Build lookup sets from the module's own registries
    param_names: set[str] = {name for name, _ in module.named_parameters()}
    buffer_names: set[str] = {name for name, _ in module.named_buffers()}

    params_dict: Dict[str, torch.Tensor] = {}
    buffers_dict: Dict[str, torch.Tensor] = {}

    for key, tensor in local_params.items():
        if key in buffer_names:
            # Buffers must never have requires_grad — detach unconditionally
            buffers_dict[key] = tensor.detach()
        else:
            # Covers registered parameters AND any unregistered keys
            params_dict[key] = tensor

    return params_dict, buffers_dict


# ── Decoder-only forward with functional_call ─────────────────────────────────
def _decoder_forward_with_blended_params(
    model,
    encoder_features,
    blended_decoder_params: Dict[str, torch.Tensor],
    batch: List[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """
    Run the YOLOF decoder with blended parameters injected via functional_call.

    The autograd graph flows through blended_decoder_params back to
    alpha_cls, alpha_bbox, alpha_obj, beta_cls, beta_bbox, and beta_obj.

    BN running statistics (running_mean, running_var, num_batches_tracked) are
    passed separately as detached buffers — functional_call requires that
    non-differentiable buffers do NOT carry requires_grad, otherwise
    cudnn_batch_norm raises a RuntimeError.

    The decoder is kept in eval() mode so that BN uses the frozen running stats
    rather than the current batch stats.  This is consistent with the rest of
    the soup construction pipeline.

    Returns a scalar mean loss tensor with a live autograd graph.
    """
    m = model.model
    gt_instances = [x["instances"].to(device) for x in batch]

    # ── Strip module prefix from keys ─────────────────────────────────────────
    decoder_prefix = "model.decoder."
    decoder_prefix2 = "decoder."
    local_params: Dict[str, torch.Tensor] = {}
    for full_key, tensor in blended_decoder_params.items():
        if full_key.startswith(decoder_prefix):
            local_key = full_key[len(decoder_prefix):]
        elif full_key.startswith(decoder_prefix2):
            local_key = full_key[len(decoder_prefix2):]
        else:
            local_key = full_key
        local_params[local_key] = tensor

    # ── Split into differentiable params and non-differentiable buffers ───────
    # BN buffers (running_mean, running_var, num_batches_tracked) must be
    # detached — cudnn_batch_norm errors if requires_grad is True on them.
    params_dict, buffers_dict = _split_params_and_buffers(m.decoder, local_params)

    # ── Decoder must be in eval mode to use frozen BN running stats ───────────
    # (model_skeleton.eval() was already called in the outer function, but we
    #  enforce it here defensively in case any code path re-enters train mode.)
    m.decoder.eval()

    anchors_image = m.anchor_generator(encoder_features)
    anchors = [copy.deepcopy(anchors_image) for _ in range(len(batch))]

    with torch.enable_grad():
        
        pred_logits, pred_anchor_deltas = functional_call(
            m.decoder,
            {**params_dict, **buffers_dict},   # merged dict — functional_call
            (encoder_features,),               # accepts params & buffers together
        )

        pred_logits = [permute_to_N_HWA_K(pred_logits, m.num_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(pred_anchor_deltas, 4)]

        indices = m.get_ground_truth(anchors, pred_anchor_deltas, gt_instances)
        
        losses: Dict[str, torch.Tensor] = m.losses(
                indices, 
                gt_instances, 
                anchors,
                pred_logits, 
                pred_anchor_deltas,
            )

    total_loss = torch.stack([v for v in losses.values() if isinstance(v, torch.Tensor)]).sum()
    return total_loss / max(len(batch), 1)



# ── Final state-dict merge helpers ────────────────────────────────────────────

def _mix_states_tri_head(
    ingredient_states: List[Dict[str, torch.Tensor]],
    alpha_cls: torch.Tensor,   # detached, normalised, shape (k,)
    alpha_bbox: torch.Tensor,   # detached, normalised, shape (k,)
    alpha_obj: torch.Tensor,   # detached, normalised, shape (k,)
) -> Dict[str, torch.Tensor]:
    """
    Produce the final merged state dict using four mixing strategies:

        cls_subnet + cls_score   → Σᵢ alpha_cls_i  · θᵢ_key  (learned)
        bbox_subnet + bbox_pred  → Σᵢ alpha_bbox_i · θᵢ_key  (learned)
        object_pred              → Σᵢ alpha_obj_i  · θᵢ_key  (learned)
        backbone / encoder / trunk → (1/k) Σᵢ θᵢ_key         (uniform)

    Args:
        ingredient_states: List of k fine-tuned state dicts.
        alpha_cls / alpha_bbox / alpha_obj: Detached, normalised vectors.

    Returns:
        Merged state dict in original dtype of each parameter.
    """
    for name, alpha in [
        ("alpha_cls",  alpha_cls),
        ("alpha_bbox", alpha_bbox),
        ("alpha_obj",  alpha_obj),
    ]:
        assert abs(alpha.sum().item() - 1.0) < 1e-4, (
            f"{name} must sum to 1.0, got {alpha.sum().item():.6f}"
        )
        assert (alpha >= 0).all(), (
            f"{name} must be non-negative — apply softmax before calling"
        )

    n = len(ingredient_states)
    mixed_state: Dict[str, torch.Tensor] = {}
    cls_keys_log, bbox_keys_log, obj_keys_log, uniform_keys_log = [], [], [], []

    for key in ingredient_states[0].keys():
        is_cls = _is_cls_head_key(key)
        is_bbox = _is_bbox_head_key(key)
        is_obj = _is_obj_head_key(key)

        if is_cls:
            cls_keys_log.append(key)
        elif is_bbox:
            bbox_keys_log.append(key)
        elif is_obj:
            obj_keys_log.append(key)
        else:
            uniform_keys_log.append(key)

        mixed: Optional[torch.Tensor] = None
        for i, state in enumerate(ingredient_states):
            if key not in state:
                continue
            param = state[key].float()
            if is_cls:
                weight = alpha_cls[i].item()
            elif is_bbox:
                weight = alpha_bbox[i].item()
            elif is_obj:
                weight = alpha_obj[i].item()
            else:
                weight = 1.0 / n
            mixed = weight * param if mixed is None else mixed + weight * param

        if mixed is not None:
            mixed_state[key] = mixed.to(ingredient_states[0][key].dtype)
        else:
            mixed_state[key] = ingredient_states[0][key].clone()

    logger.info(
        "  → Tri-head mix: %d cls keys (alpha_cls=[%s])  |  "
        "%d bbox keys (alpha_bbox=[%s])  |  "
        "%d obj keys (alpha_obj=[%s])  |  "
        "%d uniform keys (1/%d)",
        len(cls_keys_log),
        ", ".join(f"{a:.4f}" for a in alpha_cls.tolist()),
        len(bbox_keys_log),
        ", ".join(f"{a:.4f}" for a in alpha_bbox.tolist()),
        len(obj_keys_log),
        ", ".join(f"{a:.4f}" for a in alpha_obj.tolist()),
        len(uniform_keys_log),
        n,
    )
    logger.debug(
        "  → Uniform-mixed key breakdown: backbone=%d  encoder=%d  other=%d",
        sum(1 for k in uniform_keys_log if k.startswith("backbone.")),
        sum(1 for k in uniform_keys_log if k.startswith("encoder.")),
        sum(1 for k in uniform_keys_log
            if not k.startswith("backbone.") and not k.startswith("encoder.")),
    )
    if not obj_keys_log:
        logger.warning(
            "  ⚠ No 'object_pred' keys found in state dict — alpha_obj / beta_obj "
            "had no effect. Verify that your YOLOF variant includes an objectness head."
        )
    return mixed_state


def _apply_temperature_tri_head(
    state_dict: Dict[str, torch.Tensor],
    beta_cls: float,
    beta_bbox: float,
    beta_obj: float,
) -> Dict[str, torch.Tensor]:
    """
    Bake learned temperatures into prediction-layer weights and biases.

        cls_score.weight   ← beta_cls  · cls_score.weight
        cls_score.bias     ← beta_cls  · cls_score.bias
        bbox_pred.weight   ← beta_bbox · bbox_pred.weight
        bbox_pred.bias     ← beta_bbox · bbox_pred.bias
        object_pred.weight ← beta_obj  · object_pred.weight
        object_pred.bias   ← beta_obj  · object_pred.bias

    All three scalings are zero-cost at inference — absorbed into static weights.

    Args:
        state_dict: Merged model state dict (post α-mixing).
        beta_cls:   Learned temperature for classification head (β > 0).
        beta_bbox:  Learned temperature for regression head      (β > 0).
        beta_obj:   Learned temperature for objectness head      (β > 0).

    Returns:
        New state dict with prediction-layer keys scaled by their respective β.
    """
    scaled = {k: v.clone() for k, v in state_dict.items()}

    head_specs = [
        ("cls_score",   beta_cls,  "beta_cls"),
        ("bbox_pred",   beta_bbox, "beta_bbox"),
        ("object_pred", beta_obj,  "beta_obj"),
    ]

    for substr, beta, beta_name in head_specs:
        scaled_keys: List[str] = []
        for key in scaled:
            if substr in key and (key.endswith(".weight") or key.endswith(".bias")):
                if abs(beta - 1.0) >= 1e-6:
                    scaled[key] = (
                        scaled[key].float().mul(beta).to(state_dict[key].dtype)
                    )
                scaled_keys.append(key)

        if abs(beta - 1.0) < 1e-6:
            logger.info(
                "  → %s ≈ 1.0 — %s temperature scaling is a no-op.",
                beta_name, substr,
            )
        elif not scaled_keys:
            logger.warning(
                "  _apply_temperature_tri_head: no %s.weight / %s.bias found — "
                "%s not applied. Verify decoder attribute name.",
                substr, substr, beta_name,
            )
        else:
            logger.info(
                "  → %s=%.4f baked into %d %s key(s): %s",
                beta_name, beta, len(scaled_keys), substr, scaled_keys,
            )

    return scaled


# ── Main entry point ──────────────────────────────────────────────────────────

def build_tri_head_learned_soup(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataloader,
) -> Dict[str, torch.Tensor]:
    """
    M6 (Condition 6): Tri-Head Learned Soup — Decoder-Only Activation Graph.

    Maintains three independent (α, β) pairs for the three decoder prediction
    sub-heads, while backbone and encoder are fixed under uniform averaging:

        (alpha_cls,  beta_cls)   ← cls_subnet  + cls_score   (classification)
        (alpha_bbox, beta_bbox)  ← bbox_subnet + bbox_pred   (box regression)
        (alpha_obj,  beta_obj)   ← object_pred               (objectness)

    Gradient strategy — exact Wortsman et al. (2022) eq. (2) for decoder:
        1. Uniform-average backbone+encoder once; load into model skeleton.
        2. Per batch:
           a. backbone+encoder forward under no_grad → detached feature z.
           b. Build blended decoder params in-graph via α vectors.
              β values applied inline to prediction-layer tensors.
           c. Decoder forward via functional_call with blended params.
           d. Detection loss backward — exact gradients for all 6 leaf tensors.
        3. Early stopping on validation loss plateau.

    Learnable parameters (6 leaf tensors):
        alpha_cls_raw  (k,) — unconstrained logits → softmax → alpha_cls
        alpha_bbox_raw (k,) — unconstrained logits → softmax → alpha_bbox
        alpha_obj_raw  (k,) — unconstrained logits → softmax → alpha_obj
        log_beta_cls   ()   — unconstrained scalar → exp    → beta_cls
        log_beta_bbox  ()   — unconstrained scalar → exp    → beta_bbox
        log_beta_obj   ()   — unconstrained scalar → exp    → beta_obj

    All initialised to zero → uniform α, β=1.0 at optimisation start.
    Peak GPU = 1 model skeleton + 1 batch + decoder activations only.

    Args:
        ingredient_states:    List of k fine-tuned model state dicts.
        cfg:                  Detectron2 / CfgNode model configuration.
        selection_dataloader: Held-out validation dataloader for optimisation.

    Returns:
        Merged state dict — cls / bbox / obj heads mixed with their respective
        learned α and scaled by their respective β; all other parameters
        uniformly averaged.
    """
    logger.info(
        "Building Condition 6 (M6): Tri-Head Learned Soup "
        "(Decoder-Only Activation Graph — exact Wortsman et al. eq. 2)"
    )
    logger.info("  → cls  head keys %s → (alpha_cls,  beta_cls)",  CLS_HEAD_SUBSTRINGS)
    logger.info("  → bbox head keys %s → (alpha_bbox, beta_bbox)", BBOX_HEAD_SUBSTRINGS)
    logger.info("  → obj  head keys %s → (alpha_obj,  beta_obj)",  OBJ_HEAD_SUBSTRINGS)
    logger.info(
        "  → backbone/encoder/trunk → fixed uniform 1/%d (no_grad, no activations)",
        len(ingredient_states),
    )

    eval_log_freq = 100
    n_ingredients = len(ingredient_states)
    device = DEVICE

    # ── Partition keys once ───────────────────────────────────────────────────
    cls_keys, bbox_keys, obj_keys, other_keys = _partition_keys(ingredient_states[0])
    logger.info(
        "  → Key partition: %d cls  |  %d bbox  |  %d obj  |  %d backbone/encoder/trunk",
        len(cls_keys), len(bbox_keys), len(obj_keys), len(other_keys),
    )
    if not obj_keys:
        logger.warning(
            "  ⚠ No 'object_pred' keys found at partition time. "
            "alpha_obj / beta_obj will be optimised but will have no effect on mixing. "
            "Verify that your YOLOF checkpoint contains an objectness head."
        )

    # ── Build fixed uniform backbone+encoder state (loaded once) ─────────────
    uniform_be_state = _build_uniform_backbone_encoder(ingredient_states, other_keys)
    logger.info(
        "  → Uniform backbone+encoder built (%.2f MB)",
        sum(v.element_size() * v.nelement()
            for v in uniform_be_state.values()) / 1e6,
    )

    # ── Load model skeleton ───────────────────────────────────────────────────
    _register_datasets()
    model_skeleton = EvaluateModel(cfg, state_dict=ingredient_states[0])
    model_skeleton.to(device)
    model_skeleton.eval()

    for mod in model_skeleton.model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
            mod.eval()

    with torch.no_grad():
        for key, val in uniform_be_state.items():
            parts  = key.split(".")
            module = model_skeleton.model
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(val.to(device=device, dtype=param.dtype))
            else:
                setattr(module, parts[-1],
                        val.to(device=device,
                               dtype=param.dtype if hasattr(param, "dtype")
                               else val.dtype))

    logger.info("  → Model skeleton loaded with uniform backbone+encoder weights.")

    # ── CPU-pin ingredient decoder params for fast H2D transfer ──────────────
    def _pin(states, keys):
        return [
            {k: s[k].cpu().pin_memory() if s[k].is_floating_point() else s[k].cpu()
             for k in keys if k in s}
            for s in states
        ]

    ingredient_cls_params = _pin(ingredient_states, cls_keys)
    ingredient_bbox_params = _pin(ingredient_states, bbox_keys)
    ingredient_obj_params = _pin(ingredient_states, obj_keys)

    # ── Six learnable leaf tensors ────────────────────────────────────────────
    # Initialised to zero → uniform softmax, β = exp(0) = 1.0
    alpha_cls_raw = torch.zeros(n_ingredients, device=device,
                                dtype=torch.float32, requires_grad=True)
    alpha_bbox_raw = torch.zeros(n_ingredients, device=device,
                                 dtype=torch.float32, requires_grad=True)
    alpha_obj_raw = torch.zeros(n_ingredients, device=device,
                                dtype=torch.float32, requires_grad=True)
    log_beta_cls = torch.tensor(0.0, device=device,
                                dtype=torch.float32, requires_grad=True)
    log_beta_bbox = torch.tensor(0.0, device=device,
                                dtype=torch.float32, requires_grad=True)
    log_beta_obj = torch.tensor(0.0, device=device,
                                dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.AdamW(
        [alpha_cls_raw, alpha_bbox_raw, alpha_obj_raw,
         log_beta_cls,  log_beta_bbox,  log_beta_obj],
        lr=LEARNED_SOUP_LR,
    )

    best_loss: float = float("inf")
    best_alpha_cls_normalized: Optional[torch.Tensor] = None
    best_alpha_bbox_normalized: Optional[torch.Tensor] = None
    best_alpha_obj_normalized: Optional[torch.Tensor] = None
    best_log_beta_cls: Optional[torch.Tensor] = None
    best_log_beta_bbox: Optional[torch.Tensor] = None
    best_log_beta_obj: Optional[torch.Tensor] = None
    patience_counter: int = 0

    logger.info(
        "  → Initial alpha_cls  = %s",
        [f"{a:.4f}" for a in torch.softmax(alpha_cls_raw.detach(),  dim=0).tolist()],
    )
    logger.info(
        "  → Initial alpha_bbox = %s",
        [f"{a:.4f}" for a in torch.softmax(alpha_bbox_raw.detach(), dim=0).tolist()],
    )
    logger.info(
        "  → Initial alpha_obj  = %s",
        [f"{a:.4f}" for a in torch.softmax(alpha_obj_raw.detach(),  dim=0).tolist()],
    )
    logger.info(
        "  → Initial beta_cls=%.4f  beta_bbox=%.4f  beta_obj=%.4f",
        torch.exp(log_beta_cls).item(),
        torch.exp(log_beta_bbox).item(),
        torch.exp(log_beta_obj).item(),
    )

    # ── Optimisation loop ─────────────────────────────────────────────────────
    try:
        for epoch in range(LEARNED_SOUP_EPOCHS):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(selection_dataloader):

                optimizer.zero_grad(set_to_none=True)

                # Normalise α vectors and exponentiate β scalars (all in graph)
                alpha_cls_norm = torch.softmax(alpha_cls_raw,  dim=0)
                alpha_bbox_norm = torch.softmax(alpha_bbox_raw, dim=0)
                alpha_obj_norm = torch.softmax(alpha_obj_raw,  dim=0)
                beta_cls = torch.exp(log_beta_cls)
                beta_bbox = torch.exp(log_beta_bbox)
                beta_obj = torch.exp(log_beta_obj)

                # Step 1 — backbone+encoder (no_grad, detached output)
                encoder_features = _extract_encoder_features(model_skeleton, batch)

                # Step 2 — build blended decoder params (in-graph)
                blended_params = _blend_decoder_params(
                    ingredient_cls_params,
                    ingredient_bbox_params,
                    ingredient_obj_params,
                    cls_keys,
                    bbox_keys,
                    obj_keys,
                    alpha_cls_norm,
                    alpha_bbox_norm,
                    alpha_obj_norm,
                    beta_cls,
                    beta_bbox,
                    beta_obj,
                    device,
                )

                # Step 3 — decoder forward + detection loss (exact graph)
                try:
                    loss = _decoder_forward_with_blended_params(
                        model_skeleton,
                        encoder_features,
                        blended_params,
                        batch,
                        device,
                    )
                except Exception as e:
                    logger.error(
                        "  Decoder forward error at epoch %d batch %d: %s",
                        epoch, batch_idx, str(e),
                    )
                    raise

                if not loss.isfinite():
                    logger.warning(
                        "  Non-finite loss at epoch %d batch %d — skipping",
                        epoch, batch_idx,
                    )
                    del loss, blended_params, encoder_features
                    torch.cuda.empty_cache()
                    continue

                # Step 4 — exact backward through decoder activations
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [alpha_cls_raw, alpha_bbox_raw, alpha_obj_raw,
                     log_beta_cls,  log_beta_bbox,  log_beta_obj],
                    max_norm=LEARNED_SOUP_GRAD_CLIP,
                )

                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

                if batch_idx > 0 and batch_idx % eval_log_freq == 0:
                    logger.debug(
                        "  Epoch %d batch %d: loss=%.4f  avg=%.4f  "
                        "α_cls=%s β_cls=%.4f  |  "
                        "α_bbox=%s β_bbox=%.4f  |  "
                        "α_obj=%s β_obj=%.4f",
                        epoch + 1, batch_idx,
                        loss.item(),
                        epoch_loss / max(epoch_batches, 1),
                        [f"{a:.4f}" for a in alpha_cls_norm.detach().tolist()],
                        beta_cls.item(),
                        [f"{a:.4f}" for a in alpha_bbox_norm.detach().tolist()],
                        beta_bbox.item(),
                        [f"{a:.4f}" for a in alpha_obj_norm.detach().tolist()],
                        beta_obj.item(),
                    )

                del loss, blended_params, encoder_features
                torch.cuda.empty_cache()

                if epoch_loss / max(epoch_batches, 1) > 100.0:
                    logger.warning(
                        "  Loss diverging at epoch %d — stopping early", epoch
                    )
                    break

            avg_loss = epoch_loss / max(epoch_batches, 1)

            # ── Early stopping ────────────────────────────────────────────────
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_alpha_cls_normalized = torch.softmax(alpha_cls_raw,  dim=0).detach().clone()
                best_alpha_bbox_normalized = torch.softmax(alpha_bbox_raw, dim=0).detach().clone()
                best_alpha_obj_normalized = torch.softmax(alpha_obj_raw,  dim=0).detach().clone()
                best_log_beta_cls = log_beta_cls.detach().clone()
                best_log_beta_bbox = log_beta_bbox.detach().clone()
                best_log_beta_obj = log_beta_obj.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info(
                "  Epoch %d/%d — avg_loss=%.4f  best=%.4f  "
                "α_cls=%s β_cls=%.4f  |  "
                "α_bbox=%s β_bbox=%.4f  |  "
                "α_obj=%s β_obj=%.4f  |  "
                "patience=%d/%d",
                epoch + 1, LEARNED_SOUP_EPOCHS,
                avg_loss, best_loss,
                [f"{a:.4f}" for a in
                 torch.softmax(alpha_cls_raw.detach(),  dim=0).tolist()],
                torch.exp(log_beta_cls).item(),
                [f"{a:.4f}" for a in
                 torch.softmax(alpha_bbox_raw.detach(), dim=0).tolist()],
                torch.exp(log_beta_bbox).item(),
                [f"{a:.4f}" for a in
                 torch.softmax(alpha_obj_raw.detach(),  dim=0).tolist()],
                torch.exp(log_beta_obj).item(),
                patience_counter, LEARNED_SOUP_PATIENCE,
            )

            if patience_counter >= LEARNED_SOUP_PATIENCE:
                logger.info(
                    "  → Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1, LEARNED_SOUP_PATIENCE,
                )
                break

    except Exception:
        logger.exception("Tri-head learned soup optimisation failed.")
        raise

    finally:
        model_skeleton.to("cpu")
        del model_skeleton
        torch.cuda.empty_cache()
        logger.info("  → Model skeleton released from GPU.")

    # ── Fallback: uniform α and β=1.0 if no epoch improved ───────────────────
    if best_alpha_cls_normalized is None:
        logger.warning(
            "  Optimisation produced no improvement. "
            "Falling back to uniform alpha and beta=1.0 for all three heads."
        )
        uniform = torch.full(
            (n_ingredients,), 1.0 / n_ingredients, dtype=torch.float32
        )
        best_alpha_cls_normalized = uniform.clone()
        best_alpha_bbox_normalized = uniform.clone()
        best_alpha_obj_normalized = uniform.clone()
        best_log_beta_cls = torch.tensor(0.0, dtype=torch.float32)
        best_log_beta_bbox = torch.tensor(0.0, dtype=torch.float32)
        best_log_beta_obj = torch.tensor(0.0, dtype=torch.float32)

    best_beta_cls: float = torch.exp(best_log_beta_cls).item()
    best_beta_bbox: float = torch.exp(best_log_beta_bbox).item()
    best_beta_obj: float = torch.exp(best_log_beta_obj).item()

    logger.info(
        "  ✓ Optimisation complete —\n"
        "      alpha_cls=%s   beta_cls=%.4f\n"
        "      alpha_bbox=%s  beta_bbox=%.4f\n"
        "      alpha_obj=%s   beta_obj=%.4f",
        [f"{a:.4f}" for a in best_alpha_cls_normalized.tolist()],  best_beta_cls,
        [f"{a:.4f}" for a in best_alpha_bbox_normalized.tolist()], best_beta_bbox,
        [f"{a:.4f}" for a in best_alpha_obj_normalized.tolist()],  best_beta_obj,
    )

    # ── Final merge ───────────────────────────────────────────────────────────
    merged_state = _mix_states_tri_head(
        ingredient_states,
        best_alpha_cls_normalized,
        best_alpha_bbox_normalized,
        best_alpha_obj_normalized,
    )

    # ── Bake β values into prediction-layer weights (zero inference overhead) ─
    merged_state = _apply_temperature_tri_head(
        merged_state,
        best_beta_cls,
        best_beta_bbox,
        best_beta_obj,
    )

    logger.info(
        "  ✓ Tri-head learned soup built. Size: %.2f MB",
        _state_dict_size_mb(merged_state),
    )
    return merged_state


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

    model = EvaluateModel(cfg, state_dict)
    model = model.to(DEVICE)
    model.eval()

    try:
        from yolof_soup.utils.eval_utils import compute_coco_map
        results_dict = compute_coco_map(
            model, cfg, SELECTION_DATASET,
            output_dir=Path(RESULTS_DIR) / "phase3_eval", tag=tag
        )

        map_val = float(results_dict.get("AP", 0.0))
        map50_val = float(results_dict.get("AP50", 0.0))
        ar100_val = float(results_dict.get("AR-maxDets=100", 0.0))

        per_class_ap = extract_per_class_ap(
            results_dict,
            MetadataCatalog.get(SELECTION_DATASET).thing_classes
        )

        logger.info(
            "  ✓ Condition %s: mAP50:95=%.4f, mAP50=%.4f, AR@100=%.4f",
            tag, map_val, map50_val, ar100_val,
        )
    except Exception as e:
        logger.error("  ✗ Evaluation failed for %s: %s", tag, str(e), exc_info=True)
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


def setup_logger(verbose: bool = True):
    """Set up a global logger for the module."""
    global logger
    logger = get_logger(
        level=logging.DEBUG if verbose else logging.INFO,
        add_file_handler=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
MERGE_CONDITIONS = {
    "global_uniform": (build_global_uniform_souped_model, ("ingredient_states",)),
    "branch_uniform": (build_branch_uniform, ("ingredient_states",)),
    "dirichlet": (build_dirichlet_cd, ("ingredient_states", "cfg")),
    "fisher": (build_fisher_weighted, ("ingredient_states",)),
    "learned": (build_learned_soup, ("ingredient_states", "cfg", "selection_dataloader")),
    "learned_tri_head": (build_tri_head_learned_soup, ("ingredient_states", "cfg", "selection_dataloader")),
}

def run(verbose: bool = True, cal_bn: list = [], force_construction: list = []) -> Dict[str, Any]:
    """
    Main Phase 3 entry point.

    Args:
        verbose:            Whether to log debug-level progress.
        cal_bn:             Whether to calibrate BN stats using the training split
                            before evaluation (improves performance for merged models).
        force_construction: A list of condition names to force rebuild, even if cached checkpoints
                            exist. If None, no conditions are forced to rebuild.

    Returns:
        Dict with results for all 5 conditions + metadata.
    """

    start_time = time.perf_counter()
    try:

        setup_logger(verbose=verbose)


        logger.info("=" * 90)
        logger.info("PHASE 3: SOUP CONSTRUCTION & EVALUATION")
        logger.info("=" * 90)

        results_dir = Path(RESULTS_DIR);    results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(CHECKPOINT_DIR); checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ingredients_dir = Path(PHASE2_OUTPUT_DIR)

        # ── Load ingredient checkpoints ───────────────────────────────────────
        logger.info("\n[1/7] Loading ingredient checkpoints...")
        run_registry = get_run_specs()
        ingredient_runs = [r for r in run_registry if r.role == "ingredient"]
        ingredient_paths = []

        for run_spec in ingredient_runs:
            ckpt_path = Path(ingredients_dir) / f"{run_spec.run_name}/model_best.pth"
            ingredient_paths.append(ckpt_path)

        missing = [p for p in ingredient_paths if not p.exists()]
        if missing:
            logger.error("Missing checkpoints: %s", missing)
            raise FileNotFoundError(
                f"Missing phase 2 checkpoints. Expected at: {ingredient_paths[0].parent}/"
            )

        ingredient_states = load_states(ingredient_paths)
        logger.info("  ✓ Loaded %d ingredients", len(ingredient_states))

        # ── Build config ──────────────────────────────────────────────────────
        logger.info("\n[2/7] Building Detectron2 config...")
        cfg = build_eval_cfg()
        cal_cfg = build_eval_cfg(calibration=True)
        N_COLS = min(6, cfg.MODEL.YOLOF.DECODER.NUM_CLASSES * 2)
        logger.info("  ✓ Config ready")

        # ── Build dataloaders ─────────────────────────────────────────────────
        logger.info("\n[3/7] Building dataloaders...")
        # NEW: Use EVAL_DATASET for learned soup optimization (validation-based learning)
        # This provides better generalization than training on SELECTION_DATASET
        selection_dataloader = build_eval_dataloader(
            cfg, EVAL_DATASET, batch_size=LEARNED_SOUP_BATCH_SIZE
        )
        train_dataloader = build_train_dataloader(cal_cfg, TRAIN_DATASET)

        if hasattr(selection_dataloader.dataset, "sampler"):
            selection_dataset_size = selection_dataloader.dataset.sampler._size
        else:
            selection_dataset_size = len(selection_dataloader.dataset._dataset)

        if hasattr(train_dataloader.dataset.dataset, "sampler"):
            train_dataset_size = train_dataloader.dataset.dataset.sampler._size
        else:
            train_dataset_size = len(train_dataloader.dataset._dataset)

        logger.info(
            "  ✓ Dataloaders ready — train: %d batches, selection: %d batches",
            int(train_dataset_size / train_dataloader.batch_size),
            int(selection_dataset_size / selection_dataloader.batch_size),
        )

        logger.debug("  → force_construction list: %s", force_construction)
        # ── Build or load soup conditions ─────────────────────────────────────
        checkpoints = {}
        if force_construction:
            logger.info("\n[4/7] Building soup conditions 1-%d...", len(force_construction))
            logger.debug("  → Available conditions: %s", list(MERGE_CONDITIONS.keys()))
            logger.debug("  → Local variables: %s", list(locals().keys()))
            for method in force_construction:
                if method not in MERGE_CONDITIONS:
                    logger.warning(
                        "Unknown merge condition '%s' in force_construction — skipping",
                        method,
                    )
                    continue
                build_fn, arg_keys = MERGE_CONDITIONS[method]
                args = {}
                for k in arg_keys:
                    if k not in locals():
                        logger.error(
                            "Required argument '%s' for building condition '%s' not found in locals()",
                            k, method
                        )
                        raise KeyError(
                            f"Argument '{k}' required for building condition '{method}' not found"
                        )
                    args[k] = locals()[k]
                checkpoint_name = f"condition_{list(MERGE_CONDITIONS.keys()).index(method) + 1}_state"
                checkpoints[checkpoint_name] = build_fn(**args)
                # ── Save checkpoints ──────────────────────────────────────────────────
                logger.info("\nSaving checkpoints...")
                save_checkpoint(checkpoint_dir / f"{method}_soup.pth", checkpoints[checkpoint_name])
            logger.info("  ✓ Conditions 1-5 built")
            logger.info("  ✓ All 5 conditions built")

        else:
            logger.info("\n[4/7] Loading cached soup condition checkpoints...")
            checkpoints["condition_1_state"] = load_state(checkpoint_dir / "global_uniform_soup.pth")
            checkpoints["condition_2_state"] = load_state(checkpoint_dir / "branch_uniform_soup.pth")
            checkpoints["condition_3_state"] = load_state(checkpoint_dir / "dirichlet_soup.pth")
            checkpoints["condition_4_state"] = load_state(checkpoint_dir / "fisher_weighted_soup.pth")
            checkpoints["condition_5_state"] = load_state(checkpoint_dir / "learned_soup.pth")
            checkpoints["condition_6_state"] = load_state(checkpoint_dir / "learned_tri_head_soup.pth")
            logger.info("  ✓ Loaded all 6 conditions from cached checkpoints")

        # ── BN calibration ────────────────────────────────────────────────────
        if cal_bn:
            n_cal = int(train_dataset_size / train_dataloader.batch_size)
            for label, method, state in [
                ("Global Uniform", "global_uniform", "condition_1_state"),
                ("Branch Uniform", "branch_uniform", "condition_2_state"),
                ("Dirichlet CD", "dirichlet", "condition_3_state"),
                ("Fisher-weighted", "fisher", "condition_4_state"),
                ("Learned Soup", "learned", "condition_5_state"),
            ]:
                if method not in cal_bn:
                    logger.info(f"  → Skipping BN calibration for {label} (not in cal_bn list)")
                    continue
                logger.info("  Calibrating BN: %s...", label)
                checkpoints[state] = calibrate_bn(
                    cal_cfg, checkpoints[state], train_dataloader,
                    n_batches=n_cal, device=DEVICE
                )
            logger.info("  ✓ BN calibration complete for all conditions")

        # ── Evaluate conditions ─────────────────────────────────────────
        logger.info("\n[6/7] Evaluating conditions...")
        map_results = {}
        for i, condition in enumerate(list(MERGE_CONDITIONS.keys())):
            logger.info(f"Evaluating {condition} soup")
            checkpoint_name = f"condition_{i + 1}_state"
            if checkpoint_name not in checkpoints:
                logger.info(f"  → Checkpoint for {condition} not found — skipping evaluation")
                continue

            start_eval = time.perf_counter()
            condition_name = f"condition_{i + 1}"
            map_results[f"results_cond{i+1}"] = evaluate_condition(checkpoints[checkpoint_name], cfg, condition_name)
            #  = results["map50_95"]

            logger.info("\nResults summary:")
            logger.info("  Condition %i (%s):  mAP50:95=%.4f", i + 1, condition.capitalize(), map_results[f"results_cond{i+1}"]["map50_95"])

            # Log per-class AP table for best condition
            results_flatten = list(itertools.chain(*map_results[f"results_cond{i+1}"]["per_class_ap"]))
            results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d, tablefmt="pipe", floatfmt=".3f",
                headers=["category", "AP", "AR"] * (N_COLS // 2), numalign="left",
            )
            logger.info("\nCondition %i (%s) per-class AP (sample):\n%s", i + 1, condition.capitalize(), table)
            logger.info("%s Evaluation completed in: %s", condition.capitalize(), _format_duration(time.perf_counter() - start_eval))

        # ── Determine best learned condition ──────────────────────────────────
        logger.info("\n[7/7] Determining best learned condition (M3/M4/M5/M6)...")
        for map in ["results_cond3", "results_cond4", "results_cond5", "results_cond6"]:
            if map in map_results:
                map_results["best_map"] = max(map_results.get("best_map", 0.0), map_results[map]["map50_95"])

        if map_results["best_map"] == map_results.get("results_cond6")["map50_95"]:
            map_results["best_learned_cond"] = 6
            map_results["best_learned_state"] = "condition_6_state"
            map_results["best_learned_label"] = "Learned Soup (M6)"
        elif map_results["best_map"] == map_results.get("results_cond5")["map50_95"]:
            map_results["best_learned_cond"] = 5
            map_results["best_learned_state"] = "condition_5_state"
            map_results["best_learned_label"] = "Learned Soup (M5)"
        elif map_results["best_map"] == map_results.get("results_cond4")["map50_95"]:
            map_results["best_learned_cond"] = 4
            map_results["best_learned_state"] = "condition_4_state"
            map_results["best_learned_label"] = "Fisher-weighted (M4)"
        else:
            map_results["best_learned_cond"] = 3
            map_results["best_learned_state"] = "condition_3_state"
            map_results["best_learned_label"] = "Dirichlet CD (M3)"

        logger.info(
            "  → Best learned condition: %s (mAP50:95=%.4f)",
            map_results["best_learned_label"], map_results["best_map"],
        )

        # ── Save checkpoints ──────────────────────────────────────────────────
        logger.info("\nSaving checkpoints...")
        for method in force_construction:
            if method not in MERGE_CONDITIONS:
                logger.warning(
                    "Unknown merge condition '%s' in force_construction — skipping checkpoint save",
                    method,
                )
                continue
            checkpoint_name = f"condition_{list(MERGE_CONDITIONS.keys()).index(method) + 1}_state"
            if method == "learned":
                metadata = {
                    "method": method, 
                    "map50_95": map_results.get(f"results_cond{list(MERGE_CONDITIONS.keys()).index(method) + 1}", {}).get("map50_95", 0.0),
                    "learned_mixing_substrings": list(LEARNED_MIXING_SUBSTRINGS),
                    "learned_soup_lr": LEARNED_SOUP_LR,
                    "learned_soup_epochs": LEARNED_SOUP_EPOCHS,
                }
            else:
                metadata = {
                    "method": method, 
                    "map50_95": map_results.get(f"results_cond{list(MERGE_CONDITIONS.keys()).index(method) + 1}", {}).get("map50_95", 0.0)
                }

            if checkpoint_name in checkpoints:
                save_checkpoint(
                    checkpoint_dir / f"{method}_soup.pth", checkpoints[checkpoint_name],
                    metadata=metadata
                )
                logger.info(f"  ✓ {method}_soup.pth saved → {checkpoint_dir}")
            else:
                logger.warning(f"  → Checkpoint for {method} not found — skipping save")
        
        save_checkpoint(
            checkpoint_dir / "best_learned_soup.pth", checkpoints[map_results["best_learned_state"]],
            metadata={"condition": map_results["best_learned_cond"], "method": map_results["best_learned_label"], "map50_95": map_results["best_map"]},
        )

        logger.info("  ✓ best_learned_soup.pth    → %s", checkpoint_dir)

        # ── Save results JSON ─────────────────────────────────────────────────
        if len(force_construction) == len(MERGE_CONDITIONS):
            results_summary = {
                "condition_1": map_results["results_cond1"],
                "condition_2": map_results["results_cond2"],
                "condition_3": map_results["results_cond3"],
                "condition_4": map_results["results_cond4"],
                "condition_5": map_results["results_cond5"],
                "condition_6": map_results["results_cond6"],
                "best_learned_condition": map_results["best_learned_cond"],
                "best_learned_map50_95": map_results["best_map"],
                "hyperparameters": {
                    "learned_mixing_substrings": list(LEARNED_MIXING_SUBSTRINGS),
                    "learned_soup_lr": LEARNED_SOUP_LR,
                    "learned_soup_epochs": LEARNED_SOUP_EPOCHS,
                    "learned_soup_patience": LEARNED_SOUP_PATIENCE,
                    "learned_soup_batch_size": LEARNED_SOUP_BATCH_SIZE,
                    "cd_lambda_grid": CD_LAMBDA_GRID,
                },
            }

        else:
            results_summary = {
                "best_learned_condition": map_results.get("best_learned_cond", None),
                "best_learned_map50_95": map_results.get("best_map", None),
                "hyperparameters": {
                    "learned_mixing_substrings": list(LEARNED_MIXING_SUBSTRINGS),
                    "learned_soup_lr": LEARNED_SOUP_LR,
                    "learned_soup_epochs": LEARNED_SOUP_EPOCHS,
                    "learned_soup_patience": LEARNED_SOUP_PATIENCE,
                    "learned_soup_batch_size": LEARNED_SOUP_BATCH_SIZE,
                    "cd_lambda_grid": CD_LAMBDA_GRID,
                },
            }
            for method in force_construction:
                method_num = list(MERGE_CONDITIONS.keys()).index(method) + 1
                if f"results_cond{method_num}" in map_results:
                    logger.info(f"  → Condition {method_num} ({method.capitalize()}) results not saved to JSON (force_construction incomplete)")
                    results_summary[f"condition_{method_num}"] = map_results[f"results_cond{method_num}"]
                else:
                    logger.warning(f"  → Condition {method_num} ({method.capitalize()}) results not found in locals() — skipping JSON save for this condition")

        results_path = Path(RESULTS_DIR) / f"phase3_soup_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2, default=str)
        logger.info("  ✓ Results saved → %s", results_path)
        
        logger.info("\n" + "=" * 90)
        logger.info("PHASE 3 COMPLETE")
        logger.info("=" * 90)

        total_elapsed = time.perf_counter() - start_time
        logger.info("Total elapsed time: %s", _format_duration(total_elapsed))
        return results_summary

    except Exception:
        if logger:
            logger.exception("Phase 3 failed.")

        total_elapsed = time.perf_counter() - start_time
        logger.info("Total elapsed time: %s", _format_duration(total_elapsed))
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

parsed_args = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Soup Construction & Evaluation")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--cal-bn", nargs='+', choices=list(MERGE_CONDITIONS.keys()), default=[],
                        help="Perform BN calibration for specified methods")
    parser.add_argument("--force-construction", nargs='+', choices=list(MERGE_CONDITIONS.keys()), default=None,
                        help="Force rebuild of all conditions (ignore cached checkpoints)")
    parsed_args = parser.parse_args()

    # run(
    #     verbose=parsed_args.verbose,
    #     cal_bn=parsed_args.cal_bn,
    #     force_construction=parsed_args.force_construction,
    # )
    _register_datasets()
    run(
        verbose=True,
        cal_bn=list(MERGE_CONDITIONS.keys()),  # Calibrate BN for all conditions
        force_construction=list(MERGE_CONDITIONS.keys()),  # Force rebuild of all conditions (ignore cached checkpoints)
    )