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

import os
import json
import logging
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

from tabulate import tabulate

from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog

from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    DEVICE,
    TRAIN_DATASET,
    RESULTS_DIR,
    SELECTION_DATASET,
    PHASE2_OUTPUT_DIR,
    PRETRAINED_WEIGHTS,
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
CD_LAMBDA_GRID: List[float] = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

#: Whether to use Hessian (expensive) or L2 proxy for Fisher weights
USE_HESSIAN_FOR_FISHER: bool = False

#: Learned Soup hyperparameters (Condition 5)
LEARNED_SOUP_LR: float = 0.1
LEARNED_SOUP_EPOCHS: int = 5
LEARNED_SOUP_PATIENCE: int = 2
LEARNED_SOUP_BATCH_SIZE: int = 8

CORE_GROUPS = [
    list(range(0, 4)),    # Model 0 → cores 0–3
    list(range(4, 8)),    # Model 1 → cores 4–7
    list(range(8, 12)),   # Model 2 → cores 8–11
    list(range(12, 16)),  # Model 3 → cores 12–15
    list(range(16, 20)),  # Model 4 → cores 16–19
    list(range(20, 24)),  # Model 5 → cores 20–23
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

def build_dirichlet_cd(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    # selection_dataloader,
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
        anchor_cls, anchor_reg, anchor_shared,
        cls_taus, reg_taus, shared_taus,
        be_dict, cfg, 
        "cls"
    )
    best_lam_reg = _coordinate_descent_search(
        anchor_cls, anchor_reg, anchor_shared,
        cls_taus, reg_taus, shared_taus,
        be_dict, cfg, 
        "reg"
    )
    
    logger.info("  → CD search found: λ_cls=%.3f, λ_reg=%.3f", best_lam_cls, best_lam_reg)
    
    # Build final soup with optimal λ values
    cls_merged = apply_uniform_lambdas(anchor_cls, cls_taus, best_lam_cls)
    reg_merged = apply_uniform_lambdas(anchor_reg, reg_taus, best_lam_reg)
    shared_merged = apply_uniform_lambdas(anchor_shared, shared_taus, 1.0)  # shared always λ=1.0
    
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

    model = None  # Ensure model is always defined before try block

    try:
        # Build merged state with all three branches
        logger.info("  [CD search %s] Evaluating λ=%.3f...", search_branch, lam)
        if search_branch == "cls":
            cls_merged = apply_uniform_lambdas(anchor_cls, taus_cls, lam)
            reg_merged = apply_uniform_lambdas(anchor_reg, taus_reg, 1.0)
            shared_merged = apply_uniform_lambdas(anchor_shared, taus_shared, 1.0)
        elif search_branch == "reg":
            cls_merged = apply_uniform_lambdas(anchor_cls, taus_cls, 1.0)
            reg_merged = apply_uniform_lambdas(anchor_reg, taus_reg, lam)
            shared_merged = apply_uniform_lambdas(anchor_shared, taus_shared, 1.0)
        else:
            raise ValueError(f"Unknown branch: {search_branch}")

        full_state = merge_subdicts(be_dict, cls_merged, reg_merged, shared_merged)

        model = EvaluateModel(cfg, state_dict=full_state)
        # assign_state_to_model(model, full_state)
        model = model.to(DEVICE)
        model.eval()

        map_val = get_map(
            model.model,
            cfg,
            SELECTION_DATASET,
            output_dir=Path(RESULTS_DIR) / "cd_search",
            tag=f"{search_branch}_lam{lam:.2f}"
        )
        logger.debug(
            "  [CD search %s] λ=%.3f → mAP=%.4f",
            search_branch, lam, map_val
        )
        return map_val, lam

    except ValueError:
        # Re-raise configuration errors immediately — these are not recoverable
        raise

    except Exception as e:
        logger.error(
            "  [CD search %s] λ=%.3f → evaluation failed: %s",
            search_branch, lam, str(e)
        )
        return np.nan, lam

    finally:
        # Always clean up model and GPU memory, regardless of success or failure
        if model is not None:
            del model
        torch.cuda.empty_cache()

        # Restore default thread count so the worker is clean for any future tasks
        torch.set_num_threads(torch.get_num_threads.__doc__ and 1 or 1)  # No-op: keep at 1
        # Reset CPU affinity to all available cores for this worker process
        try:
            all_cores = set(range(os.cpu_count()))
            os.sched_setaffinity(0, all_cores)
        except OSError as affinity_err:
            logger.error(
                "  [CD search %s] Could not reset CPU affinity: %s",
                search_branch, affinity_err
            )


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
    search_branch: str,  # "cls" or "reg"
) -> float:
    """
    Find best λ via coordinate descent.
    Varies λ for one branch while keeping other branches at λ=1.0
    """
    import torch.multiprocessing as mp

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
            logger.error(
                "Job %d failed with error: %s: %s",
                idx, type(exc).__name__, exc,
                exc_info=exc
            )
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
                (
                    search_branch,
                    lam,
                    anchor_cls,
                    taus_cls,
                    anchor_reg,
                    taus_reg,
                    anchor_shared,
                    taus_shared,
                    be_dict,
                    cfg
                ),
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
        else:
            if map_val > best_map:
                best_map = map_val
                best_lam = lam

    if nan_count == len(CD_LAMBDA_GRID):
        logger.warning(
            "  [CD search %s] All λ values returned NaN. Defaulting to λ=1.0",
            search_branch
        )
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
# Condition 5 (M5): Learned Soup (Optimized α and β)
# Option B: Scalar-detach gradient strategy
#
# Key insight: α and β only multiply SCALAR loss values — they do not need
# gradients through the full model forward pass. We can therefore:
#   1. Run each model under torch.no_grad() → zero activation memory retained
#   2. Detach loss scalars from the model graph
#   3. Reconstruct a tiny graph: scaled_loss = β · Σᵢ αᵢ · loss_i_scalar
#      where only α and β are leaf tensors → .backward() is near-free
#
# This keeps peak GPU memory at exactly 1 model at a time.
#
# Mixing strategy:
#   • cls_subnet / reg_subnet (decoder heads) → learned α + β (optimised)
#   • All other parameters (backbone, neck, …) → uniform averaging (1/k each)
# ─────────────────────────────────────────────────────────────────────────────

# Keys whose names contain any of these substrings are treated as decoder
# parameters and mixed with the learned α. Everything else is uniformly averaged.
LEARNED_MIXING_SUBSTRINGS = ("cls_subnet", "reg_subnet")


def _is_decoder_key(key: str) -> bool:
    """Return True if *key* belongs to the decoder (cls/reg subnet)."""
    return any(sub in key for sub in LEARNED_MIXING_SUBSTRINGS)


def build_learned_soup(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataloader,
) -> Dict[str, torch.Tensor]:
    """
    M5 (Condition 5): Learned Soup — Scalar-detach gradient strategy (Option B).

    Gradient strategy
    -----------------
    α and β multiply SCALAR loss values, not full tensors. This means their
    gradients do not require backpropagation through the model internals.

    The optimisation loop therefore:
        1. Runs each model_i under torch.no_grad() — NO activations retained.
        2. Calls .item() to extract each loss as a plain Python float.
        3. Moves model_i back to CPU and frees GPU memory before model_i+1.
        4. Reconstructs a minimal graph after all k models have run:
               scaled_loss = β · Σᵢ αᵢ · loss_scalar_i
           where α and β are the only leaf tensors.
        5. Calls scaled_loss.backward() — graph is tiny (k scalars), near-free.

    This keeps peak GPU usage at 1 model + 1 batch at any time.

    Mixing strategy
    ---------------
    Decoder parameters  (cls_subnet, reg_subnet):
        θ_decoder = Σᵢ αᵢ θᵢ      (αᵢ optimised, ≥ 0, Σαᵢ = 1)

    All other parameters (backbone, neck, FPN, …):
        θ_other = (1/k) Σᵢ θᵢ     (uniform, fixed)

    Optimisation objective:
        arg min_{α, β}  β · Σᵢ αᵢ · L̄ᵢ
        subject to:     αᵢ ≥ 0,  Σαᵢ = 1  (softmax)
                        β > 0             (exp parameterisation)
    where L̄ᵢ is the mean batch loss of model i (scalar, detached).

    Inference cost remains O(1): a single merged model is produced after
    optimisation.

    Args:
        ingredient_states:    List of k fine-tuned model state dicts.
        cfg:                  Detectron2 / CfgNode model configuration.
        selection_dataloader: Held-out validation dataloader for optimisation.

    Returns:
        Merged state dict — decoder mixed with α, rest uniformly averaged.
    """
    logger.info("Building Condition 5 (M5): Learned Soup — scalar-detach gradient strategy")
    logger.info(
        "  → Decoder keys (%s) use learned α; all other keys use uniform mixing.",
        LEARNED_MIXING_SUBSTRINGS,
    )

    eval_log_freq = 100

    n_ingredients = len(ingredient_states)
    device = DEVICE

    # ── Learnable parameters ──────────────────────────────────────────────────
    # alpha_raw: unconstrained logits; softmax → simplex (uniform at init)
    alpha_raw = torch.zeros(
        n_ingredients, device=device, dtype=torch.float32, requires_grad=True
    )

    # log_beta: unconstrained; β = exp(log_beta) → always positive (β₀ = 1.0)
    log_beta = torch.tensor(
        0.0, device=device, dtype=torch.float32, requires_grad=True
    )

    # set_to_none=True frees gradient buffers completely after each step
    optimizer = torch.optim.AdamW(
        [alpha_raw, log_beta], lr=LEARNED_SOUP_LR
    )

    best_loss    = float("inf")
    best_alpha_normalized: Optional[torch.Tensor] = None
    best_log_beta: Optional[torch.Tensor]         = None
    patience_counter = 0

    logger.info(
        "  → Initial α (softmax of zeros): %s",
        [f"{a:.4f}" for a in torch.softmax(alpha_raw.detach(), dim=0).tolist()],
    )
    logger.info("  → Initial β (exp(log_β)): %.4f", torch.exp(log_beta).item())

    try:
        for epoch in range(LEARNED_SOUP_EPOCHS):
            epoch_loss   = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(selection_dataloader):

                # ── Step 1: collect per-model loss SCALARS (no graph retained) ─
                # Each model is loaded to GPU, run under no_grad, then moved
                # back to CPU before the next model is loaded.
                # Peak GPU = 1 model + 1 batch at any time.
                per_model_loss_scalars: List[float] = []

                for i, state in enumerate(ingredient_states):
                    # Build model on CPU first, then move to GPU
                    model_i = EvaluateModel(cfg, state_dict=state)
                    model_i.to(device)
                    model_i.eval()

                    # Disable BN running-stat updates — not needed for loss proxy
                    for mod in model_i.model.modules():
                        if isinstance(mod, (torch.nn.BatchNorm1d,
                                            torch.nn.BatchNorm2d,
                                            torch.nn.BatchNorm3d)):
                            mod.eval()

                    try:
                        # No graph retained — activations freed immediately
                        with torch.no_grad():
                            outputs_i = model_i.model(batch)

                        raw_loss_i = _compute_raw_loss(outputs_i, device)

                        # .item() extracts the scalar and fully severs the
                        # tensor from any remaining graph references
                        per_model_loss_scalars.append(raw_loss_i.item())

                        del outputs_i, raw_loss_i

                    except Exception as e:
                        logger.error(
                            "  Model %d, batch %d forward error: %s",
                            i, batch_idx, str(e),
                        )
                        # Contribute 0.0 — α gradient still flows from other models
                        raise

                    finally:
                        # Move model back to CPU and free GPU memory
                        # before loading the next model
                        model_i.to("cpu")
                        del model_i
                        torch.cuda.empty_cache()

                # ── Step 2: reconstruct a minimal graph over α and β only ──────
                # All loss values are plain Python floats (scalars) at this point.
                # The only tensors in this graph are alpha_raw and log_beta.
                # .backward() on this graph is near-instantaneous.
                optimizer.zero_grad(set_to_none=True)

                alpha_normalized = torch.softmax(alpha_raw, dim=0)   # (k,)
                beta              = torch.exp(log_beta)               # scalar > 0

                # Convert loss scalars to a device tensor (no grad needed)
                loss_tensor = torch.tensor(
                    per_model_loss_scalars, device=device, dtype=torch.float32
                )  # shape: (k,) — detached, no autograd history

                # Weighted sum: Σᵢ αᵢ · L̄ᵢ  — α IS in the graph
                combined_loss = (alpha_normalized * loss_tensor).sum()

                # Temperature scaling: β · combined_loss — β IS in the graph
                scaled_loss = beta * combined_loss

                if not scaled_loss.isfinite():
                    logger.warning(
                        "  Non-finite loss at epoch %d batch %d — skipping",
                        epoch, batch_idx,
                    )
                    del scaled_loss, combined_loss, loss_tensor
                    continue

                # Tiny backward — only through α and β leaf tensors
                scaled_loss.backward()
                optimizer.step()

                epoch_loss   += scaled_loss.item()
                epoch_batches += 1
                
                if batch_idx > 0 and batch_idx % eval_log_freq == 0:
                    logger.debug(
                        "  Epoch %d batch %d: batch_loss=%.4f, avg_loss=%.4f, α=%s, β=%.4f",
                        epoch,
                        batch_idx,
                        scaled_loss.item(),
                        epoch_loss / max(epoch_batches, 1),
                        [f"{a:.4f}" for a in alpha_normalized.detach().tolist()],
                        beta.item(),
                    )
                
                del scaled_loss, combined_loss, loss_tensor

                if epoch_loss / max(epoch_batches, 1) > 100.0:
                    logger.warning(
                        "  Loss diverging at epoch %d, stopping optimisation", epoch
                    )
                    break

            avg_loss = epoch_loss / max(epoch_batches, 1)

            # ── Early stopping ────────────────────────────────────────────────
            if avg_loss < best_loss:
                best_loss             = avg_loss
                best_alpha_normalized = torch.softmax(alpha_raw, dim=0).detach().clone()
                best_log_beta         = log_beta.detach().clone()
                patience_counter      = 0
            else:
                patience_counter += 1

            if patience_counter >= LEARNED_SOUP_PATIENCE:
                logger.info(
                    "  → Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, LEARNED_SOUP_PATIENCE,
                )
                break

            logger.debug(
                "  Epoch %d: avg_loss=%.6f, α=%s, β=%.4f",
                epoch,
                avg_loss,
                [f"{a:.4f}" for a in torch.softmax(alpha_raw, dim=0).detach().tolist()],
                torch.exp(log_beta).item(),
            )

        # ── Fallback guards ───────────────────────────────────────────────────
        if best_alpha_normalized is None:
            best_alpha_normalized = torch.softmax(alpha_raw, dim=0).detach()
        if best_log_beta is None:
            best_log_beta = log_beta.detach()

        best_beta = torch.exp(best_log_beta).item()

        logger.info("  → Optimization complete!")
        logger.info(
            "  → Final α: %s", [f"{a:.4f}" for a in best_alpha_normalized.tolist()]
        )
        logger.info("  → Final β: %.4f", best_beta)

    except Exception as e:
        logger.error(
            "  ✗ Learned soup optimization failed: %s", str(e), exc_info=True
        )
        raise RuntimeError("Learned soup optimization failed") from e

    finally:
        torch.cuda.empty_cache()
        logger.info(
            "  → GPU mem after optimisation: %.1f MB allocated / %.1f MB reserved",
            torch.cuda.memory_allocated(device) / 1e6,
            torch.cuda.memory_reserved(device) / 1e6,
        )

    # ── Build final merged state dict ─────────────────────────────────────────
    # Decoder keys  → learned α
    # All other keys → uniform 1/k
    final_mixed_state = _mix_states_selective(
        ingredient_states, best_alpha_normalized
    )

    logger.info(
        "  ✓ Learned soup built. Size: %.2f MB",
        _state_dict_size_mb(final_mixed_state),
    )
    return final_mixed_state


# ─────────────────────────────────────────────────────────────────────────────
# Selective mixing:
#   decoder keys  → learned α  (cls_subnet, reg_subnet)
#   all other keys → uniform 1/k  (backbone, neck, FPN, …)
# ─────────────────────────────────────────────────────────────────────────────

def _mix_states_selective(
    ingredient_states: List[Dict[str, torch.Tensor]],
    alpha: torch.Tensor,  # normalised (sum=1), detached
) -> Dict[str, torch.Tensor]:
    """
    Produce a merged state dict using two mixing strategies:

    Decoder keys  (cls_subnet, reg_subnet):
        θ_key = Σᵢ αᵢ θᵢ          ← learned, per-ingredient weights

    All other keys (backbone, neck, FPN, …):
        θ_key = (1/k) Σᵢ θᵢ       ← uniform average

    Args:
        ingredient_states: List of k state dicts.
        alpha:             Optimised, normalised weight vector. shape: (k,).
                           Must be detached before calling.

    Returns:
        Merged state dict in the original dtype of each parameter.
    """
    n              = len(ingredient_states)
    uniform_weight = 1.0 / n
    mixed_state    = {}
    decoder_keys   = []
    uniform_keys   = []

    for key in ingredient_states[0].keys():
        use_learned = _is_decoder_key(key)
        if use_learned:
            decoder_keys.append(key)
        else:
            uniform_keys.append(key)

        mixed = None
        for i, state in enumerate(ingredient_states):
            if key not in state:
                continue
            param  = state[key].float()
            weight = alpha[i].item() if use_learned else uniform_weight
            mixed  = weight * param if mixed is None else mixed + weight * param

        if mixed is not None:
            mixed_state[key] = mixed.to(ingredient_states[0][key].dtype)
        else:
            mixed_state[key] = ingredient_states[0][key].clone()

    logger.info(
        "  → Selective mix: %d decoder keys (learned α), %d other keys (uniform 1/k)",
        len(decoder_keys), len(uniform_keys),
    )
    logger.debug("  → Decoder keys mixed with α: %s", decoder_keys)
    return mixed_state


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
                    ...               (all loss keys are summed)
                }
            },
            ...  (one dict per image in the batch)
        ]

    Format B (fallback — tuple/list per image):
        outputs = [
            (predictions, loss_cls_tensor, loss_box_reg_tensor, ...),
            ...
        ]

    Args:
        outputs: Model forward output (list of per-image results).
        device:  Target device for the zero accumulator tensor.

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

    return batch_loss

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
    model = EvaluateModel(cfg, state_dict)

    # assign_state_to_model(model, state_dict)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Run evaluation
    try:
        from yolof_soup.utils.eval_utils import compute_coco_map
        results_dict = compute_coco_map(model, cfg, SELECTION_DATASET, output_dir=Path(RESULTS_DIR) / "phase3_eval", tag=tag)
        
        map_val = float(results_dict.get("AP", 0.0))
        map50_val = float(results_dict.get("AP50", 0.0))
        ar100_val = float(results_dict.get("AR-maxDets=100", 0.0))
        
        # Extract per-class AP values
        per_class_ap = extract_per_class_ap(results_dict, MetadataCatalog.get(SELECTION_DATASET).thing_classes)
        
        logger.info("  ✓ Condition %s: mAP50:95=%.4f, mAP50=%.4f, AR@100=%.4f", 
                   tag, map_val, map50_val, ar100_val)
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


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True, cal_bn: bool = True, force_construction: bool = False) -> Dict[str, Any]:
    """
    Main Phase 3 entry point.
    
    Args:
        verbose: Whether to log progress
        calibrate_bn: Whether to calibrate BN layer stats using the training split 
                      before evaluation (can improve performance for merged models)
        force_construction: Whether to force construction of all soup conditions even 
                            if cached checkpoints exist (from previous runs). 
                            If False, will load existing checkpoints if found.
    
    Returns:
        Dict with results for all 4 conditions + metadata
    """
    
    try:
        global logger
        logger = get_logger(level=logging.DEBUG if verbose else logging.INFO, add_file_handler=True)
        
        logger.info("=" * 90)
        logger.info("PHASE 3: SOUP CONSTRUCTION & EVALUATION")
        logger.info("=" * 90)
        
        # Setup directories
        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ingredients_dir = Path(PHASE2_OUTPUT_DIR)
        
        # Load ingredient checkpoints
        logger.info("\n[1/7] Loading 6 ingredient checkpoints...")
        run_registry = get_run_specs()
        ingredient_runs = [r for r in run_registry if r.role == "ingredient"]

        ingredient_paths = []
        # run_ids = []
        cfg_paths = []
        for run_spec in ingredient_runs:
            ckpt_path = Path(ingredients_dir) / f"{run_spec.run_name}/model_best.pth"
            ingredient_paths.append(ckpt_path)
            # run_ids.append(run_spec.run_id)
            cfg_paths.append(Path(ingredients_dir) / f"{run_spec.run_name}/config.yaml")
            # logger.info("Will audit: %s → %s", run_spec.run_id, ckpt_path)
        
        # Check if paths exist
        missing = [p for p in ingredient_paths if not p.exists()]
        if missing:
            logger.error("Missing checkpoints: %s", missing, exc_info=True)
            raise FileNotFoundError(f"Missing phase 2 checkpoints. Expected at: {ingredient_paths[0].parent}/")
        
        ingredient_states = load_states(ingredient_paths)
        logger.info("  ✓ Loaded 6 ingredients")
        
        # Build Detectron2 config
        logger.info("\n[2/7] Building Detectron2 config...")
        # cfgs = [build_eval_cfg(EVAL_DATASET, str(cfg_path), ckpt_file) for cfg_path, ckpt_file in zip(cfg_paths, ingredient_paths)]
        cfg = build_eval_cfg()
        cal_cfg = build_eval_cfg(calibration=True)
        N_COLS = min(6, cfg.MODEL.YOLOF.DECODER.NUM_CLASSES * 2)
        logger.info("  ✓ Config ready")
        
        # Build dataloaders
        logger.info("\n[3/7] Building dataloaders...")
        selection_dataloader = build_eval_dataloader(cfg, SELECTION_DATASET, batch_size=LEARNED_SOUP_BATCH_SIZE)
        train_dataloader = build_train_dataloader(cal_cfg, TRAIN_DATASET)

        if hasattr(selection_dataloader.dataset, 'sampler'):
            selection_dataset_size = selection_dataloader.dataset.sampler._size  
        else:
            selection_dataset_size = len(selection_dataloader.dataset._dataset)

        if hasattr(train_dataloader.dataset.dataset, 'sampler'):
            train_dataset_size = train_dataloader.dataset.dataset.sampler._size 
        else:
            train_dataset_size = len(train_dataloader.dataset._dataset)

        logger.info(f"  ✓ Dataloaders ready: Train loader with {int(train_dataset_size / train_dataloader.batch_size)} batches "
                    f"and Selection loader with {int(selection_dataset_size / selection_dataloader.batch_size)} batches")
        
        if force_construction:
            # Build all 5 conditions
            logger.info("\n[4/8] Building soup conditions...")
            condition_1_state = build_global_uniform_souped_model(ingredient_states)
            condition_2_state = build_branch_uniform(ingredient_states)
            condition_3_state = build_dirichlet_cd(ingredient_states, cfg)
            condition_4_state = build_fisher_weighted(ingredient_states)
            logger.info("  ✓ Conditions 1-4 built")
            
            logger.info("\n[5/8] Building Condition 5 (M5): Learned Soup...")
            condition_5_state = build_learned_soup(ingredient_states, cfg, selection_dataloader)
            logger.info("  ✓ All 5 conditions built")
        else:
            branch_uniform_path = checkpoint_dir / "branch_uniform_soup.pth"
            best_learned_path = checkpoint_dir / "best_learned_soup.pth"
            global_uniform_path = checkpoint_dir / "global_uniform_soup.pth"
            dirichlet_path = checkpoint_dir / "dirichlet_soup.pth"
            fisher_weighted_path = checkpoint_dir / "fisher_weighted_soup.pth"
            learned_soup_path = checkpoint_dir / "learned_soup.pth"

            logger.info("\n[4/8] Loading soup condition checkpoints...")
            condition_1_state = load_state(global_uniform_path)
            condition_2_state = load_state(branch_uniform_path)
            condition_3_state = load_state(dirichlet_path)
            condition_4_state = load_state(fisher_weighted_path)
            condition_5_state = load_state(learned_soup_path)
            logger.info("  ✓ Loaded all 5 conditions from checkpoints")

        if cal_bn:
            logger.info("\n[6/8] Calibrating BN layer stats...")
            logger.info("\n[6a/8] Calibrating BN layer stats of Global Uniform soup...")
            condition_1_state = calibrate_bn(cal_cfg, condition_1_state, train_dataloader, n_batches=int(train_dataset_size / train_dataloader.batch_size), device=DEVICE)
            logger.info("\n[6b/8] Calibrating BN layer stats of Branch Uniform soup...")
            condition_2_state = calibrate_bn(cal_cfg, condition_2_state, train_dataloader, n_batches=int(train_dataset_size / train_dataloader.batch_size), device=DEVICE)
            logger.info("\n[6c/8] Calibrating BN layer stats of Dirichlet soup...")
            condition_3_state = calibrate_bn(cal_cfg, condition_3_state, train_dataloader, n_batches=int(train_dataset_size / train_dataloader.batch_size), device=DEVICE)
            logger.info("\n[6d/8] Calibrating BN layer stats of Fisher-weighted soup...")
            condition_4_state = calibrate_bn(cal_cfg, condition_4_state, train_dataloader, n_batches=int(train_dataset_size / train_dataloader.batch_size), device=DEVICE)
            logger.info("\n[6e/8] Calibrating BN layer stats of Learned soup...")
            condition_5_state = calibrate_bn(cal_cfg, condition_5_state, train_dataloader, n_batches=int(train_dataset_size / train_dataloader.batch_size), device=DEVICE)
            logger.info("  ✓ BN calibration complete for all conditions")

        # Evaluate all 5 conditions
        logger.info("\n[7/8] Evaluating all 5 conditions on eval split...")
        results_cond1 = evaluate_condition(condition_1_state, cfg, "condition_1")
        results_flatten = list(itertools.chain(*results_cond1["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("\nCondition 1 (Global Uniform) evaluation results:")
        logger.info("  → table sample:\n%s", table)
        logger.info(f"  → Precision, Recall\nmAP: {results_cond1['map50_95']}, mAP50: {results_cond1['map50']}, AR100: {results_cond1['ar100']}")

        results_cond2 = evaluate_condition(condition_2_state, cfg, "condition_2")
        results_flatten = list(itertools.chain(*results_cond2["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("\nCondition 2 (Branch Uniform) evaluation results:")
        logger.info("  → table sample:\n%s", table)
        logger.info(f"  → Precision, Recall\nmAP: {results_cond2['map50_95']}, mAP50: {results_cond2['map50']}, AR100: {results_cond2['ar100']}")
        
        results_cond3 = evaluate_condition(condition_3_state, cfg, "condition_3")
        results_flatten = list(itertools.chain(*results_cond3["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("\nCondition 3 (Dirichlet) evaluation results:")
        logger.info("  → table sample:\n%s", table)
        logger.info(f"  → Precision, Recall\nmAP: {results_cond3['map50_95']}, mAP50: {results_cond3['map50']}, AR100: {results_cond3['ar100']}")

        results_cond4 = evaluate_condition(condition_4_state, cfg, "condition_4")
        results_flatten = list(itertools.chain(*results_cond4["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("\nCondition 4 (Fisher Weighted) evaluation results:")
        logger.info("  → table sample:\n%s", table)
        logger.info(f"  → Precision, Recall\nmAP: {results_cond4['map50_95']}, mAP50: {results_cond4['map50']}, AR100: {results_cond4['ar100']}")

        results_cond5 = evaluate_condition(condition_5_state, cfg, "condition_5")
        results_flatten = list(itertools.chain(*results_cond5["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("\nCondition 5 (Learned Soup) evaluation results:")
        logger.info("  → table sample:\n%s", table)
        logger.info(f"  → Precision, Recall\nmAP: {results_cond5['map50_95']}, mAP50: {results_cond5['map50']}, AR100: {results_cond5['ar100']}")

        # Also evaluate best individual (ingredient 0 as reference)
        results_best_individual = evaluate_condition(load_state(PRETRAINED_WEIGHTS), build_eval_cfg(cfg_file="/home/nisalperera/YOLOF/configs/yolof_R_50_DC5_1x.yaml"), "best_individual")
        results_flatten = list(itertools.chain(*results_best_individual["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("\nBest individual model evaluation results:")
        logger.info("  → table sample:\n%s", table)
        logger.info(f"  → Precision, Recall\nmAP: {results_best_individual['map50_95']}, mAP50: {results_best_individual['map50']}, AR100: {results_best_individual['ar100']}")
        logger.info("  ✓ All evaluations complete")
        
        # Determine best learned condition (M3, M4, or M5)
        map_3 = results_cond3["map50_95"]
        map_4 = results_cond4["map50_95"]
        map_5 = results_cond5["map50_95"]
        best_map_learned = max(map_3, map_4, map_5)
        if best_map_learned == map_5:
            best_learned_state = condition_5_state
            best_learned_tag = "condition_5"
        elif best_map_learned == map_3:
            best_learned_state = condition_3_state
            best_learned_tag = "condition_3"
        else:
            best_learned_state = condition_4_state
            best_learned_tag = "condition_4"
        
        logger.info("\nBest learned condition:")
        logger.info("  → Condition 3 (Dirichlet): mAP50:95=%.4f", map_3)
        logger.info("  → Condition 4 (Fisher):    mAP50:95=%.4f", map_4)
        logger.info("  → Condition 5 (Learned):   mAP50:95=%.4f", map_5)
        logger.info("  → Best: %s (mAP50:95=%.4f)", best_learned_tag, best_map_learned)

        # Save checkpoints
        logger.info("\n[8/8] Saving checkpoints...")
        branch_uniform_path = checkpoint_dir / "branch_uniform_soup.pth"
        best_learned_path = checkpoint_dir / "best_learned_soup.pth"
        global_uniform_path = checkpoint_dir / "global_uniform_soup.pth"
        dirichlet_path = checkpoint_dir / "dirichlet_soup.pth"
        fisher_weighted_path = checkpoint_dir / "fisher_weighted_soup.pth"
        learned_soup_path = checkpoint_dir / "learned_soup.pth"

        save_checkpoint(
            branch_uniform_path,
            condition_2_state,
            metadata={"condition": 2, "method": "branch_uniform", "map50_95": results_cond2["map50_95"]},
        )
        save_checkpoint(
            best_learned_path,
            best_learned_state,
            metadata={"condition": int(best_learned_tag.split("_")[1]), "method": best_learned_tag, "map50_95": best_map_learned},
        )
        save_checkpoint(
            global_uniform_path,
            condition_1_state,
            metadata={"condition": 1, "method": "global_uniform", "map50_95": results_cond1["map50_95"]},
        )
        save_checkpoint(
            dirichlet_path,
            condition_3_state,
            metadata={"condition": 3, "method": "dirichlet", "map50_95": results_cond3["map50_95"]},
        )
        save_checkpoint(
            fisher_weighted_path,
            condition_4_state,
            metadata={"condition": 4, "method": "fisher_weighted", "map50_95": results_cond4["map50_95"]},
        )
        save_checkpoint(
            learned_soup_path,
            condition_5_state,
            metadata={"condition": 5, "method": "learned_soup", "map50_95": results_cond5["map50_95"]},
        )
        logger.info("  ✓ Checkpoints saved")
        
        # Compile and save results JSON
        logger.info("\nSaving results JSON...")
        results = {
            "condition_1": results_cond1,
            "condition_2": results_cond2,
            "condition_3": results_cond3,
            "condition_4": results_cond4,
            "condition_5": results_cond5,
            "best_individual": results_best_individual,
            "best_learned_condition": best_learned_tag,
            "metadata": {
                "n_ingredients": 6,
                "cd_lambda_grid": CD_LAMBDA_GRID,
                "learned_soup_lr": LEARNED_SOUP_LR,
                "learned_soup_epochs": LEARNED_SOUP_EPOCHS,
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
        logger.info("  • Dirichlet soup:        %s", dirichlet_path)
        logger.info("  • Fisher-weighted soup:  %s", fisher_weighted_path)
        logger.info("  • Learned soup:          %s", learned_soup_path)
        logger.info("  • Best learned soup:     %s", best_learned_path)
        logger.info("  • Results JSON:          %s", results_json_path)
        logger.info("=" * 90)
        
        return results
    
    except Exception as e:
        logger.error(f"Processing has terminated. {str(e)}", exc_info=True)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description="Phase 3: Soup Construction & Evaluation")
    args.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args.add_argument("--calibrate-bn", action="store_true", help="Calibrate BN layer stats before evaluation")
    args.add_argument("--force-construction", action="store_true", help="Force construction of all soup conditions even if cached checkpoints exist")
    parsed_args = args.parse_args()

    # run(verbose=parsed_args.verbose, cal_bn=parsed_args.calibrate_bn, force_construction=parsed_args.force_construction)
    run(verbose=True, cal_bn=True, force_construction=True)
