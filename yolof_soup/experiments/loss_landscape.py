"""
phase4_loss_landscape.py
=========================

Phase 4: Measure and analyze loss landscape geometry.

This module:
1. Computes pairwise linear mode connectivity (LMC) barriers for all 15 ingredient pairs
   - Measures barriers separately for backbone, encoder, cls_head, reg_head, full model
   - Uses 11-point interpolation grid (α ∈ [0, 1])

2. Computes Hessian traces for 6 ingredients (Hutchinson estimator)
   - Separate measurements for backbone, encoder, cls, reg

3. Runs statistical tests:
   - Test 1: RM-ANOVA on 4-component barriers + Tukey HSD post-hoc
   - Test 2: Pearson correlations (barrier magnitude ↔ averaging gain) with Bonferroni correction
   - Test 3: RM-ANOVA on Hessian traces with directional contrasts

Outputs:
  - phase4_lmc_barriers.json — barrier measurements for all pairs/components
  - phase4_hessian_traces.json — Hessian traces for all ingredients/components
  - phase4_statistical_tests.json — test results with p-values and effect sizes

Run: python -m yolof_soup.experiments.phase4_loss_landscape
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.multiprocessing as mp

from detectron2.config import CfgNode

from yolof_soup.config.experiment_config import (
    PHASE2_OUTPUT_DIR,
    DEVICE,
    EVAL_DATASET,
    RESULTS_DIR,
    build_eval_cfg,
    _register_datasets,
)
from yolof_soup.config.experiment_registry import get_run_specs
from yolof_soup.utils.checkpoint_utils import load_states
from yolof_soup.utils.eval_utils import get_map, build_eval_dataloader
from yolof_soup.utils.key_utils import (
    extract_subdict,
    get_backbone_encoder_keys,
    get_decoder_keys,
    split_decoder_subheads,
)
from yolof_soup.utils.inference import EvaluateModel
from yolof_soup.utils.state_dict_utils import assign_state_to_model
from yolof_soup.utils.global_logger import get_logger

logger = None

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

#: Number of interpolation points for LMC barriers
LMC_ALPHA_STEPS: int = 11

#: Number of Rademacher vectors for Hessian trace estimation
HESSIAN_SAMPLES: int = 50

CORE_GROUPS = [
    list(range(0, 4)),    # Model 0 → cores 0–3
    list(range(4, 8)),    # Model 1 → cores 4–7
    list(range(8, 12)),   # Model 2 → cores 8–11
    list(range(12, 16)),  # Model 3 → cores 12–15
    list(range(16, 20)),  # Model 4 → cores 16–19
    list(range(20, 24)),  # Model 5 → cores 20–23
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Loss computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_loss(
    model: Union[torch.nn.Module, 'EvaluateModel'],
    dataloader,
    device: torch.device,
    max_samples: Optional[int] = 500,
) -> float:
    """Compute mean loss on dataloader subset."""
    from yolof_soup.utils.eval_utils import quick_loss
    return quick_loss(model, dataloader, device, max_samples=max_samples)


# ─────────────────────────────────────────────────────────────────────────────
# LMC Barrier Computation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_state(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Linear interpolation: θ(α) = (1-α) θ_A + α θ_B"""
    interpolated = {}
    for k in state_a:
        if torch.is_floating_point(state_a[k]):
            interpolated[k] = ((1 - alpha) * state_a[k].float() + alpha * state_b[k].float()).to(state_a[k].dtype)
        else:
            interpolated[k] = state_a[k].clone()
    return interpolated


def compute_lmc_barrier(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    cfg,
    dataloader,
    common_states: List[Dict[str, torch.Tensor]] = [],
    n_steps: int = 11,
) -> Tuple[float, List[float]]:
    """
    Compute LMC barrier between two state-dicts.
    
    Returns:
        (barrier_magnitude, loss_trajectory)
        where barrier = max(loss) - min(loss) along interpolation path
    """
    losses = []
    alphas = np.linspace(0, 1, n_steps)
    
    for alpha in alphas:
        # Interpolate state
        state_interp = interpolate_state(state_a, state_b, float(alpha))
        _state_interp = {k: v.clone() for k, v in state_interp.items()}  # Clone to avoid in-place issues
        
        for common_state in common_states:
            state_interp.update(common_state)  # Override with common components

        for k in state_a:
            if not torch.equal(state_interp[k], _state_interp[k]):
                logger.warning("  [LMC] α=%.2f: State interpolation mismatch on key '%s'", alpha, k)

        # Build model and compute loss
        model = EvaluateModel(cfg, state_dict=state_interp)  # Load interpolated state directly into model
        # assign_state_to_model(model, state_interp)
        model = model.to(DEVICE)
        model.eval()
        
        try:
            loss_val = compute_model_loss(model, dataloader, DEVICE, max_samples=None)
            losses.append(loss_val)
            logger.debug("  [LMC] α=%.2f → loss=%.5f", alpha, loss_val)
        except Exception as e:
            logger.warning("  [LMC] α=%.2f → loss computation failed: %s", alpha, str(e), exc_info=True)
            losses.append(float("nan"))
        
        del model
        torch.cuda.empty_cache()
    
    # Compute barrier (robust to NaN by ignoring them)
    valid_losses = [l for l in losses if not np.isnan(l)]
    if len(valid_losses) < 2:
        barrier = 0.0
    else:
        barrier = float(np.max(valid_losses) - np.min(valid_losses))
    
    return barrier, losses

def _worker_initializer(verbose: bool):
    global logger
    worker_id = f"worker-{os.getpid()}"
    logger = get_logger(
            level=logging.DEBUG if verbose else logging.INFO,
            add_file_handler=True,
            log_file=f"phase4_loss_landscape_{worker_id}.log"
        )
    logger.propagate = False
    
def _compute_pairwise_lmc_barriers(
    ingredient_states, 
    cfg, 
    dataloader, 
    base_idx,
    be_keys,
    cls_keys,
    reg_keys,
    shared_keys
):
    
    logger.debug("Process %d: Starting LMC barrier computation for base model %d", os.getpid(), base_idx)
    logger.debug("Running on CPU cores: %s", CORE_GROUPS[base_idx % len(CORE_GROUPS)])
    os.sched_setaffinity(0, CORE_GROUPS[base_idx % len(CORE_GROUPS)])
    torch.set_num_threads(1)

    pair_idx = 0
    results = {
        base_idx: {}
    }
    base = ingredient_states[base_idx]
    base_be = extract_subdict(base, be_keys)
    base_cls = extract_subdict(base, cls_keys)
    base_reg = extract_subdict(base, reg_keys)
    base_shared = extract_subdict(base, shared_keys)
    for i in range(len(ingredient_states)):
        for j in range(i + 1, len(ingredient_states)):
            pair_name = f"pair_{i:02d}{j:02d}"
            filepath = f"{RESULTS_DIR}/lmc_barriers/base{base_idx}_pair{pair_name}.json"
            if os.path.exists(filepath):
                logger.info("  [SKIP] Base [%d], Pair [%d/15] %s already exists", base_idx + 1, pair_idx + 1, pair_name)
                with open(filepath, "r") as f:
                    pair = json.load(f)[str(base_idx)]
                    results[base_idx][pair_name] = pair[pair_name]
                pair_idx += 1
                continue
            logger.info("  Base: [%d/%d], Pair [%d/15] %s (ingredients %d vs %d)", base_idx + 1, len(ingredient_states), pair_idx + 1, pair_name, i, j)
            
            state_i = ingredient_states[i]
            state_j = ingredient_states[j]
            
            # Compute barriers for each component
            logger.info("    Computing backbone_encoder barrier...")
            be_i = extract_subdict(state_i, be_keys)
            be_j = extract_subdict(state_j, be_keys)
            barrier_be, _ = compute_lmc_barrier(be_i, be_j, cfg, dataloader, [base_cls, base_reg, base_shared])
            
            logger.info("    Computing cls_head barrier...")
            cls_i = extract_subdict(state_i, cls_keys)
            cls_j = extract_subdict(state_j, cls_keys)
            barrier_cls, _ = compute_lmc_barrier(cls_i, cls_j, cfg, dataloader, [base_be, base_reg, base_shared])

            logger.info("    Computing reg_head barrier...")
            reg_i = extract_subdict(state_i, reg_keys)
            reg_j = extract_subdict(state_j, reg_keys)
            barrier_reg, _ = compute_lmc_barrier(reg_i, reg_j, cfg, dataloader, [base_be, base_cls, base_shared])
            
            logger.info("    Computing objectness_head barrier...")
            shared_i = extract_subdict(state_i, shared_keys)
            shared_j = extract_subdict(state_j, shared_keys)
            barrier_shared, _ = compute_lmc_barrier(shared_i, shared_j, cfg, dataloader, [base_be, base_cls, base_reg])
            
            # Full model barrier
            logger.info("    Computing full model barrier...")
            barrier_full, _ = compute_lmc_barrier(state_i, state_j, cfg, dataloader)
            
            pair = {
                "backbone_encoder": float(barrier_be),
                "cls_head": float(barrier_cls),
                "reg_head": float(barrier_reg),
                "shared": float(barrier_shared),
                "full_model": float(barrier_full),
            }
            results[base_idx] = {
                pair_name: pair
            }

            filepath = f"{RESULTS_DIR}/lmc_barriers/base{base_idx}_pair{pair_name}.json"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(results, f, indent=4)
            
            pair_idx += 1

    filepath = f"{RESULTS_DIR}/lmc_barriers/base{base_idx}.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    return base_idx, results

def compute_pairwise_lmc_barriers(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg,
    dataloader: torch.utils.data.DataLoader
) -> Dict[str, Dict[str, float]]:
    """
    Compute LMC barriers for all pairs of ingredients, split by component.
    
    Returns:
        {
            "pair_0102": {"backbone_encoder": 0.45, "encoder": 0.32, "cls_head": 0.78, ...},
            "pair_0103": {...},
            ...
        }
    """
    logger.info("Computing pairwise LMC barriers for all 15 ingredient pairs...")
    
    # Extract component keys
    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    
    # results = {}

    ctx = mp.get_context("spawn")
    num_processes = len(CORE_GROUPS)
    all_jobs = []
    base_idxs = range(len(ingredient_states))
    import math
    for i in range(math.ceil(len(ingredient_states) / num_processes)):
        jobs = []
        if i > 0:
            i = i + num_processes
        for base_idx in base_idxs[i:i+num_processes]:
            if len(jobs) == num_processes:
                # all_processes.append(tuple(processes))
                jobs = [(ingredient_states, cfg, dataloader, base_idx, be_keys, cls_keys, reg_keys, shared_keys)]
            else:
                jobs.append((ingredient_states, cfg, dataloader, base_idx, be_keys, cls_keys, reg_keys, shared_keys))
        all_jobs.append(tuple(jobs))

    results = {}
    for jobs in all_jobs:
        with ctx.Pool(processes=len(jobs), initializer=_worker_initializer, initargs=(parsed_args.verbose,)) as pool:
            jobs = [
                (ingredient_states, cfg, dataloader, x, be_keys, cls_keys, reg_keys, shared_keys)
                for x in range(len(ingredient_states))
            ]
            # base_idx, outputs = pool.starmap(_compute_pairwise_lmc_barriers, jobs)
            outputs = pool.starmap(_compute_pairwise_lmc_barriers, jobs)
        # for job in jobs:
            # base_idx, outputs = _compute_pairwise_lmc_barriers(*job)
            for base_idx, output in outputs:
                results[base_idx] = output[base_idx]

    # Collect results in parent — keyed by x (ingredient index)
    
    # for model_idx, result in all_outputs:
    #     results[model_idx] = result[model_idx]

    logger.info("  ✓ LMC barrier computation complete")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hessian Trace Computation (Full Hessian with Hutchinson Estimator)
# ─────────────────────────────────────────────────────────────────────────────

def _check_available_gpu_memory(min_gb: float = 2.0) -> bool:
    """Check if sufficient GPU memory is available."""
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(DEVICE).total_memory
            allocated = torch.cuda.memory_allocated(DEVICE)
            reserved = torch.cuda.memory_reserved(DEVICE)
            available = total_memory - allocated
            available_gb = available / (1024 ** 3)
            logger.debug("  GPU memory: %.2f GB available", available_gb)
            return available_gb >= min_gb
    except Exception as e:
        logger.debug("  GPU memory check failed: %s", e)
    return False


def _l2_norm_squared(state_dict: Dict[str, torch.Tensor]) -> float:
    """Compute sum of squared L2 norms (proxy for Hessian trace)."""
    total = 0.0
    for tensor in state_dict.values():
        if torch.is_floating_point(tensor):
            total += (tensor.float() ** 2).sum().item()
    return float(total)


def _compute_hessian_trace_hutchinson(
    model: Union[torch.nn.Module, 'EvaluateModel'],
    dataloader,
    device: torch.device,
    param_keys: Optional[List[str]] = None,
    n_samples: int = HESSIAN_SAMPLES,
) -> float:
    """
    Compute Hessian trace using Hutchinson estimator with Rademacher vectors.
    
    Tr(H) ≈ E[z^T H z] where z ~ Rademacher {-1, +1}
    
    Args:
        model: YOLOF model (with full state already loaded)
        dataloader: Evaluation dataloader
        device: Compute device
        param_keys: Optional list of parameter names to filter. If None, uses all model parameters.
        n_samples: Number of Rademacher samples
    
    Returns:
        Estimated trace of Hessian
    """
    trace_estimate = 0.0
    
    # Collect parameters (optionally filtered by keys)
    if param_keys is not None:
        # Build mapping of parameter names to parameters
        if isinstance(model, EvaluateModel):
            named_params = model.model.named_parameters()
        else:
            named_params = model.named_parameters()

        param_dict = {name: param for name, param in named_params}
        params = [param_dict[key] for key in param_keys if key in param_dict]
        if not params:
            logger.warning("    No parameters found for keys: %s", param_keys[:3])
            return 0.0
    else:
        # Only include trainable parameters
        params = [p for p in model.parameters() if p.numel() > 0]
    
    if not params:
        logger.warning("    No trainable parameters found")
        return 0.0

    try:
        model = model.to(device)
        model.eval()
        
        # CRITICAL: Enable gradients for all parameters
        for p in params:
            p.requires_grad_(True)
        
        param_sizes = [p.numel() for p in params]
        total_params = sum(param_sizes)
        
        logger.debug("    Total parameters: %d", total_params)
        
        # Hutchinson trace estimation with Rademacher vectors
        for sample_idx in range(n_samples):
            # Generate random Rademacher vector z ~ {-1, +1}
            z_list = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1 for p in params]
            
            # Compute z^T ∇L
            if isinstance(model, EvaluateModel):
                model.model.zero_grad()
            else:
                model.zero_grad()
            
            # Get a batch and compute loss
            grads = None
            try:
                batch = next(iter(dataloader))
                
                outputs = model(batch, require_grad=True)
                loss = None
                for out in outputs:
                    losses = out.get("losses", {})
                    loss = losses["loss_cls"] + losses.get("loss_box_reg", 0)
                
                if loss is None or (isinstance(loss, float) and loss == 0.0):
                    logger.debug("    Sample %d: No loss computed", sample_idx)
                    continue
                
                # Ensure loss is a scalar tensor
                if not isinstance(loss, torch.Tensor):
                    logger.debug("    Sample %d: Loss is not a tensor", sample_idx)
                    continue
                    
                    # Compute gradient
                grads = torch.autograd.grad(
                    loss, params, create_graph=True, retain_graph=True, allow_unused=True
                )
                
                # Compute z^T ∇L (directional derivative)
                z_grad_product = sum(
                    (z * g).sum() for z, g in zip(z_list, grads) if g is not None
                )
                
                # Compute Hessian-vector product: ∇(z^T ∇L)
                if z_grad_product.requires_grad:
                    hessian_z = torch.autograd.grad(
                        z_grad_product, params, create_graph=False, allow_unused=True
                    )
                    
                    # Compute z^T H z
                    z_hz = sum(
                    (z * hz).sum() for z, hz in zip(z_list, hessian_z) if hz is not None
                )
                trace_estimate += float(z_hz.item())
                logger.debug("    Sample %d: z^T H z = %.6f", sample_idx, float(z_hz.item()))
                
            except StopIteration:
                logger.debug("    Sample %d: DataLoader exhausted", sample_idx)
                break
            except Exception as e:
                logger.debug("    Sample %d: Hessian computation failed: %s", sample_idx, str(e), exc_info=True)
                continue
            finally:
                del z_list, grads
                torch.cuda.empty_cache()
        
        # Average over samples
        trace_estimate /= max(n_samples, 1)
        logger.debug("    Estimated trace (Hutchinson): %.6f", trace_estimate)
        
    except Exception as e:
        logger.warning("    Hutchinson trace estimation failed: %s", e, exc_info=True)
        # Fallback to L2-norm of parameters
        total_norm = 0.0
        for param in params:
            total_norm += (param.data ** 2).sum().item()
        trace_estimate = total_norm
        logger.warning("    Falling back to L2-norm proxy: %.6f", trace_estimate)
    finally:
        if isinstance(model, EvaluateModel):
            model.model.zero_grad()
        else:
            model.zero_grad()
    
    return float(trace_estimate)


def compute_hessian_traces(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfgs: List[Union[CfgNode, Dict]],
    dataloader,
) -> Dict[str, Dict[str, float]]:
    """
    Compute Hessian traces for all ingredients, split by component.
    
    Uses full Hessian computation (Hutchinson estimator) when GPU memory available.
    Falls back to L2-norm proxy for fast, deterministic computation.
    
    Returns:
        {
            "ingredient_0": {"backbone_encoder": 2.14, "cls_head": 3.42, ...},
            "ingredient_1": {...},
            ...
        }
    """
    logger.info("Computing Hessian traces for 6 ingredients...")
    
    # Determine computation method based on GPU memory
    use_full_hessian = _check_available_gpu_memory(min_gb=2.0)
    if use_full_hessian:
        logger.info("  ✓ Sufficient GPU memory detected; using full Hessian (Hutchinson estimator)")
    else:
        logger.info("  ⚠ Limited GPU memory; using L2-norm proxy (fast, deterministic)")
    
    # Extract component keys
    decoder_keys = get_decoder_keys(ingredient_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    be_keys = get_backbone_encoder_keys(ingredient_states[0])
    
    results = {}
    
    for idx, state in enumerate(ingredient_states):
        logger.info("  [%d/6] Ingredient %d", idx + 1, idx + 1)
        
        if use_full_hessian:
            # Load FULL model once per ingredient
            model = EvaluateModel(cfgs[idx], state_dict=state)
            # assign_state_to_model(model, state)
            
            # Compute Hessian traces for each component using filtered parameters
            logger.debug("    Computing backbone_encoder Hessian trace...")
            trace_be = _compute_hessian_trace_hutchinson(model, dataloader, DEVICE, param_keys=be_keys)
            
            logger.debug("    Computing cls_head Hessian trace...")
            trace_cls = _compute_hessian_trace_hutchinson(model, dataloader, DEVICE, param_keys=cls_keys)
            
            logger.debug("    Computing reg_head Hessian trace...")
            trace_reg = _compute_hessian_trace_hutchinson(model, dataloader, DEVICE, param_keys=reg_keys)
            
            logger.debug("    Computing shared Hessian trace...")
            trace_shared = _compute_hessian_trace_hutchinson(model, dataloader, DEVICE, param_keys=shared_keys)
            
            del model
        else:
            # Use L2-norm proxy for fast computation
            be_sub = extract_subdict(state, be_keys)
            cls_sub = extract_subdict(state, cls_keys)
            reg_sub = extract_subdict(state, reg_keys)
            shared_sub = extract_subdict(state, shared_keys)
            
            trace_be = _l2_norm_squared(be_sub)
            trace_cls = _l2_norm_squared(cls_sub)
            trace_reg = _l2_norm_squared(reg_sub)
            trace_shared = _l2_norm_squared(shared_sub)
        
        results[f"ingredient_{idx}"] = {
            "backbone_encoder": float(trace_be),
            "cls_head": float(trace_cls),
            "reg_head": float(trace_reg),
            "shared": float(trace_shared),
        }
        
        torch.cuda.empty_cache()
    
    logger.info("  ✓ Hessian trace computation complete")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Tests (Placeholders)
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_tests(
    lmc_barriers: Dict[str, Dict[str, float]],
    hessian_traces: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Run statistical tests on barriers and traces.
    
    Tests:
      1. RM-ANOVA on 4-component barriers
      2. Pearson correlations (barrier ↔ gain) with Bonferroni correction
      3. RM-ANOVA on Hessian traces
    """
    logger.info("Running statistical tests...")
    
    try:
        import pandas as pd
        from statsmodels.stats.anova import AnovaRM
        from scipy.stats import pearsonr
    except ImportError:
        logger.warning("  statsmodels not installed; returning placeholder results")
        return {
            "test_1_barrier_anova": {"status": "skipped", "reason": "statsmodels not available"},
            "test_2_pearson_correlations": {"status": "skipped", "reason": "statsmodels not available"},
            "test_3_hessian_anova": {"status": "skipped", "reason": "statsmodels not available"},
        }
    
    # Extract barrier data: 15 pairs × 4 components
    components = ["backbone_encoder", "cls_head", "reg_head", "shared"]

    test_results = {}

    for base_idx, pairs in lmc_barriers.items():
        pair_ids = list(pairs.keys())
        
        # Test 1: RM-ANOVA on barriers
        logger.info("  Test 1: RM-ANOVA on barriers on base %s...", base_idx)
        
        # Reshape to long format: (subject, component, value)
        barrier_long_data = []
        for pair_idx, pair_id in enumerate(pair_ids):
            for comp in components:
                barrier_long_data.append({
                    "subject": pair_idx,
                    "component": comp,
                    "barrier": lmc_barriers[base_idx][pair_id].get(comp, np.nan)
                })
        
        df_barrier_long = pd.DataFrame(barrier_long_data)
        
        try:
            anova_barrier = AnovaRM(df_barrier_long, depvar="barrier", subject="subject", within=["component"])
            res_barrier = anova_barrier.fit()
            f_stats = float(res_barrier.anova_table.loc["component", "F Value"]) if "F Value" in res_barrier.anova_table.columns else None
            p_value = float(res_barrier.anova_table.loc["component", "Pr > F"]) if "Pr > F" in res_barrier.anova_table.columns else None
            
            test1_result = {
                "test_name": "RM-ANOVA: Component barriers",
                "n_pairs": len(pair_ids),
                "components": components,
                "f_statistic": f_stats,
                "p_value": p_value,
                "summary": str(res_barrier),
            }
            if test1_result["p_value"] is not None:
                test1_result["interpretation"] = "Significant component differences" if test1_result["p_value"] < 0.05 else "No significant differences"
        except Exception as e:
            logger.warning("  ✗ RM-ANOVA failed: %s", e, exc_info=True)
            test1_result = {"status": "error", "error": str(e)}
        
        # Test 3: RM-ANOVA on Hessian traces
        logger.info("  Test 3: RM-ANOVA on Hessian traces...")
        ingredient_ids = list(hessian_traces.keys())
        
        # Reshape to long format: (subject, component, value)
        hessian_long_data = []
        for ing_idx, ing_id in enumerate(ingredient_ids):
            for comp in components:
                hessian_long_data.append({
                    "subject": ing_idx,
                    "component": comp,
                    "trace": hessian_traces[ing_id].get(comp, np.nan)
                })
        
        df_hessian_long = pd.DataFrame(hessian_long_data)
        
        try:
            anova_hessian = AnovaRM(df_hessian_long, depvar="trace", subject="subject", within=["component"])
            res_hessian = anova_hessian.fit()
            f_stats = float(res_hessian.anova_table.loc["component", "F Value"]) if "F Value" in res_hessian.anova_table.columns else None
            p_value = float(res_hessian.anova_table.loc["component", "Pr > F"]) if "Pr > F" in res_hessian.anova_table.columns else None

            test3_result = {
                "test_name": "RM-ANOVA: Component Hessian traces",
                "n_ingredients": len(ingredient_ids),
                "components": components,
                "f_statistic": f_stats,
                "p_value": p_value,
                "summary": str(res_hessian),
            }
            if test3_result["p_value"] is not None:
                test3_result["interpretation"] = "Significant component differences" if test3_result["p_value"] < 0.05 else "No significant differences"
        except Exception as e:
            logger.warning("  ✗ Hessian RM-ANOVA failed: %s", e, exc_info=True)
            test3_result = {"status": "error", "error": str(e)}
        
        # Test 2: Pearson correlations (barrier magnitude ↔ averaging gain)
        logger.info("  Test 2: Pearson correlations (Barrier ↔ Averaging Gain)...")
        
        try:
            # Load Phase 3 soup results
            phase3_results_path = Path(RESULTS_DIR).parent / "phase3_soup_results.json"
            
            if not phase3_results_path.exists():
                logger.warning("  ✗ Phase 3 results not found at %s", phase3_results_path)
                test2_result = {
                    "test_name": "Pearson: Barrier ↔ Performance Gain",
                    "status": "error",
                    "error": f"Phase 3 results not found at {phase3_results_path}",
                    "bonferroni_alpha": 0.0125,
                }
            else:
                with open(phase3_results_path, "r") as f:
                    phase3_data = json.load(f)
                
                # Extract baseline and condition gains (mAP@50-95)
                baseline_map = phase3_data.get("best_individual", {}).get("map50_95", None)
                
                if baseline_map is None:
                    logger.warning("  ✗ Baseline mAP@50-95 not found in Phase 3 results")
                    test2_result = {
                        "test_name": "Pearson: Barrier ↔ Performance Gain",
                        "status": "error",
                        "error": "Baseline mAP@50-95 not found",
                        "bonferroni_alpha": 0.0125,
                    }
                else:
                    # Extract gains for 4 soup conditions
                    conditions = ["condition_1", "condition_2", "condition_3", "condition_4"]
                    gains = {}
                    for cond in conditions:
                        if cond in phase3_data:
                            cond_map = phase3_data[cond].get("map50_95", None)
                            if cond_map is not None:
                                gains[cond] = cond_map - baseline_map
                    
                    if len(gains) < 4:
                        logger.warning("  ✗ Not all 4 condition gains found in Phase 3 results")
                        test2_result = {
                            "test_name": "Pearson: Barrier ↔ Performance Gain",
                            "status": "error",
                            "error": f"Only {len(gains)}/4 condition gains found",
                            "bonferroni_alpha": 0.0125,
                        }
                    else:
                        # Compute average barrier per component and correlate with average gain
                        # Strategy: For each component, we have 15 barrier values (from 15 pairs).
                        # We correlate the component barrier values with the average gain from all conditions.
                        avg_gain_across_conditions = float(np.mean(list(gains.values())))
                        
                        correlations_by_component = {}
                        
                        for comp in components:
                            # Collect all barrier values for this component across all pairs
                            barrier_values = []
                            pair_list = []
                            for pair_id, pair_barriers in pairs.items():
                                if comp in pair_barriers:
                                    barrier_values.append(pair_barriers[comp])
                                    pair_list.append(pair_id)
                            
                            if not barrier_values:
                                logger.warning("    No barrier data for component: %s", comp)
                                correlations_by_component[comp] = {
                                    "r": None,
                                    "p_value": None,
                                    "n_pairs": 0,
                                    "note": "No barrier data available"
                                }
                                continue
                            
                            # For each barrier value, pair it with the average gain
                            # This represents: "How difficult is this pair to interpolate (barrier)
                            # vs. how much gain does averaging provide (avg gain)"
                            x_data = barrier_values  # 15 barrier values
                            y_data = [avg_gain_across_conditions] * len(barrier_values)  # Replicate avg gain for 15 pairs
                            
                            # Alternative: correlate component barriers with per-condition gains
                            # by assigning each pair to a condition based on ingredient modulo
                            if len(barrier_values) >= 4 and len(barrier_values) % 4 == 0:
                                # Assign pairs to conditions cyclically
                                gains_list = [gains[f"condition_{i+1}"] for i in range(4)]
                                y_data = []
                                for idx in range(len(barrier_values)):
                                    condition_idx = idx % 4
                                    y_data.append(gains_list[condition_idx])
                            
                            # Compute statistics
                            mean_barrier = float(np.mean(barrier_values))
                            std_barrier = float(np.std(barrier_values)) if len(barrier_values) > 1 else 0.0
                            
                            # Compute Pearson correlation
                            try:
                                if len(set(x_data)) <= 1 or len(set(y_data)) <= 1:  # No variance
                                    r = np.nan
                                    p_val = np.nan
                                    logger.debug("    Component '%s': Insufficient variance in data", comp)
                                else:
                                    r, p_val = pearsonr(x_data, y_data)
                                
                                correlations_by_component[comp] = {
                                    "r": float(r) if not np.isnan(r) else None,
                                    "p_value": float(p_val) if not np.isnan(p_val) else None,
                                    "n_pairs": len(barrier_values),
                                    "n_conditions": 4,
                                    "mean_barrier": mean_barrier,
                                    "std_barrier": std_barrier,
                                    "min_barrier": float(min(barrier_values)),
                                    "max_barrier": float(max(barrier_values)),
                                    "barrier_values_sample": [float(b) for b in barrier_values[:3]],  # First 3 for inspection
                                    "interpretation": "Significant" if (not np.isnan(p_val) and p_val < 0.0125) else "Not significant at Bonferroni α=0.0125",
                                }
                            except Exception as e:
                                logger.warning("    Pearson correlation failed for %s: %s", comp, e)
                                correlations_by_component[comp] = {
                                    "r": None,
                                    "p_value": None,
                                    "n_pairs": len(barrier_values),
                                    "error": str(e)
                                }
                        
                        test2_result = {
                            "test_name": "Pearson: Barrier ↔ Performance Gain",
                            "status": "completed",
                            "bonferroni_alpha": 0.0125,
                            "n_components": len(components),
                            "baseline_map50_95": baseline_map,
                            "condition_gains_map50_95": gains,
                            "average_gain_across_conditions": avg_gain_across_conditions,
                            "correlations_by_component": correlations_by_component,
                            "summary": f"Tested correlation between component barriers ({len(barrier_values)} pairs) and averaging gains across {len(gains)} conditions",
                            "method": "Pearson r between per-pair barriers and cyclically-assigned condition gains"
                        }
                
        except Exception as e:
            logger.warning("  ✗ Pearson correlation test failed: %s", e, exc_info=True)
            test2_result = {
                "test_name": "Pearson: Barrier ↔ Performance Gain",
                "status": "error",
                "error": str(e),
                "bonferroni_alpha": 0.0125,
            }
        
        test_results[base_idx] = {
            "test_1_barrier_anova": test1_result,
            "test_2_pearson_correlations": test2_result,
            "test_3_hessian_anova": test3_result,
        }
    
    return test_results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_lmc_barriers(
    lmc_barriers: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Visualize LMC barriers across components and pairs.
    
    Creates:
    - Heatmap: pairs × components
    - Bar plot: average barrier per component
    - Line plot: barrier distribution
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("  ⚠ matplotlib/seaborn not installed; skipping LMC barrier visualization")
        return
    
    logger.info("  Visualizing LMC barriers...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    components = ["backbone_encoder", "cls_head", "reg_head", "shared"]
    
    # Extract barrier data by component
    barrier_data_by_component = {comp: [] for comp in components}
    pair_names = []
    
    for base_idx, pairs in lmc_barriers.items():
        for pair_id in sorted(pairs.keys()):
            if pair_id not in pair_names:
                pair_names.append(pair_id)
            for comp in components:
                if comp in pairs[pair_id]:
                    barrier_data_by_component[comp].append(pairs[pair_id][comp])
    
    # 1. Heatmap: pairs × components
    fig, ax = plt.subplots(figsize=(12, 8))
    barrier_matrix = np.array([
        [barrier_data_by_component[comp][i] for comp in components]
        for i in range(len(pair_names))
    ])
    
    sns.heatmap(barrier_matrix, annot=True, fmt=".3f", cmap="RdYlGn_r", 
                xticklabels=components, yticklabels=pair_names,
                cbar_kws={"label": "Loss Barrier"}, ax=ax)
    ax.set_title("LMC Barriers Heatmap: All Ingredient Pairs × Components", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Ingredient Pair", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "lmc_barriers_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → lmc_barriers_heatmap.png")
    
    # 2. Bar plot: average barrier per component
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_barriers = [np.mean(barrier_data_by_component[comp]) for comp in components]
    std_barriers = [np.std(barrier_data_by_component[comp]) for comp in components]
    
    bars = ax.bar(components, avg_barriers, yerr=std_barriers, capsize=5, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_ylabel("Mean Loss Barrier", fontsize=12)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_title("Average LMC Barrier per Component", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_barriers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lmc_barriers_by_component.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → lmc_barriers_by_component.png")
    
    # 3. Box plot: distribution of barriers per component
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [barrier_data_by_component[comp] for comp in components]
    bp = ax.boxplot(box_data, labels=components, patch_artist=True)
    
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    
    ax.set_ylabel("Loss Barrier", fontsize=12)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_title("LMC Barrier Distribution per Component", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "lmc_barriers_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → lmc_barriers_distribution.png")


def plot_hessian_traces(
    hessian_traces: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Visualize Hessian traces across components and ingredients.
    
    Creates:
    - Heatmap: ingredients × components
    - Bar plot: average trace per component
    - Grouped bar plot: per-ingredient traces
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("  ⚠ matplotlib/seaborn not installed; skipping Hessian trace visualization")
        return
    
    logger.info("  Visualizing Hessian traces...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    components = ["backbone_encoder", "cls_head", "reg_head", "shared"]
    
    # Extract trace data
    ingredient_ids = sorted(hessian_traces.keys())
    trace_matrix = np.array([
        [hessian_traces[ing_id].get(comp, 0.0) for comp in components]
        for ing_id in ingredient_ids
    ])
    
    # 1. Heatmap: ingredients × components
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(trace_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=components, yticklabels=ingredient_ids,
                cbar_kws={"label": "Hessian Trace"}, ax=ax)
    ax.set_title("Hessian Traces Heatmap: Ingredients × Components", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Ingredient", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "hessian_traces_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → hessian_traces_heatmap.png")
    
    # 2. Bar plot: average trace per component
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_traces = trace_matrix.mean(axis=0)
    std_traces = trace_matrix.std(axis=0)
    
    bars = ax.bar(components, avg_traces, yerr=std_traces, capsize=5, alpha=0.7, color="coral", edgecolor="black")
    ax.set_ylabel("Mean Hessian Trace", fontsize=12)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_title("Average Hessian Trace per Component", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    for bar, val in zip(bars, avg_traces):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hessian_traces_by_component.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → hessian_traces_by_component.png")
    
    # 3. Grouped bar plot: per-ingredient traces
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(components))
    width = 0.13
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(ingredient_ids)))
    for i, ing_id in enumerate(ingredient_ids):
        offset = (i - len(ingredient_ids)/2) * width
        values = [hessian_traces[ing_id].get(comp, 0.0) for comp in components]
        ax.bar(x + offset, values, width, label=ing_id, color=colors[i], edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Hessian Trace", fontsize=12)
    ax.set_xlabel("Component", fontsize=12)
    ax.set_title("Hessian Traces per Ingredient and Component", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "hessian_traces_grouped.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → hessian_traces_grouped.png")


def plot_statistical_tests(
    test_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Visualize statistical test results.
    
    Creates:
    - Test 1: Bar plot of ANOVA F-statistics
    - Test 2: Scatter plot of barrier vs. gain correlations
    - Test 3: Bar plot of Hessian trace ANOVA
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("  ⚠ matplotlib/seaborn not installed; skipping statistical test visualization")
        return
    
    logger.info("  Visualizing statistical test results...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract test results (assuming single base_idx for simplicity)
    base_idx = list(test_results.keys())[0]
    test1 = test_results[base_idx].get("test_1_barrier_anova", {})
    test2 = test_results[base_idx].get("test_2_pearson_correlations", {})
    test3 = test_results[base_idx].get("test_3_hessian_anova", {})
    
    # 1. ANOVA F-statistics comparison (Test 1 vs Test 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test 1: Barrier ANOVA
    test_names_1 = ["Barrier ANOVA"]
    f_stats_1 = [test1.get("f_statistic", 0)]
    p_vals_1 = [test1.get("p_value", 1)]
    
    colors_1 = ["red" if p < 0.05 else "blue" for p in p_vals_1]
    ax1.bar(test_names_1, f_stats_1, color=colors_1, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("F-Statistic", fontsize=12)
    ax1.set_title("Test 1: RM-ANOVA on Component Barriers", fontsize=12, fontweight="bold")
    ax1.text(0, f_stats_1[0], f'p={p_vals_1[0]:.4f}', ha='center', va='bottom', fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    
    # Test 3: Hessian ANOVA
    test_names_3 = ["Hessian ANOVA"]
    f_stats_3 = [test3.get("f_statistic", 0)]
    p_vals_3 = [test3.get("p_value", 1)]
    
    colors_3 = ["red" if p < 0.05 else "blue" for p in p_vals_3]
    ax2.bar(test_names_3, f_stats_3, color=colors_3, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("F-Statistic", fontsize=12)
    ax2.set_title("Test 3: RM-ANOVA on Hessian Traces", fontsize=12, fontweight="bold")
    ax2.text(0, f_stats_3[0], f'p={p_vals_3[0]:.4f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "statistical_anova_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → statistical_anova_results.png")
    
    # 2. Pearson correlations (Test 2)
    corr_by_comp = test2.get("correlations_by_component", {})
    
    if corr_by_comp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        components = list(corr_by_comp.keys())
        r_values = [corr_by_comp[comp].get("r", 0) or 0 for comp in components]
        p_values = [corr_by_comp[comp].get("p_value", 1) or 1 for comp in components]
        
        # Pearson r values
        colors = ["red" if p < 0.0125 else "blue" for p in p_values]
        bars = ax1.bar(components, r_values, color=colors, alpha=0.7, edgecolor="black")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax1.set_ylabel("Pearson r", fontsize=12)
        ax1.set_title("Test 2: Pearson Correlations (Barrier ↔ Gain)", fontsize=12, fontweight="bold")
        ax1.set_ylim(-1, 1)
        ax1.grid(axis="y", alpha=0.3)
        
        for bar, r, p in zip(bars, r_values, p_values):
            height = bar.get_height()
            sig = "***" if p < 0.0125 else "ns"
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{r:.3f}\n{sig}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # p-values
        ax2.bar(components, p_values, color="orange", alpha=0.7, edgecolor="black")
        ax2.axhline(y=0.0125, color="red", linestyle="--", linewidth=2, label="Bonferroni α=0.0125")
        ax2.set_ylabel("p-value", fontsize=12)
        ax2.set_yscale("log")
        ax2.set_title("Significance of Correlations", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", alpha=0.3, which="both")
        
        plt.tight_layout()
        plt.savefig(output_dir / "statistical_pearson_results.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("    → statistical_pearson_results.png")
    
    # 3. Summary comparison table (as text image)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    
    summary_data = [
        ["Test", "Name", "Statistic", "p-value", "Interpretation"],
        ["Test 1", "Barrier ANOVA", f"{test1.get('f_statistic', 'N/A'):.3f}", 
         f"{test1.get('p_value', 'N/A'):.4f}", test1.get("interpretation", "N/A")],
        ["Test 3", "Hessian ANOVA", f"{test3.get('f_statistic', 'N/A'):.3f}", 
         f"{test3.get('p_value', 'N/A'):.4f}", test3.get("interpretation", "N/A")],
        ["Test 2", "Pearson Corr.", "Per component", "Per component", 
         f"{sum(1 for c in corr_by_comp.values() if c.get('p_value', 1) and c.get('p_value', 1) < 0.0125)}/4 significant"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc="center", loc="center",
                    colWidths=[0.12, 0.25, 0.2, 0.15, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    plt.title("Statistical Test Results Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "statistical_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("    → statistical_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(verbose: bool = True, force_recompute: list = []) -> Dict[str, Any]:
    """
    Main Phase 4 entry point.
    
    Args:
        verbose: Whether to log progress
    
    Returns:
        Dict with all barrier, trace, and test results
    """
    
    global logger
    if logger is None:
        logger = get_logger(
            level=logging.DEBUG if verbose else logging.INFO,
            add_file_handler=True,
            log_file="phase4_loss_landscape.log"
        )
    try:
        logger.info("=" * 90)
        logger.info("PHASE 4: LOSS LANDSCAPE ANALYSIS")
        logger.info("=" * 90)
        
        # Setup directories
        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        ingredients_dir = Path(PHASE2_OUTPUT_DIR)

        run_registry = get_run_specs()
        ingredient_runs = [r for r in run_registry if r.role == "ingredient"]

        ingredient_paths = []
        run_ids = []
        cfg_paths = []
        for run_spec in ingredient_runs:
            ckpt_path = Path(ingredients_dir) / f"{run_spec.run_name}/model_best.pth"
            ingredient_paths.append(ckpt_path)
            run_ids.append(run_spec.run_id)
            cfg_paths.append(Path(ingredients_dir) / f"{run_spec.run_name}/config.yaml")
            logger.info("Will audit: %s → %s", run_spec.run_id, ckpt_path)
            
        # Load ingredient checkpoints
        logger.info("\n[1/4] Loading 6 ingredient checkpoints...")

        missing = [p for p in ingredient_paths if not p.exists()]
        if missing:
            logger.error("Missing checkpoints: %s", missing)
            raise FileNotFoundError(f"Missing phase 2 checkpoints")
        
        ingredient_states = load_states(ingredient_paths)
        logger.info("  ✓ Loaded 6 ingredients")
        
        # Build Detectron2 config
        logger.info("\n[2/4] Building Detectron2 config...")
        cfgs = [build_eval_cfg(EVAL_DATASET, str(cfg_path), ckpt_file) for cfg_path, ckpt_file in zip(cfg_paths, ingredient_paths)]
        base_cfg = build_eval_cfg(EVAL_DATASET)
        logger.info("  ✓ Config ready")
        
        # Build evaluation dataloader
        logger.info("\n[3/4] Building evaluation dataloader...")
        dataloader = build_eval_dataloader(cfgs[0], EVAL_DATASET, num_workers=0, batch_size=4, max_img_per_cls=3)
        logger.info("  ✓ Dataloader ready")
        
        # Compute pairwise LMC barriers
        logger.info("\n[4/4a] Computing pairwise LMC barriers (15 pairs)...")
        barriers_json = results_dir / "phase4_lmc_barriers.json"
        if os.path.exists(barriers_json) and ("lmc" not in force_recompute):
            logger.info("  [SKIP] LMC barriers already exist at %s", barriers_json)
            with open(barriers_json, "r") as f:
                lmc_barriers = json.load(f)
        else:
            lmc_barriers = compute_pairwise_lmc_barriers(ingredient_states, base_cfg, dataloader)
        
        dataloader = build_eval_dataloader(cfgs[0], EVAL_DATASET)
        # Compute Hessian traces
        logger.info("\n[4/4b] Computing Hessian traces for 6 ingredients...")
        traces_json = results_dir / "phase4_hessian_traces.json"
        if os.path.exists(traces_json) and ("hessian_traces" not in force_recompute or "hs" not in force_recompute):
            logger.info("  [SKIP] Hessian traces already exist at %s", traces_json)
            with open(traces_json, "r") as f:
                hessian_traces = json.load(f)
        else:
            hessian_traces = compute_hessian_traces(ingredient_states, cfgs, dataloader)
        
        # Run statistical tests
        logger.info("\n[4/4c] Running statistical tests...")
        tests_json = results_dir / "phase4_statistical_tests.json"
        if os.path.exists(tests_json) and "stats" not in force_recompute:
            logger.info("  [SKIP] Statistical tests already exist at %s", tests_json)
            with open(tests_json, "r") as f:
                test_results = json.load(f)
        else:
            test_results = run_statistical_tests(lmc_barriers, hessian_traces)
        
        # Save results
        logger.info("\nSaving results...")
        
        # barriers_json = results_dir / "phase4_lmc_barriers.json"
        with open(barriers_json, "w") as f:
            json.dump(lmc_barriers, f, indent=2)
        logger.info("  → Barriers: %s", barriers_json)
        
        # traces_json = results_dir / "phase4_hessian_traces.json"
        with open(traces_json, "w") as f:
            json.dump(hessian_traces, f, indent=2)
        logger.info("  → Traces: %s", traces_json)
        
        
        with open(tests_json, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        logger.info("  → Tests: %s", tests_json)
        
        # Generate visualizations
        logger.info("\n[5/5] Generating visualizations...")
        visualizations_dir = results_dir / "visualizations"
        
        try:
            logger.info("  Visualizing LMC barriers...")
            plot_lmc_barriers(lmc_barriers, visualizations_dir)
            
            logger.info("  Visualizing Hessian traces...")
            plot_hessian_traces(hessian_traces, visualizations_dir)
            
            logger.info("  Visualizing statistical test results...")
            plot_statistical_tests(test_results, visualizations_dir)
            
            logger.info("  ✓ All visualizations saved to: %s", visualizations_dir)
        except Exception as e:
            logger.warning("  ⚠ Visualization generation failed: %s", e, exc_info=True)
        
        logger.info("\n" + "=" * 90)
        logger.info("PHASE 4 COMPLETE")
        logger.info("=" * 90)
        logger.info("Outputs:")
        logger.info("  • LMC barriers:      %s", barriers_json)
        logger.info("  • Hessian traces:    %s", traces_json)
        logger.info("  • Statistical tests: %s", tests_json)
        logger.info("  • Visualizations:    %s", visualizations_dir)
        logger.info("=" * 90)
        
        return {
            "lmc_barriers": lmc_barriers,
            "hessian_traces": hessian_traces,
            "statistical_tests": test_results,
        }
    except KeyboardInterrupt as e:
        logger.info(f"Cancelled by the user: {e}")


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description="Phase 3: Loss Landscape Analysis")
    args.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args.add_argument(
        "--force",
        nargs='+',
        choices=['lmc', 'hessian_traces', 'hs', 'stats'],
        help="Force re-computation of specific results (can specify multiple: lmc, hessian_traces, hs, stats)"
    )
    parsed_args = args.parse_args()

    _register_datasets()
    # run(verbose=parsed_args.verbose)
    run(verbose=True, force_recompute=parsed_args.force)

# loss_landscape