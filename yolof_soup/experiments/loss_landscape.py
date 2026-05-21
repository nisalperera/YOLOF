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
    list(range(0, 6)),    # Model 0 → cores 0–5
    list(range(6, 12)),   # Model 1 → cores 6–11
    list(range(12, 18)),  # Model 2 → cores 12–17
    list(range(18, 24)),  # Model 3 → cores 18–23
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
    

    os.sched_setaffinity(0, CORE_GROUPS[base_idx % len(CORE_GROUPS)])
    torch.set_num_threads(1)

    pair_idx = 0
    results = {}
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
                pair_idx += 1
                continue
            logger.info("  Base: [%d/%d], Pair [%d/15] %s (ingredients %d vs %d)", base_idx + 1, len(ingredient_states), pair_idx + 1, pair_name, i, j)
            
            state_i = ingredient_states[i]
            state_j = ingredient_states[j]
            
            # Compute barriers for each component
            be_i = extract_subdict(state_i, be_keys)
            be_j = extract_subdict(state_j, be_keys)
            barrier_be, _ = compute_lmc_barrier(be_i, be_j, cfg, dataloader, [base_cls, base_reg, base_shared])
            
            cls_i = extract_subdict(state_i, cls_keys)
            cls_j = extract_subdict(state_j, cls_keys)
            barrier_cls, _ = compute_lmc_barrier(cls_i, cls_j, cfg, dataloader, [base_be, base_reg, base_shared])

            reg_i = extract_subdict(state_i, reg_keys)
            reg_j = extract_subdict(state_j, reg_keys)
            barrier_reg, _ = compute_lmc_barrier(reg_i, reg_j, cfg, dataloader, [base_be, base_cls, base_shared])
            
            shared_i = extract_subdict(state_i, shared_keys)
            shared_j = extract_subdict(state_j, shared_keys)
            barrier_shared, _ = compute_lmc_barrier(shared_i, shared_j, cfg, dataloader, [base_be, base_cls, base_reg])
            
            # Full model barrier
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
    all_processes = []
    for base_idx in range(len(ingredient_states)):
        processes = []
        if base_idx % 6 == 0:
            all_processes.append(tuple(processes))
            processes = [(ingredient_states, cfg, dataloader, base_idx, be_keys, cls_keys, reg_keys, shared_keys)]
        else:
            processes.append((ingredient_states, cfg, dataloader, base_idx, be_keys, cls_keys, reg_keys, shared_keys))

    all_outputs = []
    for jobs in all_processes:
        with ctx.Pool(processes=len(jobs), initializer=_worker_initializer, initargs=(parsed_args.verbose,)) as pool:
            # jobs = [
            #     (ingredient_states, cfg, dataloader, x, be_keys, cls_keys, reg_keys, shared_keys)
            #     for x in range(len(ingredient_states))
            # ]
            outputs = pool.starmap(_compute_pairwise_lmc_barriers, jobs)
            all_outputs.extend(outputs)

    # Collect results in parent — keyed by x (ingredient index)
    results = {}
    for model_idx, result in all_outputs:
        results[model_idx] = result

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
            model = EvaluateModel(cfgs[idx])
            assign_state_to_model(model, state)
            
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
    pair_ids = list(lmc_barriers.keys())
    
    # Test 1: RM-ANOVA on barriers
    logger.info("  Test 1: RM-ANOVA on barriers...")
    
    # Reshape to long format: (subject, component, value)
    barrier_long_data = []
    for pair_idx, pair_id in enumerate(pair_ids):
        for comp in components:
            barrier_long_data.append({
                "subject": pair_idx,
                "component": comp,
                "barrier": lmc_barriers[pair_id].get(comp, np.nan)
            })
    
    df_barrier_long = pd.DataFrame(barrier_long_data)
    
    try:
        anova_barrier = AnovaRM(df_barrier_long, depvar="barrier", subject="subject", within=["component"])
        res_barrier = anova_barrier.fit()
        
        test1_result = {
            "test_name": "RM-ANOVA: Component barriers",
            "n_pairs": len(pair_ids),
            "components": components,
            "f_statistic": float(res_barrier.anova_table.loc["component", "F"]) if "F" in res_barrier.anova_table.columns else None,
            "p_value": float(res_barrier.anova_table.loc["component", "PR(>F)"]) if "PR(>F)" in res_barrier.anova_table.columns else None,
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
        
        test3_result = {
            "test_name": "RM-ANOVA: Component Hessian traces",
            "n_ingredients": len(ingredient_ids),
            "components": components,
            "f_statistic": f_stat,
            "p_value": p_value,
            "summary": str(res_hessian),
        }
        if p_value is not None:
            test3_result["interpretation"] = "Significant component differences" if p_value < 0.05 else "No significant differences"
    except Exception as e:
        logger.warning("  ✗ Hessian RM-ANOVA failed: %s", e, exc_info=True)
        test3_result = {"status": "error", "error": str(e)}
    
    # Test 2: Pearson correlations (placeholder - requires Phase 3 gains)
    test2_result = {
        "test_name": "Pearson: Barrier ↔ Performance Gain",
        "status": "requires_phase3_gains",
        "bonferroni_alpha": 0.0125,
        "note": "Needs averaging gains from Phase 3 results"
    }
    
    test_results = {
        "test_1_barrier_anova": test1_result,
        "test_2_pearson_correlations": test2_result,
        "test_3_hessian_anova": test3_result,
    }
    
    return test_results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True) -> Dict[str, Any]:
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
        lmc_barriers = compute_pairwise_lmc_barriers(ingredient_states, base_cfg, dataloader)
        # barriers_json = results_dir / "phase4_lmc_barriers.json"
        # with open(barriers_json, "r") as f:
        #     lmc_barriers = json.load(f)
        
        dataloader = build_eval_dataloader(cfgs[0], EVAL_DATASET)
        # Compute Hessian traces
        logger.info("\n[4/4b] Computing Hessian traces for 6 ingredients...")
        hessian_traces = compute_hessian_traces(ingredient_states, cfgs, dataloader)
        
        # Run statistical tests
        logger.info("\n[4/4c] Running statistical tests...")
        test_results = run_statistical_tests(lmc_barriers, hessian_traces)
        
        # Save results
        logger.info("\nSaving results...")
        
        barriers_json = results_dir / "phase4_lmc_barriers.json"
        # with open(barriers_json, "w") as f:
        #     json.dump(lmc_barriers, f, indent=2)
        logger.info("  → Barriers: %s", barriers_json)
        
        traces_json = results_dir / "phase4_hessian_traces.json"
        with open(traces_json, "w") as f:
            json.dump(hessian_traces, f, indent=2)
        logger.info("  → Traces: %s", traces_json)
        
        tests_json = results_dir / "phase4_statistical_tests.json"
        with open(tests_json, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        logger.info("  → Tests: %s", tests_json)
        
        logger.info("\n" + "=" * 90)
        logger.info("PHASE 4 COMPLETE")
        logger.info("=" * 90)
        logger.info("Outputs:")
        logger.info("  • LMC barriers:      %s", barriers_json)
        logger.info("  • Hessian traces:    %s", traces_json)
        logger.info("  • Statistical tests: %s", tests_json)
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
    parsed_args = args.parse_args()

    # run(verbose=parsed_args.verbose)
    run(verbose=True)

# loss_landscape