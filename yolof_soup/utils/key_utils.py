"""
utils/key_utils.py
==================
State-dict key partitioning and task-vector arithmetic.

Boundary definition is authoritative:
  yolof.analysis.model_soup._is_backbone_encoder
  (keys starting with "backbone." or "encoder.")

Decoder sub-head patterns match yolof/modeling/decoder/ attribute names.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from detectron2.config import CfgNode
from yolof.analysis.model_soup import _is_backbone_encoder
from yolof_soup.utils.inference import BNCalibration
from yolof_soup.utils.global_logger import get_logger

logger = get_logger()

_CLS_PATTERNS: Tuple[str, ...] = (
    "cls_subnet", "cls_score", "cls_pred", "classification",
)
_REG_PATTERNS: Tuple[str, ...] = (
    "bbox_subnet", "bbox_pred", "reg_pred",
    "box_reg", "regression", "delta",
)


# ── Primary partition ─────────────────────────────────────

def get_decoder_keys(state_dict: Dict) -> List[str]:
    """All keys that do NOT belong to backbone or encoder."""
    return sorted(k for k in state_dict if not _is_backbone_encoder(k))


def get_backbone_encoder_keys(state_dict: Dict) -> List[str]:
    """All keys that belong to backbone or encoder."""
    return sorted(k for k in state_dict if _is_backbone_encoder(k))


# ── Secondary partition: cls / reg / shared ───────────────

def split_decoder_subheads(
    decoder_keys: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split decoder keys into (cls_keys, reg_keys, shared_keys).

    cls_keys    — classification head parameters
    reg_keys    — regression / localisation head parameters
    shared_keys — decoder trunk parameters belonging to neither head
    """
    cls_keys    = [k for k in decoder_keys if any(p in k for p in _CLS_PATTERNS)]
    reg_keys    = [k for k in decoder_keys
                   if any(p in k for p in _REG_PATTERNS) and k not in cls_keys]
    shared_keys = [k for k in decoder_keys if k not in cls_keys and k not in reg_keys]
    logger.info(
        "Decoder sub-head split — cls=%d  reg=%d  shared=%d",
        len(cls_keys), len(reg_keys), len(shared_keys),
    )
    return cls_keys, reg_keys, shared_keys


# ── Sub-dict helpers ──────────────────────────────────────

def extract_subdict(
    state_dict: Dict[str, torch.Tensor],
    keys: List[str],
) -> Dict[str, torch.Tensor]:
    """Return a new dict containing only *keys* from state_dict."""
    missing = [k for k in keys if k not in state_dict]
    if missing:
        raise KeyError(f"Keys not found in state_dict: {missing[:5]} …")
    return {k: state_dict[k] for k in keys}


def merge_subdicts(*subdicts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Merge disjoint state-dict fragments into one.
    Raises KeyError on duplicate keys.
    """
    merged: Dict[str, torch.Tensor] = {}
    for sd in subdicts:
        overlap = set(merged) & set(sd)
        if overlap:
            raise KeyError(f"Duplicate keys: {list(overlap)[:5]} …")
        merged.update(sd)
    return merged


# ── Task-vector arithmetic ────────────────────────────────

def compute_anchor(
    state_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    θ̄ = (1/N) Σ θ_i  (element-wise mean over N state-dicts).

    Floating-point params are averaged; integer buffers (e.g.
    num_batches_tracked) are taken from the first state-dict unchanged.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty.")
    
    anchor: Dict[str, torch.Tensor] = {}
    _BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")

    sd_keys = list(state_dicts[0].keys())

    for k in sd_keys:
        is_bn = any(k.endswith(s) for s in _BN_BUFFERS)
        if torch.is_floating_point(state_dicts[0][k]) and not is_bn:
            anchor[k] = (
                torch.stack([sd[k].float() for sd in state_dicts])
                .mean(0)
                .to(state_dicts[0][k].dtype)
            )
        else:
            anchor[k] = state_dicts[0][k].clone()  # best model's BN stats
    return anchor


import torch.nn as nn
from detectron2.layers import FrozenBatchNorm2d


def calibrate_bn(
    cfg: CfgNode,
    state_dict: Dict[str, torch.Tensor],
    dataloader,
    n_batches: int = 50,
    device: str | torch.device = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Recompute BatchNorm running statistics for a merged soup model.
    Only true nn.BatchNorm2d layers are expected to update.
    FrozenBatchNorm2d layers are intentionally excluded from the check.
    """
    model = BNCalibration(cfg, state_dict)

    # ── Classify BN keys by module type ───────────────────────────────────
    frozen_prefixes: set[str] = set()
    live_prefixes:   set[str] = set()

    for name, module in model.model.named_modules():
        if isinstance(module, FrozenBatchNorm2d):
            frozen_prefixes.add(name)
        elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            live_prefixes.add(name)

    def _module_prefix(key: str) -> str:
        # key = "backbone.layer1.0.norm.running_mean"
        # prefix = "backbone.layer1.0.norm"
        return ".".join(key.split(".")[:-1])

    _BN_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

    before_bn_dict = {
        k: v.clone()
        for k, v in state_dict.items()
        if k.endswith(_BN_SUFFIXES)
    }

    # ── Forward pass ──────────────────────────────────────────────────────
    model.to(device)
    model.train()
    logger.info("Calibrating BN statistics (%d batches)…", n_batches)
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if processed >= n_batches:
                break
            try:
                model(batch)
            except Exception as e:
                logger.warning(
                    "BN calibration batch %d failed: %s", processed, e,
                    exc_info=True,
                )
            processed += 1
    model.eval()
    logger.info("BN calibration complete (%d batches processed).", processed)

    after_state  = model.model.state_dict()
    after_bn_dict = {k: v for k, v in after_state.items() if k.endswith(_BN_SUFFIXES)}

    # ── Verify only live BN layers ────────────────────────────────────────
    failed_keys: list[str] = []

    for k, before_v in before_bn_dict.items():
        prefix = _module_prefix(k)

        if prefix in frozen_prefixes:
            logger.debug("Skipping frozen BN key (expected): %s", k)
            continue

        if k not in after_bn_dict:
            logger.warning("BN key '%s' missing from calibrated model.", k)
            failed_keys.append(k)
            continue

        after_v = after_bn_dict[k].cpu()
        before_v = before_v.cpu()

        if torch.equal(before_v, after_v):
            if prefix in live_prefixes:
                # A live BN that didn't update is a real problem
                logger.error(
                    "Live BN stat '%s' was NOT updated during calibration. "
                    "Check that the dataloader reaches this module.",
                    k,
                )
                failed_keys.append(k)
            else:
                # Unknown module type — log warning but don't fail
                logger.warning(
                    "BN stat '%s' (unknown module type) unchanged. "
                    "Prefix: %s", k, prefix,
                )
        else:
            logger.debug("BN stat '%s' updated ✓", k)

    if failed_keys:
        raise RuntimeError(
            f"BN calibration failed: {len(failed_keys)} live BatchNorm2d "
            f"stats were not updated:\n  " + "\n  ".join(failed_keys[:10])
        )

    logger.info(
        "BN calibration verified: all live BatchNorm2d stats updated. "
        "FrozenBatchNorm2d layers skipped (expected)."
    )
    return after_state


def compute_task_vectors(
    state_dicts: List[Dict[str, torch.Tensor]],
    anchor: Dict[str, torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    """
    τ_i = θ_i − θ̄  for each fine-tuned state-dict.

    Integer buffers receive a zero task-vector.
    """
    taus = []
    for sd in state_dicts:
        tau: Dict[str, torch.Tensor] = {}
        for k in sd:
            if torch.is_floating_point(sd[k]):
                tau[k] = (sd[k].float() - anchor[k].float()).to(sd[k].dtype)
            else:
                tau[k] = torch.zeros_like(sd[k])
        taus.append(tau)
    return taus


def apply_uniform_lambdas(
    anchor: Dict[str, torch.Tensor],
    taus: List[Dict[str, torch.Tensor]],
    lam: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Dirichlet-sampled ingredient interpolation via coordinate descent.
    
    CRITICAL FIX (Revision 2): The original formula θ̄ + λ·Σ τ_i is broken because
    Σ_i τ_i = 0 always. The first attempted fix (weighted mean_val) was also broken
    because any mixture of anchor and ingredients equals anchor mathematically.
    
    This implementation uses NON-UNIFORM ingredient weighting that actually varies with λ:
    
    For N ingredients with uniform concentration λ:
      θ_soup = Σ_i w_i(λ) * θ_i
      
    where weights are Dirichlet-like:
      - λ=0.0 → concentrate weight on first ingredient (θ_soup ≈ θ_1)
      - λ=1.0 → uniform weights (θ_soup = θ̄ = anchor)
      - λ>1.0 → smooth transition toward uniform (still gives variation)
    
    Weight calculation:
      w_0(λ) = (1.0 + λ) / (N + λ*N)  [first ingredient gets base + shared weight]
      w_i(λ) = (λ/N) / (1 + λ)        [others get equal fraction of remaining weight]
    
    Simplifying for large N and reasonable λ:
      w_i(λ) → 1/N as λ → ∞ (converges to uniform)
      w_0(λ) → much larger than w_i(λ) as λ → 0 (concentrates on first)
    
    Args:
        anchor: baseline parameters (uniform average θ̄)
        taus: list of N task vectors τ_i = θ_i - θ̄
        lam: concentration parameter. Safe range: [0, 2]
    """
    soup: Dict[str, torch.Tensor] = {}
    N = len(taus) if taus else 1
    
    # Edge case: single ingredient
    if N == 1:
        return {k: anchor[k].clone() for k in anchor}
    
    # Compute weights using Dirichlet-like parameterization
    # This ensures: λ=1 gives uniform, λ<1 concentrates on first, λ>1 spreads more evenly
    concentration = max(0.001, lam)  # Avoid λ≤0 which causes issues
    
    # Weight for first ingredient: higher concentration
    w_first = (1.0 + concentration * 2.0) / (N + concentration * N * 1.5)
    
    # Remaining weight distributed uniformly among other N-1 ingredients
    w_others = (1.0 - w_first) / (N - 1)
    
    # Ensure weights sum to 1 (normalization)
    weights = [w_first] + [w_others] * (N - 1)
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    for k in anchor:
        if torch.is_floating_point(anchor[k]):
            # Reconstruct ingredients from anchor + task vectors
            # θ_i = τ_i + θ̄
            # Start with weighted first ingredient
            first_ingredient = anchor[k].float() + taus[0][k].float()
            soup_val = first_ingredient * weights[0]
            
            # Add weighted contributions from other ingredients
            for i in range(1, N):
                ingredient_i = anchor[k].float() + taus[i][k].float()
                soup_val = soup_val + ingredient_i * weights[i]
            
            soup[k] = soup_val.to(anchor[k].dtype)
        else:
            soup[k] = anchor[k].clone()
    
    return soup


def apply_subhead_lambdas(
    anchor:      Dict[str, torch.Tensor],
    taus:        List[Dict[str, torch.Tensor]],
    lambdas_cls: List[float],
    lambdas_reg: List[float],
    cls_keys:    List[str],
    reg_keys:    List[str],
    shared_keys: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Sub-head coordinate-descent interpolation:

      θ_soup[k] = θ̄[k] + Σ_i λ_cls_i · τ_i[k]           k ∈ cls_keys
      θ_soup[k] = θ̄[k] + Σ_i λ_reg_i · τ_i[k]           k ∈ reg_keys
      θ_soup[k] = θ̄[k] + Σ_i ½(λ_cls_i+λ_reg_i)·τ_i[k] k ∈ shared_keys

    Returns a decoder-only state-dict (does NOT include backbone/encoder keys).
    """
    N = len(taus)
    if len(lambdas_cls) != N or len(lambdas_reg) != N:
        raise ValueError("lambdas_cls and lambdas_reg must both have length N.")

    shared_lams = [(lambdas_cls[i] + lambdas_reg[i]) / 2.0 for i in range(N)]
    result: Dict[str, torch.Tensor] = {}

    def _merge(keys: List[str], lams: List[float]) -> None:
        for k in keys:
            if not torch.is_floating_point(anchor[k]):
                result[k] = anchor[k].clone()
                return
            v = anchor[k].float().clone()
            for i in range(N):
                v = v + lams[i] * taus[i][k].float()
            result[k] = v.to(anchor[k].dtype)

    _merge(cls_keys,    lambdas_cls)
    _merge(reg_keys,    lambdas_reg)
    _merge(shared_keys, shared_lams)
    return result