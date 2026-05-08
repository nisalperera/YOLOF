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

import logging
from typing import Dict, List, Tuple

import torch

from yolof.analysis.model_soup import _is_backbone_encoder
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

    for k in state_dicts[0]:
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
    θ_soup = θ̄ + λ · Σ_i τ_i   (uniform scalar λ for all parameters).
    lam=1.0 produces the standard uniform average soup.
    """
    soup: Dict[str, torch.Tensor] = {}
    for k in anchor:
        if torch.is_floating_point(anchor[k]):
            v = anchor[k].float().clone()
            for tau in taus:
                v = v + lam * tau[k].float()
            soup[k] = v.to(anchor[k].dtype)
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