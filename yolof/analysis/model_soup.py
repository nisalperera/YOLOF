"""
Model Soup Builder
==================
Constructs uniform-average soups from a list of YOLOF checkpoints.

Two soup strategies are supported:

  "full"
      Average ALL parameters (backbone + encoder + decoder).
      Used as a reference baseline.

  "backbone_encoder"
      Average ONLY the backbone + encoder parameters.
      The decoder is left at the state of the *first* checkpoint
      (it will be re-initialised anyway when used with D1/D2/C3).
      This is the soup that feeds into the decoder-only fine-tune
      experiments (RQ3 / RQ4).

Usage::

    from yolof.analysis.model_soup import build_soup, BACKBONE_ENCODER_PREFIXES

    soup_sd = build_soup(
        checkpoint_paths=["output/L1/.../model_best.pth",
                          "output/L2/.../model_best.pth",
                          "output/L3/.../model_best.pth",
                          "output/L4/.../model_best.pth"],
        strategy="backbone_encoder",   # or "full"
    )
    torch.save({"model": soup_sd}, "output/soups/backbone_encoder_L1-L4.pth")
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch

from yolof.analysis.mode_connectivity import load_checkpoint_state_dict

logger = logging.getLogger(__name__)

# Parameter-name prefixes that belong to the backbone and encoder.
# Adjust if your model's attribute names differ.
BACKBONE_ENCODER_PREFIXES: tuple = (
    "backbone.",
    "encoder.",
)

SoupStrategy = Literal["full", "backbone_encoder", "branch_uniform"]

_CLS_PATTERNS = (
    "cls_subnet",
    "cls_score",
    "cls_pred",
    "classification",
)
_REG_PATTERNS = (
    "bbox_subnet",
    "bbox_pred",
    "reg_pred",
    "box_reg",
    "regression",
    "delta",
    "object_pred",
)


def _is_backbone_encoder(key: str) -> bool:
    return any(key.startswith(p) for p in BACKBONE_ENCODER_PREFIXES)


def _is_cls_key(key: str) -> bool:
    return any(p in key for p in _CLS_PATTERNS)


def _is_reg_key(key: str) -> bool:
    return any(p in key for p in _REG_PATTERNS)


def _branch_group_for_key(key: str) -> str:
    if _is_backbone_encoder(key):
        return "backbone_encoder"
    if _is_cls_key(key):
        return "cls"
    if _is_reg_key(key):
        return "reg"
    return "shared"


def uniform_branch_coefficients(n_models: int) -> Dict[str, List[float]]:
    """Uniform simplex coefficients for branch-uniform soup."""
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    weights = [1.0 / n_models] * n_models
    return {
        "cls": weights,
        "reg": weights,
        "shared": weights,
        "backbone_encoder": weights,
    }


def normalize_simplex(values: List[float]) -> List[float]:
    """Normalize positive values onto a simplex. Falls back to uniform."""
    if not values:
        raise ValueError("values must be non-empty")
    clamped = [max(float(v), 0.0) for v in values]
    total = sum(clamped)
    if total <= 0:
        return [1.0 / len(clamped)] * len(clamped)
    return [v / total for v in clamped]


def fisher_branch_coefficients_from_traces(
    cls_traces: List[float],
    reg_traces: List[float],
) -> Dict[str, List[float]]:
    """
    Convert per-model branch traces to simplex-constrained branch coefficients.

    This uses trace-proportional weighting for cls/reg and uniform weights for
    shared and backbone_encoder blocks.
    """
    if len(cls_traces) != len(reg_traces):
        raise ValueError("cls_traces and reg_traces must have same length")
    n = len(cls_traces)
    if n == 0:
        raise ValueError("trace lists must be non-empty")

    uniform = [1.0 / n] * n
    return {
        "cls": normalize_simplex(cls_traces),
        "reg": normalize_simplex(reg_traces),
        "shared": uniform,
        "backbone_encoder": uniform,
    }


def build_branch_weighted_soup_from_states(
    state_dicts: List[Dict[str, torch.Tensor]],
    branch_weights: Dict[str, List[float]],
) -> Dict[str, torch.Tensor]:
    """Build soup from already-loaded state dicts using branch-specific weights."""
    if len(state_dicts) < 2:
        raise ValueError("Need at least two state dicts to build a soup.")

    n = len(state_dicts)
    for group in ("cls", "reg", "shared", "backbone_encoder"):
        if group not in branch_weights:
            raise ValueError(f"Missing branch weights for group '{group}'")
        if len(branch_weights[group]) != n:
            raise ValueError(
                f"Weights for group '{group}' must have length {n}, "
                f"got {len(branch_weights[group])}"
            )

    soup: Dict[str, torch.Tensor] = {}
    for k, v0 in state_dicts[0].items():
        if not torch.is_floating_point(v0):
            soup[k] = v0.clone()
            continue

        group = _branch_group_for_key(k)
        w = branch_weights[group]
        acc = state_dicts[0][k].float() * w[0]
        for i in range(1, n):
            acc = acc + state_dicts[i][k].float() * w[i]
        soup[k] = acc.to(v0.dtype)

    return soup


def build_branch_weighted_soup(
    checkpoint_paths: List,
    branch_weights: Dict[str, List[float]],
) -> Dict[str, torch.Tensor]:
    """Load checkpoints and build a branch-weighted soup."""
    state_dicts = [load_checkpoint_state_dict(p) for p in checkpoint_paths]
    return build_branch_weighted_soup_from_states(state_dicts, branch_weights)


def build_soup(
    checkpoint_paths: List,
    strategy: SoupStrategy = "backbone_encoder",
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Construct a weight-averaged soup from ``checkpoint_paths``.

    Args:
        checkpoint_paths: Ordered list of checkpoint .pth files.
        strategy:         "full" averages all params.
                          "backbone_encoder" averages only backbone + encoder
                          params; decoder comes from the first checkpoint.
                          "branch_uniform" averages backbone+encoder and
                          decoder cls/reg/shared groups with separate uniform
                          branch partitions.
        weights:          Optional per-model weights (sum-to-1).  If None,
                          uniform averaging is used.

    Returns:
        A state-dict ready to be passed to ``torch.save({"model": ...})``.
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least two checkpoints to build a soup.")

    if strategy not in ("full", "backbone_encoder", "branch_uniform"):
        raise ValueError(f"Unsupported soup strategy: {strategy}")

    n = len(checkpoint_paths)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("len(weights) must equal len(checkpoint_paths).")
        total = sum(weights)
        weights = [w / total for w in weights]

    logger.info(
        "[ModelSoup] Building %s soup from %d checkpoints (strategy=%s)",
        "uniform" if all(w == weights[0] for w in weights) else "weighted",
        n, strategy,
    )

    state_dicts = [load_checkpoint_state_dict(p) for p in checkpoint_paths]

    # Start from the first checkpoint as the base (full copy)
    soup: Dict[str, torch.Tensor] = {
        k: v.clone().float() * weights[0]
        for k, v in state_dicts[0].items()
    }

    branch_weights: Optional[Dict[str, List[float]]] = None
    if strategy == "branch_uniform":
        branch_weights = uniform_branch_coefficients(n)
        soup = {
            k: v.clone().float() * branch_weights[_branch_group_for_key(k)][0]
            for k, v in state_dicts[0].items()
        }

    for i, (sd, w) in enumerate(zip(state_dicts[1:], weights[1:]), start=1):
        for k, v in sd.items():
            if k not in soup:
                logger.warning("[ModelSoup] Key %s missing in checkpoint 0; skipping.", k)
                continue

            if strategy == "backbone_encoder" and not _is_backbone_encoder(k):
                # Non-backbone/encoder params: keep from checkpoint 0 only
                continue

            if torch.is_floating_point(v):
                if strategy == "branch_uniform" and branch_weights is not None:
                    group = _branch_group_for_key(k)
                    soup[k] = soup[k] + v.float() * branch_weights[group][i]
                else:
                    soup[k] = soup[k] + v.float() * w
            # Integer buffers (e.g. num_batches_tracked) stay as-is from ckpt 0

    # Convert back to original dtypes
    for k in soup:
        orig_dtype = state_dicts[0][k].dtype
        soup[k] = soup[k].to(orig_dtype)

    logger.info("[ModelSoup] Soup built -- %d parameters.", len(soup))
    return soup


def save_soup(
    soup: Dict[str, torch.Tensor],
    output_path,
    metadata: Optional[dict] = None,
) -> None:
    """Save soup state-dict in YOLOF checkpoint format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"model": soup}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, str(output_path))
    logger.info("[ModelSoup] Saved -> %s", output_path)
