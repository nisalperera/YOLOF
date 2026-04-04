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

SoupStrategy = Literal["full", "backbone_encoder"]


def _is_backbone_encoder(key: str) -> bool:
    return any(key.startswith(p) for p in BACKBONE_ENCODER_PREFIXES)


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
        weights:          Optional per-model weights (sum-to-1).  If None,
                          uniform averaging is used.

    Returns:
        A state-dict ready to be passed to ``torch.save({"model": ...})``.
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least two checkpoints to build a soup.")

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

    for i, (sd, w) in enumerate(zip(state_dicts[1:], weights[1:]), start=1):
        for k, v in sd.items():
            if k not in soup:
                logger.warning("[ModelSoup] Key %s missing in checkpoint 0; skipping.", k)
                continue

            if strategy == "backbone_encoder" and not _is_backbone_encoder(k):
                # Non-backbone/encoder params: keep from checkpoint 0 only
                continue

            if torch.is_floating_point(v):
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
