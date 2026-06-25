"""
utils/checkpoint_utils.py
==========================
Checkpoint loading / saving, wired to nisalperera/YOLOF
feature/layer-freezing checkpoint format:  {"model": state_dict, ...}

Wraps:
  yolof.analysis.mode_connectivity.load_checkpoint_state_dict
  yolof.analysis.model_soup.save_soup
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from yolof.analysis.mode_connectivity import load_checkpoint_state_dict
from yolof.analysis.model_soup import save_soup
from yolof_soup.utils.global_logger import get_logger


logger = get_logger(logging.DEBUG, add_file_handler=True)

# ── Loading ───────────────────────────────────────────────

def load_state(path: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load a YOLOF checkpoint and return its bare state-dict (CPU tensors).

    Handles {"model": sd}, {"model": sd, "metadata": ...},
    and raw state-dict dicts.
    """
    return load_checkpoint_state_dict(path)


def load_states(paths: List[str | Path]) -> List[Dict[str, torch.Tensor]]:
    """Load an ordered list of checkpoints; return state-dicts in the same order."""
    states = []
    for p in paths:
        logger.info("Loading checkpoint: %s", p)
        states.append(load_state(p))
    return states


def load_metadata(path: str | Path) -> Optional[Dict[str, Any]]:
    """Return the optional 'metadata' dict from a thesis soup checkpoint."""
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        return ckpt.get("metadata", None)
    return None


# ── Saving ────────────────────────────────────────────────

def save_checkpoint(
    path: str | Path,
    state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a state-dict in YOLOF format {"model": ...}.
    Parent directories are created automatically.
    Delegates to save_soup so the format is byte-identical to all
    other thesis soup checkpoints.
    """
    save_soup(state_dict, path, metadata=metadata)
    logger.info("Checkpoint saved → %s", path)


def save_ingredients(
    paths: List[str | Path],
    state_dicts: List[Dict[str, torch.Tensor]],
    metadatas: Optional[List[Optional[Dict]]] = None,
) -> None:
    """Bulk-save a list of state-dicts to their corresponding paths."""
    if len(paths) != len(state_dicts):
        raise ValueError("paths and state_dicts must have the same length.")
    metas = metadatas or [None] * len(paths)
    for p, sd, m in zip(paths, state_dicts, metas):
        save_checkpoint(p, sd, metadata=m)