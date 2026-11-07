"""
phase3_greedy_soup.py — Phase 3 (Greedy Soup Experiments)

Three greedy soup experiments, all following Wortsman et al. (2022):

  GA           — Greedy soup over the ENTIRE model.
                 All keys (backbone, encoder, decoder) are uniformly averaged
                 at each candidate step. A candidate is accepted only if it
                 improves validation mAP over the current soup.

  GB           — Greedy soup over the DECODER only.
                 Backbone + encoder are always taken from ingredient 0 (the
                 baseline / best individual model) and never changed.
                 Greedy selection controls only the decoder weights.

  GD_PER_HEAD  — Greedy soup over the three DECODER sub-heads independently.
                 Backbone + encoder are fixed from ingredient 0 throughout.
                 Three sequential coordinate-descent passes are run:
                     Pass 1: optimise cls  head (cls_subnet + cls_score)
                     Pass 2: optimise bbox head (bbox_subnet + bbox_pred)
                     Pass 3: optimise obj  head (object_pred)
                 Each pass runs a standard greedy loop over the k ingredients
                 while the other two heads stay at the result of the previous
                 pass. The backbone/encoder/shared trunk never change.

Algorithm (identical core for all three experiments):
    1. Evaluate every ingredient individually → sort by mAP (descending).
    2. soup_scope ← ingredient[best]
    3. for i in remaining (in descending mAP order):
           candidate_scope = uniform_avg(soup_scope, ingredient_scope[i])
           if mAP(splice candidate into full model) ≥ mAP(current soup):
               soup_scope ← candidate_scope   # accept

Run:
    python -m yolof_soup.experiments.phase3_greedy_soup
    python -m yolof_soup.experiments.phase3_greedy_soup --experiments GA GB
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from tabulate import tabulate
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog

from yolof.utils import _format_duration
from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    DEVICE,
    RESULTS_DIR,
    PHASE2_OUTPUT_DIR,
    _register_datasets,
    build_eval_cfg,
)
from yolof_soup.config.experiment_registry import get_run_specs
from yolof_soup.utils.inference import EvaluateModel
from yolof_soup.utils.checkpoint_utils import load_states, save_checkpoint
from yolof_soup.utils.eval_utils import get_map, extract_per_class_ap, compute_coco_map
from yolof_soup.utils.key_utils import (
    get_decoder_keys,
    get_backbone_encoder_keys,
    split_decoder_subheads,   # returns (cls_keys, bbox_keys, shared_keys)
    merge_subdicts,
    extract_subdict,
    compute_anchor,
)
from yolof_soup.utils.state_dict_utils import assign_state_to_model
from yolof_soup.utils.global_logger import get_logger

from yolof_soup.experiments.greedy_be_tri_head import build_greedy_tri_head_learned_soup

logger: logging.Logger = None  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Sub-head substring groups
# ─────────────────────────────────────────────────────────────────────────────

CLS_HEAD_SUBSTRINGS:  Tuple[str, ...] = ("cls_subnet",  "cls_score")
BBOX_HEAD_SUBSTRINGS: Tuple[str, ...] = ("bbox_subnet", "bbox_pred")
OBJ_HEAD_SUBSTRINGS:  Tuple[str, ...] = ("object_pred",)


def _is_cls_head_key(key: str) -> bool:
    return any(s in key for s in CLS_HEAD_SUBSTRINGS)


def _is_bbox_head_key(key: str) -> bool:
    return any(s in key for s in BBOX_HEAD_SUBSTRINGS)


def _is_obj_head_key(key: str) -> bool:
    return any(s in key for s in OBJ_HEAD_SUBSTRINGS)


def _partition_decoder_keys(
    decoder_keys: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Partition decoder keys into four disjoint groups:
        cls_keys   — cls_subnet + cls_score
        bbox_keys  — bbox_subnet + bbox_pred
        obj_keys   — object_pred
        shared_keys — everything else in the decoder
    """
    cls_keys: List[str] = []
    bbox_keys: List[str] = []
    obj_keys: List[str] = []
    shared_keys: List[str] = []

    for k in decoder_keys:
        if _is_cls_head_key(k):
            cls_keys.append(k)
        elif _is_bbox_head_key(k):
            bbox_keys.append(k)
        elif _is_obj_head_key(k):
            obj_keys.append(k)
        else:
            shared_keys.append(k)

    return cls_keys, bbox_keys, obj_keys, shared_keys


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_avg(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Uniform average of two matching state-dicts (or any sub-dicts sharing the
    same keys).  Non-floating-point tensors are taken from *a*.
    """
    out: Dict[str, torch.Tensor] = {}
    for k in a:
        if k not in b:
            out[k] = a[k].clone()
        elif torch.is_floating_point(a[k]):
            out[k] = (a[k].float() + b[k].float()).mul(0.5).to(a[k].dtype)
        else:
            out[k] = a[k].clone()
    return out


def _state_dict_size_mb(state: Dict[str, torch.Tensor]) -> float:
    return sum(v.element_size() * v.nelement() for v in state.values()) / 1e6


def _evaluate_state(
    state: Dict[str, torch.Tensor],
    cfg: CfgNode,
    dataset: str,
    tag: str,
) -> float:
    """Load state into a fresh model, evaluate on *dataset*, return mAP@[.5:.95]."""
    _register_datasets()
    model = EvaluateModel(cfg, state_dict=state)
    model.to(DEVICE)
    model.eval()
    try:
        mAP = get_map(model.model, cfg, dataset)
        logger.debug("  [eval %s] mAP = %.4f", tag, mAP)
        return mAP
    finally:
        model.to("cpu")
        del model
        torch.cuda.empty_cache()


def _evaluate_state_full(
    full_state: Dict[str, torch.Tensor],
    cfg: CfgNode,
    selection_dataset: str,
    tag: str,
) -> float:
    return _evaluate_state(full_state, cfg, selection_dataset, tag)


def _splice(
    base_state: Dict[str, torch.Tensor],
    scope_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Return a copy of *base_state* with all keys from *scope_state* overwritten.
    Used to evaluate a partial (scoped) candidate in the context of a full model.
    """
    merged = {k: v.clone() for k, v in base_state.items()}
    merged.update({k: v.clone() for k, v in scope_state.items()})
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Core greedy selection loop
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_selection(
    sorted_scope_dicts: List[Dict[str, torch.Tensor]],
    sorted_full_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataset: str,
    scope_tag: str,
) -> Dict[str, torch.Tensor]:
    """
    Core greedy selection loop (Wortsman et al., 2022 Algorithm 1).

    Parameters
    ----------
    sorted_scope_dicts : ingredients' state-dicts restricted to the *scope*
        (the subset of keys being greedily optimised), sorted by descending
        per-ingredient mAP.
    sorted_full_states : full model state-dicts corresponding to each entry in
        sorted_scope_dicts.  Used to evaluate candidates in the full-model
        context (backbone + encoder always present).
    cfg : CfgNode for model construction.
    selection_dataset : dataset name for mAP evaluation.
    scope_tag : short label used in log messages (e.g. "GA", "GB_decoder").

    Returns
    -------
    The greedy-selected scope state-dict (not a full model state-dict).
    """
    assert len(sorted_scope_dicts) == len(sorted_full_states), (
        "scope and full-state lists must have equal length"
    )

    soup_scope = {k: v.clone() for k, v in sorted_scope_dicts[0].items()}
    # The initial "full" evaluation reference is the best single ingredient.
    best_full_state = {k: v.clone() for k, v in sorted_full_states[0].items()}
    best_map = _evaluate_state_full(best_full_state, cfg, selection_dataset,
                                    f"{scope_tag}/seed")

    logger.info(
        "  [%s] Seed (ingredient 0 / best individual): mAP = %.4f",
        scope_tag, best_map,
    )

    n_accepted = 1
    for idx in range(1, len(sorted_scope_dicts)):
        candidate_scope = _uniform_avg(soup_scope, sorted_scope_dicts[idx])
        candidate_full = _splice(best_full_state, candidate_scope)

        mAP = _evaluate_state_full(
            candidate_full, cfg, selection_dataset,
            f"{scope_tag}/step{idx}",
        )

        if mAP >= best_map:
            soup_scope = candidate_scope
            best_full_state = candidate_full
            best_map = mAP
            n_accepted += 1
            logger.info(
                "  [%s] Step %d/%d — ACCEPT  mAP = %.4f  (soup now has %d ingredients)",
                scope_tag, idx, len(sorted_scope_dicts) - 1, mAP, n_accepted,
            )
        else:
            logger.info(
                "  [%s] Step %d/%d — REJECT  mAP = %.4f  (best = %.4f)",
                scope_tag, idx, len(sorted_scope_dicts) - 1, mAP, best_map,
            )

    logger.info(
        "  [%s] Greedy selection complete — accepted %d / %d ingredients, "
        "best mAP = %.4f",
        scope_tag, n_accepted, len(sorted_scope_dicts), best_map,
    )
    return soup_scope


# ─────────────────────────────────────────────────────────────────────────────
# Experiment GA — Greedy soup over the ENTIRE model
# ─────────────────────────────────────────────────────────────────────────────

def build_GA(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataset: str,
) -> Dict[str, torch.Tensor]:
    """
    GA — Greedy soup applied to the ENTIRE model.

    All keys (backbone, encoder, decoder) participate in the uniform average
    at every candidate step.  A candidate is accepted iff it improves
    validation mAP over the current soup.

    Returns
    -------
    Full merged state-dict.
    """
    logger.info("=" * 70)
    logger.info("GA — Greedy Soup: ENTIRE MODEL")
    logger.info("=" * 70)
    logger.info(
        "  Scoring %d ingredients individually to determine seed order...",
        len(ingredient_states),
    )

    # Score all ingredients, sort descending
    scores: List[Tuple[float, int]] = []
    for i, state in enumerate(ingredient_states):
        mAP = _evaluate_state(state, cfg, selection_dataset, f"GA/ingredient_{i}")
        scores.append((mAP, i))
        logger.info("  Ingredient %d: mAP = %.4f", i, mAP)

    scores.sort(key=lambda x: x[0], reverse=True)
    sorted_states = [ingredient_states[i] for _, i in scores]
    logger.info(
        "  Sorted order (best→worst): %s",
        [f"ing{i}({m:.4f})" for m, i in scores],
    )

    # Full-scope greedy: scope == full state
    result = _greedy_selection(
        sorted_scope_dicts=sorted_states,
        sorted_full_states=sorted_states,
        cfg=cfg,
        selection_dataset=selection_dataset,
        scope_tag="GA",
    )

    logger.info(
        "  ✓ GA soup built.  Size: %.2f MB", _state_dict_size_mb(result)
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment GB — Greedy soup over the DECODER only
# ─────────────────────────────────────────────────────────────────────────────

def build_GB(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataset: str,
) -> Dict[str, torch.Tensor]:
    """
    GB — Greedy soup applied to the DECODER only.

    Backbone + encoder are always kept from ingredient 0 (the individual model
    with the highest validation mAP) and are never changed during selection.
    Greedy selection controls only the decoder weights (all decoder keys as a
    single block — no per-head split).

    Returns
    -------
    Full merged state-dict (backbone/encoder from ingredient 0, decoder greedy-merged).
    """
    logger.info("=" * 70)
    logger.info("GB — Greedy Soup: DECODER ONLY (backbone+encoder fixed from best)")
    logger.info("=" * 70)

    # Score ingredients, determine best individual → fixed backbone/encoder
    logger.info("  Scoring %d ingredients to determine best backbone/encoder seed...",
                len(ingredient_states))
    scores: List[Tuple[float, int]] = []
    for i, state in enumerate(ingredient_states):
        mAP = _evaluate_state(state, cfg, selection_dataset, f"GB/ingredient_{i}")
        scores.append((mAP, i))
        logger.info("  Ingredient %d: mAP = %.4f", i, mAP)

    scores.sort(key=lambda x: x[0], reverse=True)
    sorted_states = [ingredient_states[i] for _, i in scores]
    logger.info(
        "  Sorted order (best→worst): %s",
        [f"ing{i}({m:.4f})" for m, i in scores],
    )

    # Fixed backbone+encoder from ingredient 0 (best individual)
    be_keys = get_backbone_encoder_keys(sorted_states[0])
    fixed_be_state = extract_subdict(sorted_states[0], be_keys)
    logger.info(
        "  Fixed backbone+encoder from ingredient %d (mAP=%.4f), %d keys",
        scores[0][1], scores[0][0], len(be_keys),
    )

    # Decoder key scopes per ingredient
    decoder_keys = get_decoder_keys(sorted_states[0])
    logger.info("  Decoder scope: %d keys", len(decoder_keys))

    sorted_decoder_dicts = [
        extract_subdict(s, decoder_keys) for s in sorted_states
    ]

    # Full states for evaluation: fixed BE + current decoder candidate
    def make_full(decoder_scope: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return merge_subdicts(fixed_be_state, decoder_scope)

    sorted_full_for_eval = [make_full(d) for d in sorted_decoder_dicts]

    result_decoder = _greedy_selection(
        sorted_scope_dicts=sorted_decoder_dicts,
        sorted_full_states=sorted_full_for_eval,
        cfg=cfg,
        selection_dataset=selection_dataset,
        scope_tag="GB",
    )

    final_state = merge_subdicts(fixed_be_state, result_decoder)
    logger.info(
        "  ✓ GB soup built.  Size: %.2f MB", _state_dict_size_mb(final_state)
    )
    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# Experiment GD_PER_HEAD — Greedy soup per decoder sub-head
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_pass_for_head(
    head_name: str,
    current_full_state: Dict[str, torch.Tensor],
    sorted_head_dicts: List[Dict[str, torch.Tensor]],
    head_keys: List[str],
    cfg: CfgNode,
    selection_dataset: str,
) -> Dict[str, torch.Tensor]:
    """
    Run one greedy pass for a single decoder sub-head.

    Parameters
    ----------
    head_name : label for logging (e.g. "cls", "bbox", "obj").
    current_full_state : the full model state reflecting the results of all
        previously completed greedy passes.  The pass starts from this state
        and only updates *head_keys*.
    sorted_head_dicts : sub-dicts restricted to *head_keys*, sorted by the
        per-ingredient mAP order established at the start of GD_PER_HEAD.
    head_keys : keys belonging to this sub-head.
    cfg, selection_dataset : evaluation configuration.

    Returns
    -------
    Updated full model state-dict with greedy-selected weights for this head.
    """
    tag = f"GD_PER_HEAD/{head_name}"

    if not head_keys:
        logger.warning(
            "  [%s] No keys found for head '%s' — skipping pass.", tag, head_name
        )
        return current_full_state

    logger.info("  --- GD_PER_HEAD: head = '%s' (%d keys) ---", head_name, len(head_keys))

    # Seed: head from ingredient 0 (best individual, already in current_full_state)
    soup_head = extract_subdict(current_full_state, head_keys)
    best_full_state = {k: v.clone() for k, v in current_full_state.items()}
    best_map = _evaluate_state_full(best_full_state, cfg, selection_dataset,
                                    f"{tag}/seed")
    logger.info("  [%s] Seed mAP = %.4f", tag, best_map)

    n_accepted = 1
    for idx in range(1, len(sorted_head_dicts)):
        candidate_head = _uniform_avg(soup_head, sorted_head_dicts[idx])
        candidate_full = _splice(best_full_state, candidate_head)

        mAP = _evaluate_state_full(
            candidate_full, cfg, selection_dataset, f"{tag}/step{idx}"
        )

        if mAP >= best_map:
            soup_head = candidate_head
            best_full_state = candidate_full
            best_map = mAP
            n_accepted += 1
            logger.info(
                "  [%s] Step %d/%d — ACCEPT  mAP = %.4f  (soup has %d ingredients)",
                tag, idx, len(sorted_head_dicts) - 1, mAP, n_accepted,
            )
        else:
            logger.info(
                "  [%s] Step %d/%d — REJECT  mAP = %.4f  (best = %.4f)",
                tag, idx, len(sorted_head_dicts) - 1, mAP, best_map,
            )

    logger.info(
        "  [%s] Pass complete — accepted %d / %d, best mAP = %.4f",
        tag, n_accepted, len(sorted_head_dicts), best_map,
    )
    return best_full_state


def build_GD_per_head(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataset: str,
) -> Dict[str, torch.Tensor]:
    """
    GD_PER_HEAD — Greedy soup applied per decoder sub-head independently.

    Backbone + encoder are always fixed from ingredient 0 (best individual).
    Three sequential coordinate-descent greedy passes are run over the decoder:

        Pass 1: cls  head (cls_subnet + cls_score)
        Pass 2: bbox head (bbox_subnet + bbox_pred)
        Pass 3: obj  head (object_pred)

    Each pass runs the standard greedy loop over k ingredients while the other
    two heads and backbone/encoder remain at the result of the previous pass.
    The decoder shared trunk (non-head keys) is uniformly averaged once from
    all ingredients and fixed throughout.

    Returns
    -------
    Full merged state-dict.
    """
    logger.info("=" * 70)
    logger.info("GD_PER_HEAD — Greedy Soup: DECODER HEADS INDEPENDENTLY")
    logger.info("  Heads: cls (%s) | bbox (%s) | obj (%s)",
                CLS_HEAD_SUBSTRINGS, BBOX_HEAD_SUBSTRINGS, OBJ_HEAD_SUBSTRINGS)
    logger.info("  Backbone + encoder: fixed from best individual ingredient")
    logger.info("=" * 70)

    # ── Score ingredients, determine ordering ─────────────────────────────────
    logger.info("  Scoring %d ingredients to determine seed + sort order...",
                len(ingredient_states))
    scores: List[Tuple[float, int]] = []
    for i, state in enumerate(ingredient_states):
        mAP = _evaluate_state(state, cfg, selection_dataset,
                               f"GD_PER_HEAD/ingredient_{i}")
        scores.append((mAP, i))
        logger.info("  Ingredient %d: mAP = %.4f", i, mAP)

    scores.sort(key=lambda x: x[0], reverse=True)
    sorted_states = [ingredient_states[i] for _, i in scores]
    logger.info(
        "  Sorted order (best→worst): %s",
        [f"ing{i}({m:.4f})" for m, i in scores],
    )

    # ── Fixed backbone + encoder from ingredient 0 (best individual) ──────────
    be_keys = get_backbone_encoder_keys(sorted_states[0])
    fixed_be = extract_subdict(sorted_states[0], be_keys)
    logger.info(
        "  Fixed backbone+encoder: ingredient %d (mAP=%.4f), %d keys",
        scores[0][1], scores[0][0], len(be_keys),
    )

    # ── Decoder key partition ──────────────────────────────────────────────────
    decoder_keys = get_decoder_keys(sorted_states[0])
    cls_keys, bbox_keys, obj_keys, shared_keys = _partition_decoder_keys(decoder_keys)
    logger.info(
        "  Decoder key partition — cls: %d | bbox: %d | obj: %d | shared trunk: %d",
        len(cls_keys), len(bbox_keys), len(obj_keys), len(shared_keys),
    )

    if not obj_keys:
        logger.warning(
            "  No 'object_pred' keys found — obj head pass will be skipped. "
            "Verify that your YOLOF checkpoint contains an objectness head."
        )

    # ── Uniform-average the shared decoder trunk (fixed) ──────────────────────
    shared_states = [extract_subdict(s, shared_keys) for s in sorted_states]
    fixed_shared = compute_anchor(shared_states) if shared_keys else {}
    if shared_keys:
        logger.info(
            "  Decoder shared trunk (uniform avg of %d ingredients): %d keys",
            len(sorted_states), len(shared_keys),
        )

    # ── Build per-ingredient head sub-dicts (sorted order) ────────────────────
    sorted_cls_dicts  = [extract_subdict(s, cls_keys)  for s in sorted_states]
    sorted_bbox_dicts = [extract_subdict(s, bbox_keys) for s in sorted_states]
    sorted_obj_dicts  = [extract_subdict(s, obj_keys)  for s in sorted_states]

    # ── Initial full state: best ingredient backbone/encoder/decoder ──────────
    current_full = merge_subdicts(
        fixed_be,
        extract_subdict(sorted_states[0], decoder_keys),
    )
    # Overwrite shared trunk with uniform average
    if shared_keys:
        current_full.update(fixed_shared)

    # ── Pass 1: cls head ──────────────────────────────────────────────────────
    current_full = _greedy_pass_for_head(
        head_name="cls",
        current_full_state=current_full,
        sorted_head_dicts=sorted_cls_dicts,
        head_keys=cls_keys,
        cfg=cfg,
        selection_dataset=selection_dataset,
    )

    # ── Pass 2: bbox head ─────────────────────────────────────────────────────
    current_full = _greedy_pass_for_head(
        head_name="bbox",
        current_full_state=current_full,
        sorted_head_dicts=sorted_bbox_dicts,
        head_keys=bbox_keys,
        cfg=cfg,
        selection_dataset=selection_dataset,
    )

    # ── Pass 3: obj head ──────────────────────────────────────────────────────
    current_full = _greedy_pass_for_head(
        head_name="obj",
        current_full_state=current_full,
        sorted_head_dicts=sorted_obj_dicts,
        head_keys=obj_keys,
        cfg=cfg,
        selection_dataset=selection_dataset,
    )

    logger.info(
        "  ✓ GD_PER_HEAD soup built.  Size: %.2f MB",
        _state_dict_size_mb(current_full),
    )
    return current_full


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_condition(
    state: Dict[str, torch.Tensor],
    cfg: CfgNode,
    eval_dataset: str,
    tag: str,
) -> Dict[str, Any]:
    """
    Full COCO evaluation of *state* on *eval_dataset*.

    Returns
    -------
    dict with keys: mAP50_95, mAP50, AR100, per_class_ap
    """
    logger.info("Evaluating condition '%s' on eval split...", tag)
    _register_datasets()
    model = EvaluateModel(cfg, state_dict=state)
    model.to(DEVICE)
    model.eval()
    try:
        results_dict = compute_coco_map(
            model, cfg, eval_dataset,
            output_dir=Path(RESULTS_DIR) / "phase3_greedy_eval" / tag,
        )
        mAP = float(results_dict.get("AP", 0.0))
        mAP50 = float(results_dict.get("AP50", 0.0))
        ar100 = float(results_dict.get("AR-maxDets=100", 0.0))
        per_class  = extract_per_class_ap(
            results_dict,
            MetadataCatalog.get(eval_dataset).thing_classes,
        )
        logger.info(
            "  Condition '%s' — mAP@[.5:.95]=%.4f  mAP50=%.4f  AR100=%.4f",
            tag, mAP, mAP50, ar100,
        )
    except Exception as exc:
        logger.error("Evaluation failed for '%s': %s", tag, str(exc), exc_info=True)
        mAP, mAP50, ar100, per_class = 0.0, 0.0, 0.0, [0.0] * 80
    finally:
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

    return dict(mAP50_95=mAP, mAP50=mAP50, AR100=ar100, per_class_ap=per_class)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

# Experiment registry: name → builder function
GREEDY_EXPERIMENTS: Dict[str, Any] = {
    "GA":          build_GA,
    "GB":          build_GB,
    "GD_PER_HEAD": build_GD_per_head,
    "GD_BE_TRI_HEAD": build_greedy_tri_head_learned_soup,
}


def run(
    verbose: bool = True,
    experiments: Optional[List[str]] = None,
    force_rebuild: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Phase 3 greedy soup experiments.

    Parameters
    ----------
    verbose : enable DEBUG-level logging.
    experiments : list of experiment names to run (default: all three).
    force_rebuild : list of experiment names whose cached checkpoints
        should be ignored and rebuilt from scratch.

    Returns
    -------
    Dict mapping experiment name → evaluation metrics dict.
    """
    start_time = time.perf_counter()

    global logger
    logger = get_logger(
        level=logging.DEBUG if verbose else logging.INFO,
        add_file_handler=True,
    )

    logger.info("=" * 90)
    logger.info("PHASE 3 — GREEDY SOUP EXPERIMENTS")
    logger.info("=" * 90)

    if experiments is None:
        experiments = list(GREEDY_EXPERIMENTS.keys())
    if force_rebuild is None:
        force_rebuild = []

    unknown = set(experiments) - set(GREEDY_EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. "
                         f"Valid: {list(GREEDY_EXPERIMENTS)}")

    results_dir = Path(RESULTS_DIR)
    checkpoint_dir = Path(CHECKPOINT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ingredients ──────────────────────────────────────────────────────
    _register_datasets()
    cfg = build_eval_cfg()
    run_registry = get_run_specs()
    ingredient_runs = [r for r in run_registry if r.role == "ingredient"]
    ingredients_dir = Path(PHASE2_OUTPUT_DIR)

    ingredient_paths = []
    for run_spec in ingredient_runs:
        ckpt_path = Path(ingredients_dir) / f"{run_spec.run_name}/model_best.pth"
        ingredient_paths.append(ckpt_path)

    logger.info("Loading ingredient checkpoints from %s ...", PHASE2_OUTPUT_DIR)
    ingredient_states = load_states(ingredient_paths)
    logger.info("  Loaded %d ingredients.", len(ingredient_states))

    # Determine selection + eval dataset names
    from yolof_soup.config.experiment_config import SELECTION_DATASET, EVAL_DATASET
    selection_dataset = SELECTION_DATASET
    eval_dataset = EVAL_DATASET

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results: Dict[str, Any] = {}
    summary_rows: List[List] = []

    for exp_name in experiments:
        logger.info("")
        logger.info("▶ Running experiment: %s", exp_name)

        ckpt_path = checkpoint_dir / f"greedy_{exp_name.lower()}.pth"
        result_path = results_dir / f"greedy_{exp_name.lower()}_metrics.json"

        # Use cached checkpoint if it exists and rebuild is not forced
        if ckpt_path.exists() and exp_name not in force_rebuild:
            logger.info(
                "  Found cached checkpoint for %s at %s — loading.",
                exp_name, ckpt_path,
            )
            soup_state = torch.load(str(ckpt_path), map_location="cpu")
        else:
            builder = GREEDY_EXPERIMENTS[exp_name]
            soup_state = builder(ingredient_states, cfg, selection_dataset)

            logger.info("  Saving %s checkpoint → %s", exp_name, ckpt_path)
            save_checkpoint(str(ckpt_path), soup_state)

        # ── Evaluate ─────────────────────────────────────────────────────────
        metrics = _evaluate_condition(soup_state, cfg, eval_dataset, exp_name)
        all_results[exp_name] = metrics

        # Persist per-experiment JSON
        with open(result_path, "w") as f:
            json.dump(
                {
                    "experiment": exp_name,
                    "timestamp": datetime.now().isoformat(),
                    **{k: v for k, v in metrics.items() if k != "per_class_ap"},
                },
                f, indent=2,
            )

        summary_rows.append([
            exp_name,
            f"{metrics['mAP50_95']:.4f}",
            f"{metrics['mAP50']:.4f}",
            f"{metrics['AR100']:.4f}",
        ])

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("GREEDY SOUP — RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "\n%s",
        tabulate(
            summary_rows,
            headers=["Experiment", "mAP@[.5:.95]", "mAP@.50", "AR@100"],
            tablefmt="github",
        ),
    )

    # ── Save combined results JSON ────────────────────────────────────────────
    combined_path = results_dir / "phase3_greedy_results.json"
    with open(combined_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "experiments": {
                    name: {k: v for k, v in m.items() if k != "per_class_ap"}
                    for name, m in all_results.items()
                },
            },
            f, indent=2,
        )
    logger.info("Combined results saved to %s", combined_path)
    logger.info(
        "Total elapsed: %s", _format_duration(time.perf_counter() - start_time)
    )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 greedy soup experiments: GA, GB, GD_PER_HEAD"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(GREEDY_EXPERIMENTS.keys()),
        default=list(GREEDY_EXPERIMENTS.keys()),
        help="Which experiments to run (default: all three).",
    )
    parser.add_argument(
        "--force-rebuild",
        nargs="*",
        default=[],
        metavar="EXP",
        help="Force rebuilding cached checkpoints for these experiments.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    parsed_args = parser.parse_args()

    run(
        verbose=parsed_args.verbose,
        experiments=parsed_args.experiments,
        force_rebuild=parsed_args.force_rebuild,
    )
