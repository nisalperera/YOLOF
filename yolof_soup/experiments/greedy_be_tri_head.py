# ─────────────────────────────────────────────────────────────────────────────
# M7: Greedy Soup (Backbone + Encoder) + Learned Tri-Head Soup (Decoder)
# ─────────────────────────────────────────────────────────────────────────────
#
# Design rationale
# ─────────────────
# Backbone and encoder are feature-extraction layers: their weight manifold is
# smooth and well-conditioned, making greedy selection (Wortsman et al., 2022)
# the safest merging strategy — only models that provably improve validation
# mAP on the backbone+encoder sub-graph are included.
#
# The decoder heads (cls_subnet / cls_score, bbox_subnet / bbox_pred,
# object_pred) are task-specific prediction layers whose optimal mixing
# coefficients differ across sub-heads.  The tri-head learned soup (M6)
# addresses this by maintaining three independent (α, β) pairs and running
# exact decoder-only backpropagation via functional_call.
#
# M7 combines both strategies:
#   1. Greedy soup selects the best backbone+encoder weight average from the
#      ingredient pool, using validation mAP as the selection criterion.
#   2. The greedy-selected backbone+encoder is loaded as a fixed skeleton.
#   3. Tri-head learned soup optimises α_cls, α_bbox, α_obj and
#      β_cls, β_bbox, β_obj for the decoder using the exact gradient path
#      through functional_call — identical to M6.
#
# Algorithm (backbone / encoder greedy selection)
# ────────────────────────────────────────────────
#   Input : ingredient state-dicts {θᵢ}ᵢ₌₁…ₙ  sorted by validation mAP ↓
#   Init  : θ̄ ← θ₁  (best single model)
#   For i = 2 … n:
#       θ_candidate ← (θ̄ + θᵢ) / 2           (uniform average of accepted + new)
#       if mAP(backbone_encoder(θ_candidate)) > mAP(backbone_encoder(θ̄)):
#           θ̄ ← θ_candidate                   (accept)
#   Output: θ̄_be  (greedy-selected backbone+encoder weights)
#
# Note: mAP is computed by replacing only the backbone+encoder in a reference
# model skeleton — the decoder uses a fixed uniform-averaged starting point so
# that the greedy signal reflects backbone+encoder quality alone.
#
# Peak GPU during greedy phase: 1 model + 1 eval batch at a time.
# Peak GPU during decoder optimisation: same as M6.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from detectron2.config import CfgNode

from yolof_soup.config.experiment_config import (
    DEVICE,
    SELECTION_DATASET,
    _register_datasets,
    build_eval_cfg,
)
from yolof_soup.utils.inference import EvaluateModel
from yolof_soup.utils.eval_utils import build_eval_dataloader, get_map
from yolof_soup.utils.key_utils import (
    get_backbone_encoder_keys,
    get_decoder_keys,
    extract_subdict,
    merge_subdicts,
)
from yolof_soup.utils.global_logger import get_logger

# Re-used from the existing phase3 module (imported at runtime to avoid circular
# imports and to keep M7 self-contained as a drop-in addition).
from yolof_soup.experiments.soup_construction import (
    # Tri-head helpers — identical to M6 internals
    LEARNED_SOUP_BATCH_SIZE,
    LEARNED_SOUP_BATCH_SIZE,
    _partition_keys,
    _build_uniform_backbone_encoder,
    _blend_decoder_params,
    _extract_encoder_features,
    _split_params_and_buffers,
    _decoder_forward_with_blended_params,
    _mix_states_tri_head,
    _apply_temperature_tri_head,
    setup_logger,
    # Hyper-parameters
    LEARNED_SOUP_LR,
    LEARNED_SOUP_EPOCHS,
    LEARNED_SOUP_PATIENCE,
    LEARNED_SOUP_GRAD_CLIP,
    LEARNED_SOUP_TEMP_MIN,
    LEARNED_SOUP_TEMP_MAX,
    LEARNED_SOUP_ALPHA_THRESHOLD,
    LEARNED_SOUP_ENTROPY_WEIGHT,
    # Sub-head substring groups
    CLS_HEAD_SUBSTRINGS,
    BBOX_HEAD_SUBSTRINGS,
    OBJ_HEAD_SUBSTRINGS,
    # State-dict size helper
    _state_dict_size_mb,
)
from yolof.modeling.yolof import permute_to_N_HWA_K
from torch.func import functional_call  # requires PyTorch >= 2.0

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Greedy Soup — Backbone + Encoder
# ─────────────────────────────────────────────────────────────────────────────

def _sort_ingredients_by_map(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
) -> List[Dict[str, torch.Tensor]]:
    """
    Evaluate each ingredient on SELECTION_DATASET and sort descending by mAP.

    Only the backbone + encoder portion of each ingredient is swapped into a
    shared model skeleton during evaluation so that the ranking reflects
    backbone+encoder quality independently of decoder variance.

    Returns:
        Ingredient state-dicts sorted by validation mAP (highest first).
    """
    logger.info(
        "  [Greedy] Ranking %d ingredients by validation mAP...",
        len(ingredient_states),
    )
    scores: List[tuple[float, int]] = []

    for i, state in enumerate(ingredient_states):
        try:
            _register_datasets()
            model = EvaluateModel(cfg, state_dict=state)
            model.to(DEVICE)
            model.eval()
            map_val = get_map(model, cfg, SELECTION_DATASET)
            logger.info("  [Greedy] Ingredient %d → mAP=%.4f", i, map_val)
            scores.append((map_val, i))
        except Exception as exc:
            logger.warning(
                "  [Greedy] Ingredient %d evaluation failed: %s — scoring as 0.0",
                i, exc,
            )
            scores.append((0.0, i))
        finally:
            if "model" in dir():
                model.to("cpu")
                del model
            torch.cuda.empty_cache()

    scores.sort(key=lambda x: x[0], reverse=True)
    logger.info(
        "  [Greedy] Sorted order: %s",
        [(rank, f"mAP={s:.4f}") for rank, (s, _) in enumerate(scores)],
    )
    return [ingredient_states[idx] for _, idx in scores]


def _eval_backbone_encoder_swap(
    be_state: Dict[str, torch.Tensor],
    decoder_state: Dict[str, torch.Tensor],
    cfg: CfgNode,
    be_keys: List[str],
    decoder_keys: List[str],
) -> float:
    """
    Evaluate a candidate backbone+encoder weight dict against a fixed decoder.

    Builds a full state-dict from *be_state* (backbone+encoder) and
    *decoder_state* (fixed decoder), loads it into a model, evaluates on
    SELECTION_DATASET, and returns mAP@0.5:0.95.

    The decoder is held constant so that the greedy selection signal reflects
    only the quality of the backbone+encoder combination.
    """
    full_state = {**be_state, **decoder_state}
    _register_datasets()
    try:
        model = EvaluateModel(cfg, state_dict=full_state)
        model.to(DEVICE)
        model.eval()
        map_val = get_map(model, cfg, SELECTION_DATASET)
        return map_val
    except Exception as exc:
        logger.warning("  [Greedy] Evaluation error: %s", exc)
        return float("nan")
    finally:
        if "model" in dir():
            model.to("cpu")
            del model
        torch.cuda.empty_cache()


def build_greedy_backbone_encoder(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
) -> tuple[Dict[str, torch.Tensor], List[int]]:
    """
    Greedy soup over backbone and encoder weights only.

    Algorithm (Wortsman et al., 2022 — Algorithm 1, adapted for sub-graph):
        1. Sort ingredients by descending validation mAP.
        2. Initialise θ̄_be with the best individual model's backbone+encoder.
        3. For each remaining ingredient i (in sorted order):
               θ_candidate = mean(θ̄_be, θᵢ_be)
               if mAP(θ_candidate | fixed_decoder) > mAP(θ̄_be | fixed_decoder):
                   θ̄_be ← θ_candidate       [accept]
               else:
                   skip ingredient i         [reject]
        4. Return θ̄_be.

    The fixed_decoder used for evaluation is the uniform average of ALL
    ingredient decoders (computed once before the loop).  This ensures that
    the greedy signal is backbone+encoder-specific rather than confounded by
    decoder quality.

    Args:
        ingredient_states: Fine-tuned model state-dicts (unsorted).
        cfg:               Detectron2 CfgNode.

    Returns:
        (greedy_be_state, accepted_indices)
            greedy_be_state — merged backbone+encoder state dict.
            accepted_indices — indices (in sorted order) of accepted ingredients.
    """
    logger.info("Building Greedy Backbone+Encoder Soup...")
    n = len(ingredient_states)

    # ── Step 1: Sort ingredients by validation mAP ────────────────────────────
    sorted_states = _sort_ingredients_by_map(ingredient_states, cfg)

    # ── Identify backbone+encoder and decoder key sets ────────────────────────
    be_keys = get_backbone_encoder_keys(sorted_states[0])
    decoder_keys = get_decoder_keys(sorted_states[0])

    # ── Build fixed uniform decoder for isolated backbone+encoder evaluation ──
    logger.info("  [Greedy] Building fixed uniform decoder for evaluation isolation...")
    decoder_states_list = [extract_subdict(s, decoder_keys) for s in sorted_states]
    uniform_decoder = {}
    for key in decoder_keys:
        mixed: Optional[torch.Tensor] = None
        for s in decoder_states_list:
            p = s[key].float()
            mixed = p / n if mixed is None else mixed + p / n
        if mixed is not None:
            uniform_decoder[key] = mixed.to(sorted_states[0][key].dtype)
        else:
            uniform_decoder[key] = sorted_states[0][key].clone()

    # ── Step 2: Initialise with best individual model ─────────────────────────
    current_be = extract_subdict(sorted_states[0], be_keys)
    best_map = _eval_backbone_encoder_swap(
        current_be, uniform_decoder, cfg, be_keys, decoder_keys
    )
    accepted_indices = [0]
    logger.info(
        "  [Greedy] Initialised with ingredient 0 (sorted) → mAP=%.4f", best_map
    )

    # ── Step 3: Greedily add remaining ingredients ────────────────────────────
    for i in range(1, len(sorted_states)):
        candidate_be_i = extract_subdict(sorted_states[i], be_keys)

        # Average current greedy soup backbone+encoder with candidate i
        candidate_be: Dict[str, torch.Tensor] = {}
        n_accepted = len(accepted_indices)
        for key in be_keys:
            a = current_be[key].float()
            b = candidate_be_i[key].float()
            # Running mean: new_mean = (n * old_mean + b) / (n + 1)
            candidate_be[key] = (
                ((a * n_accepted) + b) / (n_accepted + 1)
            ).to(current_be[key].dtype)

        candidate_map = _eval_backbone_encoder_swap(
            candidate_be, uniform_decoder, cfg, be_keys, decoder_keys
        )
        logger.info(
            "  [Greedy] Candidate with ingredient %d → mAP=%.4f  (current best=%.4f)",
            i, candidate_map, best_map,
        )

        if not (candidate_map != candidate_map):  # nan-safe check
            if candidate_map > best_map:
                current_be = candidate_be
                best_map = candidate_map
                accepted_indices.append(i)
                logger.info(
                    "  [Greedy] ✓ Ingredient %d accepted (mAP improved to %.4f)",
                    i, best_map,
                )
            else:
                logger.info(
                    "  [Greedy] ✗ Ingredient %d rejected (mAP %.4f ≤ %.4f)",
                    i, candidate_map, best_map,
                )
        else:
            logger.warning(
                "  [Greedy] Ingredient %d returned NaN mAP — rejected",
                i,
            )

    logger.info(
        "  [Greedy] ✓ Greedy backbone+encoder soup complete. "
        "Accepted %d / %d ingredients. Final mAP=%.4f. Size=%.2f MB",
        len(accepted_indices),
        len(sorted_states),
        best_map,
        _state_dict_size_mb(current_be),
    )
    return current_be, accepted_indices


# ─────────────────────────────────────────────────────────────────────────────
# M7 — Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def build_greedy_tri_head_learned_soup(
    ingredient_states: List[Dict[str, torch.Tensor]],
    cfg: CfgNode,
    selection_dataset,
) -> Dict[str, torch.Tensor]:
    """
    M7: Greedy Soup (Backbone + Encoder) + Learned Tri-Head Soup (Decoder).

    Combines two complementary merging strategies:

        Backbone + Encoder:
            Greedy weight selection (Wortsman et al., 2022 Algorithm 1).
            Sorted by validation mAP; only ingredients that improve the running
            average are accepted.  Produces a single fixed backbone+encoder
            state used as the skeleton for decoder optimisation.

        Decoder (cls_subnet / cls_score, bbox_subnet / bbox_pred, object_pred):
            Tri-head learned soup identical to M6.
            Three independent (α, β) pairs are optimised end-to-end via exact
            gradient flow through the decoder using functional_call.  The
            greedy-selected backbone+encoder is frozen throughout.

    Learnable parameters (6 leaf tensors — decoder only):
        alpha_cls_raw  (k,) — unconstrained logits → softmax → α_cls
        alpha_bbox_raw (k,) — unconstrained logits → softmax → α_bbox
        alpha_obj_raw  (k,) — unconstrained logits → softmax → α_obj
        log_beta_cls   ()   — unconstrained scalar  → exp    → β_cls
        log_beta_bbox  ()   — unconstrained scalar  → exp    → β_bbox
        log_beta_obj   ()   — unconstrained scalar  → exp    → β_obj

    Gradient strategy (decoder optimisation):
        1. Load greedy backbone+encoder into model skeleton (frozen).
        2. Per batch:
           a. backbone+encoder forward under no_grad → detached feature z.
           b. Build blended decoder params in-graph via α vectors.
              β values applied inline to prediction-layer tensors.
           c. Decoder forward via functional_call — exact gradient path.
           d. Detection loss backward — gradients for all 6 leaf tensors.
        3. Early stopping on validation loss plateau.

    Peak GPU = 1 model skeleton + 1 batch + decoder activations only.

    Args:
        ingredient_states:    List of k fine-tuned model state-dicts.
        cfg:                  Detectron2 / CfgNode model configuration.
        selection_dataloader: Held-out validation dataloader for decoder
                              optimisation (same split used for greedy eval).

    Returns:
        Merged state dict:
            backbone + encoder — greedy-selected uniform average.
            cls / bbox / obj heads — tri-head learned mixing with learned α, β.
    """

    setup_logger()

    logger.info("=" * 80)
    logger.info(
        "Building Condition 7 (M7): Greedy Backbone+Encoder + "
        "Tri-Head Learned Decoder Soup"
    )
    logger.info("=" * 80)
    logger.info(
        "  → Phase A: Greedy soup for backbone + encoder "
        "(Wortsman et al. 2022, Alg. 1)"
    )
    logger.info(
        "  → Phase B: Tri-head learned soup for decoder "
        "(exact grad via functional_call)"
    )

    n_ingredients = len(ingredient_states)
    device = DEVICE

    dataloader = build_eval_dataloader(
            cfg, selection_dataset, batch_size=LEARNED_SOUP_BATCH_SIZE
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A — Greedy Backbone + Encoder
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Phase A: Greedy Backbone + Encoder ──")
    greedy_be_state, accepted_indices = build_greedy_backbone_encoder(
        ingredient_states, cfg
    )
    logger.info(
        "  ✓ Phase A complete — %d / %d ingredients accepted",
        len(accepted_indices), n_ingredients,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B — Tri-Head Learned Decoder Soup
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Phase B: Tri-Head Learned Decoder Soup ──")
    logger.info("  → cls  head keys %s → (alpha_cls,  beta_cls)",  CLS_HEAD_SUBSTRINGS)
    logger.info("  → bbox head keys %s → (alpha_bbox, beta_bbox)", BBOX_HEAD_SUBSTRINGS)
    logger.info("  → obj  head keys %s → (alpha_obj,  beta_obj)",  OBJ_HEAD_SUBSTRINGS)
    logger.info(
        "  → backbone/encoder → greedy-selected fixed weights (no_grad)"
    )

    # ── Partition keys ────────────────────────────────────────────────────────
    cls_keys, bbox_keys, obj_keys, other_keys = _partition_keys(ingredient_states[0])
    logger.info(
        "  → Key partition: %d cls | %d bbox | %d obj | %d backbone/encoder/trunk",
        len(cls_keys), len(bbox_keys), len(obj_keys), len(other_keys),
    )
    if not obj_keys:
        logger.warning(
            "  ⚠ No 'object_pred' keys found — alpha_obj / beta_obj will be "
            "optimised but have no mixing effect. Verify your YOLOF checkpoint."
        )

    # ── Build greedy BE state for model skeleton ──────────────────────────────
    # other_keys = backbone + encoder + shared trunk keys.
    # Replace the uniform average from _partition_keys with the greedy-selected
    # backbone+encoder.  Shared trunk keys that are NOT part of backbone/encoder
    # are still uniformly averaged.
    be_keys_set = set(get_backbone_encoder_keys(ingredient_states[0]))
    trunk_only_keys = [k for k in other_keys if k not in be_keys_set]

    # Uniform average for non-BE trunk keys (if any)
    if trunk_only_keys:
        logger.info(
            "  → %d shared trunk keys (non-backbone/encoder) → uniform 1/%d",
            len(trunk_only_keys), n_ingredients,
        )
        trunk_state: Dict[str, torch.Tensor] = {}
        for key in trunk_only_keys:
            mixed: Optional[torch.Tensor] = None
            for s in ingredient_states:
                p = s[key].float()
                mixed = p / n_ingredients if mixed is None else mixed + p / n_ingredients
            if mixed is not None:
                trunk_state[key] = mixed.to(ingredient_states[0][key].dtype)
            else:
                trunk_state[key] = ingredient_states[0][key].clone()
        skeleton_fixed_state = {**greedy_be_state, **trunk_state}
    else:
        skeleton_fixed_state = greedy_be_state

    logger.info(
        "  → Skeleton fixed state size: %.2f MB",
        _state_dict_size_mb(skeleton_fixed_state),
    )

    # ── Load model skeleton with greedy backbone+encoder ─────────────────────
    _register_datasets()
    model_skeleton = EvaluateModel(cfg, state_dict=ingredient_states[0])
    model_skeleton.to(device)
    model_skeleton.eval()

    for mod in model_skeleton.model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
            mod.eval()

    # Inject greedy backbone+encoder weights into skeleton
    with torch.no_grad():
        for key, val in skeleton_fixed_state.items():
            parts = key.split(".")
            module = model_skeleton.model
            try:
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data.copy_(val.to(device=device, dtype=param.dtype))
                else:
                    setattr(
                        module,
                        parts[-1],
                        val.to(
                            device=device,
                            dtype=param.dtype if hasattr(param, "dtype") else val.dtype,
                        ),
                    )
            except AttributeError as e:
                logger.warning(
                    "  [Skeleton] Could not set key '%s': %s — skipping", key, e
                )

    logger.info("  → Model skeleton loaded with greedy backbone+encoder weights.")

    # ── CPU-pin ingredient decoder params ─────────────────────────────────────
    def _pin(states, keys):
        return [
            {
                k: s[k].cpu().pin_memory() if s[k].is_floating_point() else s[k].cpu()
                for k in keys
                if k in s
            }
            for s in states
        ]

    ingredient_cls_params  = _pin(ingredient_states, cls_keys)
    ingredient_bbox_params = _pin(ingredient_states, bbox_keys)
    ingredient_obj_params  = _pin(ingredient_states, obj_keys)

    # ── Six learnable leaf tensors ────────────────────────────────────────────
    # Initialised to zero → uniform softmax, β = exp(0) = 1.0
    alpha_cls_raw  = torch.zeros(n_ingredients, device=device,
                                 dtype=torch.float32, requires_grad=True)
    alpha_bbox_raw = torch.zeros(n_ingredients, device=device,
                                 dtype=torch.float32, requires_grad=True)
    alpha_obj_raw  = torch.zeros(n_ingredients, device=device,
                                 dtype=torch.float32, requires_grad=True)
    log_beta_cls   = torch.tensor(0.0, device=device,
                                  dtype=torch.float32, requires_grad=True)
    log_beta_bbox  = torch.tensor(0.0, device=device,
                                  dtype=torch.float32, requires_grad=True)
    log_beta_obj   = torch.tensor(0.0, device=device,
                                  dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.AdamW(
        [alpha_cls_raw, alpha_bbox_raw, alpha_obj_raw,
         log_beta_cls,  log_beta_bbox,  log_beta_obj],
        lr=LEARNED_SOUP_LR,
    )

    best_loss: float = float("inf")
    best_alpha_cls_normalized:  Optional[torch.Tensor] = None
    best_alpha_bbox_normalized: Optional[torch.Tensor] = None
    best_alpha_obj_normalized:  Optional[torch.Tensor] = None
    best_log_beta_cls:  Optional[torch.Tensor] = None
    best_log_beta_bbox: Optional[torch.Tensor] = None
    best_log_beta_obj:  Optional[torch.Tensor] = None
    patience_counter: int = 0

    eval_log_freq = 100

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
            epoch_loss   = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(dataloader):

                optimizer.zero_grad(set_to_none=True)

                # Normalise α and exponentiate β (all in autograd graph)
                alpha_cls_norm  = torch.softmax(alpha_cls_raw,  dim=0)
                alpha_bbox_norm = torch.softmax(alpha_bbox_raw, dim=0)
                alpha_obj_norm  = torch.softmax(alpha_obj_raw,  dim=0)
                beta_cls   = torch.exp(log_beta_cls)
                beta_bbox  = torch.exp(log_beta_bbox)
                beta_obj   = torch.exp(log_beta_obj)

                # Step 1 — backbone+encoder forward under no_grad (greedy weights)
                encoder_features = _extract_encoder_features(model_skeleton, batch)

                # Step 2 — blended decoder params (in-graph)
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

                # Step 4 — backward + gradient clip + step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [alpha_cls_raw, alpha_bbox_raw, alpha_obj_raw,
                     log_beta_cls,  log_beta_bbox,  log_beta_obj],
                    max_norm=LEARNED_SOUP_GRAD_CLIP,
                )
                optimizer.step()

                epoch_loss    += loss.item()
                epoch_batches += 1

                # ── Per-batch entropy regularisation diagnostic ───────────────
                if batch_idx > 0 and batch_idx % eval_log_freq == 0:
                    alpha_cls_list  = alpha_cls_norm.detach().tolist()
                    alpha_bbox_list = alpha_bbox_norm.detach().tolist()
                    alpha_obj_list  = alpha_obj_norm.detach().tolist()
                    weak_cls  = [(i, w) for i, w in enumerate(alpha_cls_list)
                                 if w < LEARNED_SOUP_ALPHA_THRESHOLD]
                    weak_bbox = [(i, w) for i, w in enumerate(alpha_bbox_list)
                                 if w < LEARNED_SOUP_ALPHA_THRESHOLD]
                    weak_obj  = [(i, w) for i, w in enumerate(alpha_obj_list)
                                 if w < LEARNED_SOUP_ALPHA_THRESHOLD]

                    logger.debug(
                        "  Epoch %d batch %d: loss=%.4f  avg=%.4f\n"
                        "      α_cls=%s  β_cls=%.4f%s\n"
                        "      α_bbox=%s β_bbox=%.4f%s\n"
                        "      α_obj=%s  β_obj=%.4f%s",
                        epoch + 1, batch_idx,
                        loss.item(), epoch_loss / max(epoch_batches, 1),
                        [f"{a:.4f}" for a in alpha_cls_list],  beta_cls.item(),
                        f" ⚠ weak={weak_cls}"  if weak_cls  else "",
                        [f"{a:.4f}" for a in alpha_bbox_list], beta_bbox.item(),
                        f" ⚠ weak={weak_bbox}" if weak_bbox else "",
                        [f"{a:.4f}" for a in alpha_obj_list],  beta_obj.item(),
                        f" ⚠ weak={weak_obj}"  if weak_obj  else "",
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
                best_alpha_cls_normalized  = torch.softmax(alpha_cls_raw,  dim=0).detach().clone()
                best_alpha_bbox_normalized = torch.softmax(alpha_bbox_raw, dim=0).detach().clone()
                best_alpha_obj_normalized  = torch.softmax(alpha_obj_raw,  dim=0).detach().clone()
                best_log_beta_cls  = log_beta_cls.detach().clone()
                best_log_beta_bbox = log_beta_bbox.detach().clone()
                best_log_beta_obj  = log_beta_obj.detach().clone()
                patience_counter   = 0
            else:
                patience_counter += 1

            logger.info(
                "  Epoch %d/%d — avg_loss=%.4f  best=%.4f\n"
                "      α_cls=%s  β_cls=%.4f\n"
                "      α_bbox=%s β_bbox=%.4f\n"
                "      α_obj=%s  β_obj=%.4f\n"
                "      patience=%d/%d",
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
        logger.exception("M7 decoder optimisation failed.")
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
            "Falling back to uniform α and β=1.0 for all three heads."
        )
        uniform = torch.full(
            (n_ingredients,), 1.0 / n_ingredients, dtype=torch.float32
        )
        best_alpha_cls_normalized  = uniform.clone()
        best_alpha_bbox_normalized = uniform.clone()
        best_alpha_obj_normalized  = uniform.clone()
        best_log_beta_cls  = torch.tensor(0.0, dtype=torch.float32)
        best_log_beta_bbox = torch.tensor(0.0, dtype=torch.float32)
        best_log_beta_obj  = torch.tensor(0.0, dtype=torch.float32)

    best_beta_cls:  float = torch.exp(best_log_beta_cls).item()
    best_beta_bbox: float = torch.exp(best_log_beta_bbox).item()
    best_beta_obj:  float = torch.exp(best_log_beta_obj).item()

    logger.info(
        "  ✓ Phase B complete —\n"
        "      alpha_cls=%s   beta_cls=%.4f\n"
        "      alpha_bbox=%s  beta_bbox=%.4f\n"
        "      alpha_obj=%s   beta_obj=%.4f",
        [f"{a:.4f}" for a in best_alpha_cls_normalized.tolist()],  best_beta_cls,
        [f"{a:.4f}" for a in best_alpha_bbox_normalized.tolist()], best_beta_bbox,
        [f"{a:.4f}" for a in best_alpha_obj_normalized.tolist()],  best_beta_obj,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL MERGE
    # Backbone + encoder  → greedy-selected weights (skeleton_fixed_state)
    # Decoder heads       → tri-head learned mixing (α) + temperature (β)
    # ══════════════════════════════════════════════════════════════════════════

    # Step 1 — tri-head α mix over decoder sub-heads (all ingredients)
    decoder_merged = _mix_states_tri_head(
        ingredient_states,
        best_alpha_cls_normalized,
        best_alpha_bbox_normalized,
        best_alpha_obj_normalized,
    )
    # _mix_states_tri_head also computes uniform average for other_keys;
    # we override those with the greedy-selected backbone+encoder below.

    # Step 2 — bake β into prediction-layer weights (zero inference overhead)
    decoder_merged = _apply_temperature_tri_head(
        decoder_merged,
        best_beta_cls,
        best_beta_bbox,
        best_beta_obj,
    )

    # Step 3 — overwrite backbone+encoder keys with greedy-selected weights
    final_state = dict(decoder_merged)
    overwritten = 0
    for key, val in skeleton_fixed_state.items():
        final_state[key] = val
        overwritten += 1
    logger.info(
        "  → Overwrote %d backbone+encoder keys with greedy-selected weights.",
        overwritten,
    )

    logger.info(
        "  ✓ M7 merged state built. Size: %.2f MB",
        _state_dict_size_mb(final_state),
    )
    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# Registration helper
# ─────────────────────────────────────────────────────────────────────────────
#
# To register M7 alongside M1-M6 in phase3_soup_construction.py, add the
# following entry to the MERGE_CONDITIONS dict in that module:
#
#   from yolof_soup.experiments.phase3_m7_greedy_tri_head import (
#       build_greedy_tri_head_learned_soup,
#   )
#
#   MERGE_CONDITIONS = {
#       ...existing entries...,
#       "greedy_tri_head": (
#           build_greedy_tri_head_learned_soup,
#           ("ingredient_states", "cfg", "selection_dataloader"),
#       ),
#   }
#
# And in the run() function add a corresponding checkpoint load line:
#   checkpoints["condition_7_state"] = load_state(
#       checkpoint_dir / "greedy_tri_head_soup.pth"
#   )
#
# CLI usage (once registered):
#   python -m yolof_soup.experiments.phase3_soup_construction \
#       --force-construction greedy_tri_head \
#       --verbose
# ─────────────────────────────────────────────────────────────────────────────