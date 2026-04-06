"""
Phase 3 – Soup construction and primary evaluation.
Builds uniform, greedy, and learned (sub-head coordinate descent) soups,
then evaluates all variants on the 4 000-image held-out evaluation split.

Addresses: RQ1/H1

Run: python -m experiments.phase3_soup_construction
"""

from __future__ import annotations

import json
import logging

import torch

from yolof_soup.config.experiment_config import (
    BACKBONE_ENC_CKPT,
    DECODER_CKPT_PATHS,
    GLOBAL_CKPT_PATHS,
    BASELINE_CKPT,
    SELECTION_DATASET,
    EVAL_DATASET,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    LAMBDA_GRID,
    MAX_CD_PASSES,
    CONVERGE_TOL,
    DEVICE,
    build_eval_cfg,
)
from yolof_soup.utils import (
    # checkpoint I/O
    load_state,
    load_states,
    save_checkpoint,
    # key partitioning
    get_decoder_keys,
    get_backbone_encoder_keys,
    split_decoder_subheads,
    extract_subdict,
    merge_subdicts,
    # task-vector arithmetic
    compute_anchor,
    compute_task_vectors,
    apply_uniform_lambdas,
    apply_subhead_lambdas,
    # evaluation
    get_map,
    compute_coco_map,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model_with_state(full_state: dict, device):
    """Load a state-dict into a fresh YOLOF model."""
    from detectron2.modeling import build_model
    cfg   = build_eval_cfg()
    model = build_model(cfg).to(device)
    model.load_state_dict(full_state, strict=True)
    return model


def assemble_full_state(
    backbone_enc_state: dict,
    decoder_state: dict,
) -> dict:
    """Merge backbone-encoder and decoder dicts into one full state-dict."""
    return merge_subdicts(backbone_enc_state, decoder_state)


# ── Uniform soup ──────────────────────────────────────────────────────────────

def build_uniform_soup(
    decoder_states: list,
    decoder_keys: list,
) -> dict:
    """θ̄_dec = (1/N) Σ θ_i  over decoder keys."""
    anchor = compute_anchor(decoder_states)
    return extract_subdict(anchor, decoder_keys)


# ── Greedy soup ───────────────────────────────────────────────────────────────

def build_greedy_soup(
    decoder_states: list,
    decoder_keys: list,
    backbone_enc_state: dict,
    device,
) -> tuple:
    """
    Add ingredients in descending individual-mAP order.
    Accept ingredient only when it does not decrease the current soup's mAP.
    Returns (greedy_decoder_dict, greedy_selection_mAP, order_log).
    """
    cfg = build_eval_cfg(SELECTION_DATASET)

    # Score individual ingredients on the selection subset
    scores = []
    for i, dec in enumerate(decoder_states):
        full  = assemble_full_state(backbone_enc_state, dec)
        model = build_model_with_state(full, device)
        m     = get_map(model, cfg, SELECTION_DATASET)
        scores.append((i, m))
        logger.info("Ingredient %d  selection_mAP=%.4f", i + 1, m)

    scores.sort(key=lambda x: x[1], reverse=True)

    selected, current_soup, current_map = [], None, 0.0
    order_log = []

    for rank, (idx, ind_map) in enumerate(scores):
        candidate_indices = selected + [idx]
        candidate_decs    = [decoder_states[i] for i in candidate_indices]
        candidate_soup    = build_uniform_soup(candidate_decs, decoder_keys)

        full  = assemble_full_state(backbone_enc_state, candidate_soup)
        model = build_model_with_state(full, device)
        cmap  = get_map(model, cfg, SELECTION_DATASET)

        accepted = cmap >= current_map
        order_log.append(dict(rank=rank + 1, ingredient=idx + 1,
                               individual_map=ind_map,
                               candidate_soup_map=cmap, accepted=accepted))
        logger.info("Greedy rank %d  ingredient %d  cmap=%.4f  accepted=%s",
                    rank + 1, idx + 1, cmap, accepted)
        if accepted:
            selected, current_soup, current_map = candidate_indices, candidate_soup, cmap

    return current_soup, current_map, order_log


# ── Learned soup – sub-head coordinate descent ───────────────────────────────

def coordinate_descent_subhead(
    anchor_dec: dict,
    taus: list,
    cls_keys: list,
    reg_keys: list,
    shared_keys: list,
    backbone_enc_state: dict,
    device,
) -> tuple:
    """
    Coordinate descent over (λ_cls_i, λ_reg_i) × N_ingredients.
    Maximises COCO mAP on the selection subset.
    Returns (lambdas_cls, lambdas_reg, best_map, history).
    """
    cfg = build_eval_cfg(SELECTION_DATASET)
    N   = len(taus)
    lc  = [1.0] * N
    lr  = [1.0] * N

    def _eval(lc_, lr_):
        dec   = apply_subhead_lambdas(
            anchor_dec, taus, lc_, lr_, cls_keys, reg_keys, shared_keys
        )
        full  = assemble_full_state(backbone_enc_state, dec)
        model = build_model_with_state(full, device)
        return get_map(model, cfg, SELECTION_DATASET)

    best_map = _eval(lc, lr)
    logger.info("[CD init] mAP=%.4f", best_map)
    history  = []

    for pass_idx in range(MAX_CD_PASSES):
        prev_lc, prev_lr = lc.copy(), lr.copy()
        updates = []

        for i in range(N):
            best_lam = lc[i]
            for lam in LAMBDA_GRID:
                if lam == lc[i]:
                    continue
                lc_trial    = lc.copy()
                lc_trial[i] = lam
                m = _eval(lc_trial, lr)
                logger.debug("[Pass %d][CLS] i=%d λ=%.2f mAP=%.4f", pass_idx+1, i, lam, m)
                if m > best_map:
                    best_map, best_lam = m, lam
            if best_lam != lc[i]:
                updates.append(dict(head="cls", i=i, old=lc[i], new=best_lam))
                lc[i] = best_lam

        for i in range(N):
            best_lam = lr[i]
            for lam in LAMBDA_GRID:
                if lam == lr[i]:
                    continue
                lr_trial    = lr.copy()
                lr_trial[i] = lam
                m = _eval(lc, lr_trial)
                logger.debug("[Pass %d][REG] i=%d λ=%.2f mAP=%.4f", pass_idx+1, i, lam, m)
                if m > best_map:
                    best_map, best_lam = m, lam
            if best_lam != lr[i]:
                updates.append(dict(head="reg", i=i, old=lr[i], new=best_lam))
                lr[i] = best_lam

        history.append(dict(pass_idx=pass_idx+1, updates=updates,
                            lambdas_cls=lc.copy(), lambdas_reg=lr.copy(),
                            best_map=best_map))
        logger.info("[Pass %d] best_mAP=%.4f  updates=%d",
                    pass_idx+1, best_map, len(updates))

        delta = max(
            max(abs(lc[i] - prev_lc[i]) for i in range(N)),
            max(abs(lr[i] - prev_lr[i]) for i in range(N)),
        )
        if delta < CONVERGE_TOL:
            logger.info("Converged at pass %d (delta=%.6f)", pass_idx+1, delta)
            break

    return lc, lr, best_map, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 3: Soup Construction & Evaluation")
    logger.info("=" * 60)

    backbone_enc_state = load_state(BACKBONE_ENC_CKPT)
    decoder_states     = load_states(DECODER_CKPT_PATHS)
    decoder_keys       = get_decoder_keys(decoder_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)

    # Task vectors (decoder-only keys)
    anchor_dec = compute_anchor(decoder_states)
    taus       = compute_task_vectors(decoder_states, anchor_dec)

    cfg_eval   = build_eval_cfg(EVAL_DATASET)
    results    = {}

    # ── Baseline (best individual model) ──────────────────
    logger.info("\n--- Baseline ---")
    baseline_dec = load_state(BASELINE_CKPT)
    full         = assemble_full_state(backbone_enc_state, baseline_dec)
    model        = build_model_with_state(full, DEVICE)
    baseline_m   = compute_coco_map(model, cfg_eval, EVAL_DATASET,
                                     RESULTS_DIR, tag="baseline")
    results["baseline"] = baseline_m
    logger.info("Baseline mAP: %.4f", baseline_m["AP"])

    # ── Uniform head-specific soup ─────────────────────────
    logger.info("\n--- Uniform Head-Specific Soup ---")
    uniform_dec = build_uniform_soup(decoder_states, decoder_keys)
    full        = assemble_full_state(backbone_enc_state, uniform_dec)
    model       = build_model_with_state(full, DEVICE)
    uniform_m   = compute_coco_map(model, cfg_eval, EVAL_DATASET,
                                    RESULTS_DIR, tag="uniform_head")
    results["uniform_head"] = uniform_m
    logger.info("Uniform head-specific mAP: %.4f", uniform_m["AP"])

    # ── Greedy head-specific soup ──────────────────────────
    logger.info("\n--- Greedy Head-Specific Soup ---")
    greedy_dec, greedy_sel_map, greedy_log = build_greedy_soup(
        decoder_states, decoder_keys, backbone_enc_state, DEVICE
    )
    full      = assemble_full_state(backbone_enc_state, greedy_dec)
    model     = build_model_with_state(full, DEVICE)
    greedy_m  = compute_coco_map(model, cfg_eval, EVAL_DATASET,
                                  RESULTS_DIR, tag="greedy_head")
    results["greedy_head"] = greedy_m
    results["greedy_log"]  = greedy_log
    logger.info("Greedy head-specific mAP: %.4f", greedy_m["AP"])

    # ── Learned sub-head soup (coordinate descent) ─────────
    logger.info("\n--- Learned Sub-Head Soup (Coordinate Descent) ---")
    best_lc, best_lr, cd_sel_map, cd_history = coordinate_descent_subhead(
        anchor_dec, taus, cls_keys, reg_keys, shared_keys,
        backbone_enc_state, DEVICE
    )
    learned_dec = apply_subhead_lambdas(
        anchor_dec, taus, best_lc, best_lr, cls_keys, reg_keys, shared_keys
    )
    full      = assemble_full_state(backbone_enc_state, learned_dec)
    model     = build_model_with_state(full, DEVICE)
    learned_m = compute_coco_map(model, cfg_eval, EVAL_DATASET,
                                  RESULTS_DIR, tag="learned_head")
    results["learned_head"] = learned_m
    results["cd_lambdas"]   = dict(cls=best_lc, reg=best_lr)
    results["cd_history"]   = cd_history
    logger.info("Learned head-specific mAP: %.4f", learned_m["AP"])

    # ── Global uniform soup (backbone NOT frozen) ──────────
    logger.info("\n--- Global Uniform Soup ---")
    global_states  = load_states(GLOBAL_CKPT_PATHS)
    global_anchor  = compute_anchor(global_states)
    model          = build_model_with_state(global_anchor, DEVICE)
    global_unif_m  = compute_coco_map(model, cfg_eval, EVAL_DATASET,
                                       RESULTS_DIR, tag="uniform_global")
    results["uniform_global"] = global_unif_m
    logger.info("Global uniform mAP: %.4f", global_unif_m["AP"])

    # ── Persist ────────────────────────────────────────────
    out = f"{RESULTS_DIR}/phase3_soup_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Phase 3 results saved → %s", out)

    save_checkpoint(
        f"{CHECKPOINT_DIR}/learned_head_soup.pth",
        assemble_full_state(backbone_enc_state, learned_dec),
        metadata=dict(lambdas_cls=best_lc, lambdas_reg=best_lr,
                      metrics=learned_m, phase="3"),
    )
    save_checkpoint(
        f"{CHECKPOINT_DIR}/global_uniform_soup.pth",
        global_anchor,
        metadata=dict(metrics=global_unif_m, phase="3"),
    )
    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()