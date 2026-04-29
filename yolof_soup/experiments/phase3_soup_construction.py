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
from pathlib import Path
from typing import Dict, cast

import numpy as np
import torch
from torch import nn

from yolof.analysis.model_soup import (
    build_branch_weighted_soup_from_states,
    build_soup,
    fisher_branch_coefficients_from_traces,
    uniform_branch_coefficients,
)
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
from yolof_soup.config.experiment_registry import (
    build_experiment_manifest,
    ingredient_checkpoint_paths,
    ingredient_run_ids,
    validate_registry_specs,
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
    build_eval_dataloader,
    get_map,
    compute_coco_map,
)

logger = logging.getLogger(__name__)

M3_DIRICHLET_ALPHA = 1.0
M3_NUM_SAMPLES = 64
M3_RANDOM_SEED = 42
M4_TRACE_METHOD = "pyhessian_hutchinson"
M4_TRACE_FALLBACK = "state_dict_l2_proxy"
M4_HESSIAN_MAX_ITERS = 50


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


def random_dirichlet_search_subhead(
    anchor_dec: dict,
    taus: list,
    cls_keys: list,
    reg_keys: list,
    shared_keys: list,
    backbone_enc_state: dict,
    device,
    alpha: float = M3_DIRICHLET_ALPHA,
    num_samples: int = M3_NUM_SAMPLES,
    seed: int = M3_RANDOM_SEED,
) -> tuple:
    """
    Condition M3: independent Dirichlet random simplex search for cls/reg.

    Returns (best_lambdas_cls, best_lambdas_reg, best_map, search_log).
    """
    cfg = build_eval_cfg(SELECTION_DATASET)
    n = len(taus)
    rng = np.random.default_rng(seed)

    best_lc = [1.0 / n] * n
    best_lr = [1.0 / n] * n
    best_map = float("-inf")
    search_log = []

    for sample_id in range(num_samples):
        lc = rng.dirichlet(alpha=np.full(n, alpha)).tolist()
        lr = rng.dirichlet(alpha=np.full(n, alpha)).tolist()
        dec = apply_subhead_lambdas(
            anchor_dec,
            taus,
            lc,
            lr,
            cls_keys,
            reg_keys,
            shared_keys,
        )
        full = assemble_full_state(backbone_enc_state, dec)
        model = build_model_with_state(full, device)
        m = get_map(model, cfg, SELECTION_DATASET)
        search_log.append(
            {
                "sample_id": sample_id,
                "lambdas_cls": lc,
                "lambdas_reg": lr,
                "selection_map": float(m),
            }
        )

        if m > best_map:
            best_map = m
            best_lc = lc
            best_lr = lr

    return best_lc, best_lr, float(best_map), search_log


def _trace_proxy_from_states(states: list, cls_keys: list, reg_keys: list) -> tuple:
    """Fallback M4 proxy: branch scores from tensor squared norms."""
    cls_traces = []
    reg_traces = []
    for sd in states:
        cls_val = 0.0
        reg_val = 0.0
        for k in cls_keys:
            if k in sd and torch.is_floating_point(sd[k]):
                cls_val += float((sd[k].float() ** 2).sum().item())
        for k in reg_keys:
            if k in sd and torch.is_floating_point(sd[k]):
                reg_val += float((sd[k].float() ** 2).sum().item())
        cls_traces.append(cls_val)
        reg_traces.append(reg_val)
    return cls_traces, reg_traces


def _extract_scalar_trace(trace_output) -> float:
    """Handle pyhessian trace outputs that may be scalar/list/tuple."""
    if isinstance(trace_output, (list, tuple)):
        vals = [float(v) for v in trace_output]
        return float(np.mean(vals)) if vals else 0.0
    return float(trace_output)


class _YOLOFLossWrapper(nn.Module):
    """Adapter that exposes YOLOF validation losses as a scalar objective."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, batch):
        outputs = self.base_model(batch, return_val_loss=True)
        if isinstance(outputs, dict):
            losses = outputs.get("losses", outputs)
        elif isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            first = outputs[0]
            losses = first.get("losses", first)
        else:
            raise RuntimeError("Unexpected YOLOF loss output format")

        loss_cls = losses.get("loss_cls", torch.tensor(0.0, device=next(self.base_model.parameters()).device))
        loss_reg = losses.get("loss_box_reg", torch.tensor(0.0, device=next(self.base_model.parameters()).device))
        return loss_cls + loss_reg


def _set_trainable_by_key(model: nn.Module, allowed_keys: set[str]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name in allowed_keys


def branch_trace_from_pyhessian_or_proxy(
    global_states: list,
    cls_keys: list,
    reg_keys: list,
    cfg,
    device,
) -> tuple:
    """Estimate per-model cls/reg traces via pyhessian, else fallback proxy."""
    try:
        from pyhessian import hessian as pyhessian_hessian

        dataloader = build_eval_dataloader(cfg, SELECTION_DATASET)
        batch = next(iter(dataloader))

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_device(v) for v in x]
            if isinstance(x, tuple):
                return tuple(_to_device(v) for v in x)
            return x

        batch = _to_device(batch)

        cls_traces = []
        reg_traces = []
        for state in global_states:
            model = build_model_with_state(state, device)
            wrapped = _YOLOFLossWrapper(model)

            _set_trainable_by_key(wrapped.base_model, set(cls_keys))
            h_cls = pyhessian_hessian(
                wrapped,
                criterion=lambda out, _: out,
                data=(batch, None),
                cuda=(str(device).startswith("cuda")),
            )
            cls_trace = _extract_scalar_trace(h_cls.trace(maxIter=M4_HESSIAN_MAX_ITERS))

            _set_trainable_by_key(wrapped.base_model, set(reg_keys))
            h_reg = pyhessian_hessian(
                wrapped,
                criterion=lambda out, _: out,
                data=(batch, None),
                cuda=(str(device).startswith("cuda")),
            )
            reg_trace = _extract_scalar_trace(h_reg.trace(maxIter=M4_HESSIAN_MAX_ITERS))

            cls_traces.append(cls_trace)
            reg_traces.append(reg_trace)

        coeffs = fisher_branch_coefficients_from_traces(cls_traces, reg_traces)
        return coeffs, cls_traces, reg_traces, M4_TRACE_METHOD

    except Exception as exc:
        logger.warning("M4 pyhessian trace failed, falling back to proxy: %s", exc)
        cls_traces, reg_traces = _trace_proxy_from_states(global_states, cls_keys, reg_keys)
        coeffs = fisher_branch_coefficients_from_traces(cls_traces, reg_traces)
        return coeffs, cls_traces, reg_traces, M4_TRACE_FALLBACK


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 3: Soup Construction & Evaluation")
    logger.info("=" * 60)

    registry_errors = validate_registry_specs()
    if registry_errors:
        raise ValueError(f"Invalid experiment registry: {registry_errors}")

    run_manifest = build_experiment_manifest(
        decoder_paths=DECODER_CKPT_PATHS,
        global_paths=GLOBAL_CKPT_PATHS,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )
    ingredient_ids = ingredient_run_ids()
    decoder_paths = ingredient_checkpoint_paths(DECODER_CKPT_PATHS)
    run_manifest_runs = cast(Dict[str, Dict[str, object]], run_manifest["runs"])
    global_paths = [
        Path(str(run_manifest_runs[run_id]["global_checkpoint"]))
        for run_id in ingredient_ids
    ]

    manifest_out = f"{RESULTS_DIR}/experiment_run_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info("Run manifest saved → %s", manifest_out)

    backbone_enc_state = load_state(BACKBONE_ENC_CKPT)
    decoder_states     = load_states(decoder_paths)
    global_states      = load_states(cast(list[str | Path], global_paths))
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
    m1_global = build_soup(
        checkpoint_paths=cast(list[str | Path], global_paths),
        strategy="full",
    )
    model       = build_model_with_state(m1_global, DEVICE)
    m1_global_m = compute_coco_map(
        model,
        cfg_eval,
        EVAL_DATASET,
        RESULTS_DIR,
        tag="condition_m1_global_uniform",
    )
    results["m1_global_uniform"] = m1_global_m
    results["uniform_global"] = m1_global_m

    logger.info("\n--- Condition M2 Branch Uniform Soup ---")
    m2_branch = build_soup(
        checkpoint_paths=cast(list[str | Path], global_paths),
        strategy="branch_uniform",
    )
    model       = build_model_with_state(m2_branch, DEVICE)
    m2_branch_m = compute_coco_map(
        model,
        cfg_eval,
        EVAL_DATASET,
        RESULTS_DIR,
        tag="condition_m2_branch_uniform",
    )
    results["m2_branch_uniform"] = m2_branch_m
    results["m2_coefficients"] = uniform_branch_coefficients(len(global_paths))

    logger.info("\n--- Condition M3 Branch Dirichlet Search ---")
    m3_lc, m3_lr, m3_sel_map, m3_search_log = random_dirichlet_search_subhead(
        anchor_dec=anchor_dec,
        taus=taus,
        cls_keys=cls_keys,
        reg_keys=reg_keys,
        shared_keys=shared_keys,
        backbone_enc_state=backbone_enc_state,
        device=DEVICE,
    )
    m3_dec = apply_subhead_lambdas(
        anchor_dec,
        taus,
        m3_lc,
        m3_lr,
        cls_keys,
        reg_keys,
        shared_keys,
    )
    m3_full = assemble_full_state(backbone_enc_state, m3_dec)
    model = build_model_with_state(m3_full, DEVICE)
    m3_branch_m = compute_coco_map(
        model,
        cfg_eval,
        EVAL_DATASET,
        RESULTS_DIR,
        tag="condition_m3_branch_dirichlet",
    )
    results["m3_branch_dirichlet"] = m3_branch_m
    results["m3_coefficients"] = {
        "cls": m3_lc,
        "reg": m3_lr,
        "selection_map": m3_sel_map,
        "alpha": M3_DIRICHLET_ALPHA,
        "num_samples": M3_NUM_SAMPLES,
        "seed": M3_RANDOM_SEED,
    }
    results["m3_search_log"] = m3_search_log

    logger.info("\n--- Condition M4 Branch Fisher-Weighted Soup ---")
    m4_coeffs, m4_cls_traces, m4_reg_traces, m4_trace_method = branch_trace_from_pyhessian_or_proxy(
        global_states=global_states,
        cls_keys=cls_keys,
        reg_keys=reg_keys,
        cfg=cfg_eval,
        device=DEVICE,
    )
    m4_soup = build_branch_weighted_soup_from_states(global_states, m4_coeffs)
    model = build_model_with_state(m4_soup, DEVICE)
    m4_branch_m = compute_coco_map(
        model,
        cfg_eval,
        EVAL_DATASET,
        RESULTS_DIR,
        tag="condition_m4_branch_fisher",
    )
    results["m4_branch_fisher"] = m4_branch_m
    results["m4_coefficients"] = {
        "coefficients": m4_coeffs,
        "cls_traces": m4_cls_traces,
        "reg_traces": m4_reg_traces,
        "trace_method": m4_trace_method,
    }

    results["registry"] = {
        "ingredient_run_ids": ingredient_ids,
        "manifest": manifest_out,
    }
    logger.info("M1 global uniform mAP: %.4f", m1_global_m["AP"])
    logger.info("M2 branch uniform mAP: %.4f", m2_branch_m["AP"])
    logger.info("M3 branch dirichlet mAP: %.4f", m3_branch_m["AP"])
    logger.info("M4 branch fisher mAP: %.4f", m4_branch_m["AP"])

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
        m1_global,
        metadata=dict(metrics=m1_global_m, phase="3", condition_id="M1"),
    )
    save_checkpoint(
        f"{CHECKPOINT_DIR}/branch_uniform_soup.pth",
        m2_branch,
        metadata=dict(
            metrics=m2_branch_m,
            phase="3",
            condition_id="M2",
            coefficients=uniform_branch_coefficients(len(global_paths)),
        ),
    )
    save_checkpoint(
        f"{CHECKPOINT_DIR}/branch_dirichlet_soup.pth",
        m3_full,
        metadata=dict(
            metrics=m3_branch_m,
            phase="3",
            condition_id="M3",
            lambdas_cls=m3_lc,
            lambdas_reg=m3_lr,
            selection_map=m3_sel_map,
            alpha=M3_DIRICHLET_ALPHA,
            num_samples=M3_NUM_SAMPLES,
            seed=M3_RANDOM_SEED,
        ),
    )
    save_checkpoint(
        f"{CHECKPOINT_DIR}/branch_fisher_soup.pth",
        m4_soup,
        metadata=dict(
            metrics=m4_branch_m,
            phase="3",
            condition_id="M4",
            coefficients=m4_coeffs,
            cls_traces=m4_cls_traces,
            reg_traces=m4_reg_traces,
            trace_method=m4_trace_method,
        ),
    )
    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()