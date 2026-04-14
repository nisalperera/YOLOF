"""
Phase 4 – Loss landscape measurement.
Computes loss barriers B (LMC) and sharpness for all 15 decoder pairs
and all 15 backbone-encoder pairs, then runs the three statistical tests
required by RQ2/H2a and RQ2/H2b.

Run: python -m experiments.phase4_loss_landscape
"""

from __future__ import annotations

import json
import itertools
import logging
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch

from yolof_soup.config.experiment_config import (
    BACKBONE_ENC_CKPT,
    DECODER_CKPT_PATHS,
    GLOBAL_CKPT_PATHS,
    SELECTION_DATASET,
    RESULTS_DIR,
    LMC_ALPHA_STEPS,
    SAM_RHO,
    SHARPNESS_STEPS,
    DEVICE,
    build_eval_cfg,
)
from yolof_soup.config.experiment_registry import (
    build_experiment_manifest,
    ingredient_checkpoint_paths,
    ingredient_run_ids,
    validate_registry_specs,
)
from utils import (
    load_state,
    load_states,
    get_decoder_keys,
    split_decoder_subheads,
    merge_subdicts,
    build_eval_dataloader,
    quick_loss,
    mann_whitney_u_test,
    wilcoxon_paired,
    spearman_r,
)

logger = logging.getLogger(__name__)
BRANCH_BARRIER_ALPHA_STEPS = 21


def _summarize_barriers(items: list[dict]) -> dict:
    vals = [float(x.get("barrier", 0.0)) for x in items]
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _compare_barrier_sets(
    lhs_name: str,
    lhs_items: list[dict],
    rhs_name: str,
    rhs_items: list[dict],
) -> dict:
    """Compute pair-aligned barrier deltas and paired test statistics."""
    if len(lhs_items) != len(rhs_items):
        raise ValueError(
            f"Barrier set length mismatch for {lhs_name} vs {rhs_name}: "
            f"{len(lhs_items)} != {len(rhs_items)}"
        )

    per_pair = []
    lhs_vals = []
    rhs_vals = []
    deltas = []

    for li, ri in zip(lhs_items, rhs_items):
        pair_l = tuple(li.get("pair", ()))
        pair_r = tuple(ri.get("pair", ()))
        if pair_l != pair_r:
            raise ValueError(
                f"Pair mismatch for {lhs_name} vs {rhs_name}: {pair_l} != {pair_r}"
            )
        lhs_b = float(li.get("barrier", 0.0))
        rhs_b = float(ri.get("barrier", 0.0))
        delta = lhs_b - rhs_b
        lhs_vals.append(lhs_b)
        rhs_vals.append(rhs_b)
        deltas.append(delta)
        per_pair.append(
            {
                "pair": pair_l,
                "lhs_barrier": lhs_b,
                "rhs_barrier": rhs_b,
                "delta": delta,
            }
        )

    arr = np.asarray(deltas, dtype=float)
    try:
        wil = wilcoxon_paired(lhs_vals, rhs_vals)
    except Exception as exc:
        wil = {"error": str(exc)}

    return {
        "lhs": lhs_name,
        "rhs": rhs_name,
        "count": int(arr.size),
        "mean_delta": float(np.mean(arr)) if arr.size else 0.0,
        "median_delta": float(np.median(arr)) if arr.size else 0.0,
        "std_delta": float(np.std(arr)) if arr.size else 0.0,
        "min_delta": float(np.min(arr)) if arr.size else 0.0,
        "max_delta": float(np.max(arr)) if arr.size else 0.0,
        "positive_fraction": float(np.mean(arr > 0.0)) if arr.size else 0.0,
        "wilcoxon": wil,
        "per_pair": per_pair,
    }


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model_with_state(full_state: dict, device):
    from detectron2.modeling import build_model
    cfg   = build_eval_cfg()
    model = build_model(cfg).to(device)
    model.load_state_dict(full_state, strict=True)
    return model


# ── Loss barrier B (LMC) ─────────────────────────────────────────────────────

def compute_loss_barrier(
    state_A: dict,
    state_B: dict,
    dataloader,
    device,
    n_steps: int = LMC_ALPHA_STEPS,
    interpolate_keys: list[str] | None = None,
) -> dict:
    """
    B(A, B) = max_{alpha} L(alpha·A + (1-alpha)·B)
              - [(1-alpha)·L(A) + alpha·L(B)]
    """
    alphas = np.linspace(0, 1, n_steps)
    losses = []

    interp_keys = set(interpolate_keys) if interpolate_keys is not None else set(state_A.keys())

    for alpha in alphas:
        interp = {}
        for k in state_A:
            a = state_A[k]
            b = state_B[k]
            if k in interp_keys and torch.is_floating_point(a) and torch.is_floating_point(b):
                interp[k] = ((1 - alpha) * a.float() + alpha * b.float()).to(a.dtype)
            else:
                interp[k] = a
        model = build_model_with_state(interp, device)
        loss  = quick_loss(model, dataloader, device)
        losses.append(loss)
        logger.debug("  alpha=%.2f  loss=%.5f", alpha, loss)

    linear = [(1 - a) * losses[0] + a * losses[-1] for a in alphas]
    curve  = [losses[i] - linear[i] for i in range(n_steps)]
    barrier = max(curve)

    return dict(
        barrier  = float(barrier),
        losses   = [float(l) for l in losses],
        alphas   = [float(a) for a in alphas],
        loss_A   = float(losses[0]),
        loss_B   = float(losses[-1]),
    )


# ── Sharpness (SAM-style gradient ascent) ─────────────────────────────────────

def compute_sharpness(
    full_state: dict,
    dataloader,
    device,
    rho: float     = SAM_RHO,
    n_steps: int   = SHARPNESS_STEPS,
) -> float:
    """
    Sharpness ≈ max_{||ε|| ≤ ρ} L(θ + ε) − L(θ)
    Approximated by gradient ascent on the perturbation for n_steps steps.
    Uses the YOLOF forward pass in train mode with return_val_loss=True.
    """
    model     = build_model_with_state(full_state, device)
    base_loss = quick_loss(model, dataloader, device)

    model.train()
    # Collect only floating-point leaf parameters that exist in full_state
    params = {n: p for n, p in model.named_parameters()
              if p.requires_grad and n in full_state}

    for _ in range(n_steps):
        # Forward + loss
        model.zero_grad()
        loss = quick_loss(model, dataloader, device)

        grads = torch.autograd.grad(
            torch.tensor(loss, requires_grad=True),
            list(params.values()),
            allow_unused=True,
        )
        with torch.no_grad():
            grad_norm = torch.sqrt(
                sum(g.norm() ** 2 for g in grads if g is not None)
            )
            if grad_norm > 0:
                for p, g in zip(params.values(), grads):
                    if g is not None:
                        p.data.add_(rho * g / (grad_norm + 1e-8))

    perturbed_loss = quick_loss(model, dataloader, device)
    return float(perturbed_loss - base_loss)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 4: Loss Landscape Measurement")
    logger.info("=" * 60)

    registry_errors = validate_registry_specs()
    if registry_errors:
        raise ValueError(f"Invalid experiment registry: {registry_errors}")

    run_manifest = build_experiment_manifest(
        decoder_paths=DECODER_CKPT_PATHS,
        global_paths=GLOBAL_CKPT_PATHS,
        checkpoint_dir=str(RESULTS_DIR),
    )
    ingredient_ids = ingredient_run_ids()
    decoder_paths = ingredient_checkpoint_paths(DECODER_CKPT_PATHS)
    run_manifest_runs = cast(Dict[str, Dict[str, object]], run_manifest["runs"])
    global_paths = [
        Path(str(run_manifest_runs[run_id]["global_checkpoint"]))
        for run_id in ingredient_ids
    ]

    backbone_enc_state = load_state(BACKBONE_ENC_CKPT)
    decoder_states     = load_states(decoder_paths)
    decoder_keys       = get_decoder_keys(decoder_states[0])
    cls_keys, reg_keys, shared_keys = split_decoder_subheads(decoder_keys)
    global_states      = load_states(cast(list[str | Path], global_paths))

    cfg        = build_eval_cfg(SELECTION_DATASET)
    dataloader = build_eval_dataloader(cfg, SELECTION_DATASET)

    results: Dict[str, Any] = dict(
        decoder_barriers   = [],
        decoder_barriers_branch = {"cls": [], "reg": [], "shared": [], "full_decoder": []},
        decoder_barrier_summary = {},
        branch_comparisons = {},
        backbone_barriers  = [],
        backbone_barrier_summary = {},
        decoder_sharpness  = [],
        backbone_sharpness = [],
        ingredient_run_ids = ingredient_ids,
        branch_alpha_steps = BRANCH_BARRIER_ALPHA_STEPS,
    )
    pairs = list(itertools.combinations(range(len(decoder_states)), 2))

    # ── 15 decoder pairs ──────────────────────────────────
    logger.info("\n--- Decoder pairs (n=15) ---")
    for i, j in pairs:
        logger.info("Decoder pair (%s, %s)", ingredient_ids[i], ingredient_ids[j])
        full_A = merge_subdicts(backbone_enc_state,
                                {k: decoder_states[i][k] for k in decoder_keys})
        full_B = merge_subdicts(backbone_enc_state,
                                {k: decoder_states[j][k] for k in decoder_keys})
        res = compute_loss_barrier(full_A, full_B, dataloader, DEVICE)
        results["decoder_barriers"].append(
            dict(pair=(ingredient_ids[i], ingredient_ids[j]), **res)
        )

        cls_res = compute_loss_barrier(
            full_A,
            full_B,
            dataloader,
            DEVICE,
            n_steps=BRANCH_BARRIER_ALPHA_STEPS,
            interpolate_keys=cls_keys,
        )
        reg_res = compute_loss_barrier(
            full_A,
            full_B,
            dataloader,
            DEVICE,
            n_steps=BRANCH_BARRIER_ALPHA_STEPS,
            interpolate_keys=reg_keys,
        )
        shared_res = compute_loss_barrier(
            full_A,
            full_B,
            dataloader,
            DEVICE,
            n_steps=BRANCH_BARRIER_ALPHA_STEPS,
            interpolate_keys=shared_keys,
        )
        full_dec_res = compute_loss_barrier(
            full_A,
            full_B,
            dataloader,
            DEVICE,
            n_steps=BRANCH_BARRIER_ALPHA_STEPS,
            interpolate_keys=decoder_keys,
        )
        results["decoder_barriers_branch"]["cls"].append(
            dict(pair=(ingredient_ids[i], ingredient_ids[j]), **cls_res)
        )
        results["decoder_barriers_branch"]["reg"].append(
            dict(pair=(ingredient_ids[i], ingredient_ids[j]), **reg_res)
        )
        results["decoder_barriers_branch"]["shared"].append(
            dict(pair=(ingredient_ids[i], ingredient_ids[j]), **shared_res)
        )
        results["decoder_barriers_branch"]["full_decoder"].append(
            dict(pair=(ingredient_ids[i], ingredient_ids[j]), **full_dec_res)
        )
        logger.info("  B = %.5f", res["barrier"])

    results["decoder_barrier_summary"] = {
        "full": _summarize_barriers(results["decoder_barriers"]),
        "cls": _summarize_barriers(results["decoder_barriers_branch"]["cls"]),
        "reg": _summarize_barriers(results["decoder_barriers_branch"]["reg"]),
        "shared": _summarize_barriers(results["decoder_barriers_branch"]["shared"]),
        "full_decoder": _summarize_barriers(results["decoder_barriers_branch"]["full_decoder"]),
    }

    branch_sets = {
        "full": results["decoder_barriers"],
        "cls": results["decoder_barriers_branch"]["cls"],
        "reg": results["decoder_barriers_branch"]["reg"],
        "shared": results["decoder_barriers_branch"]["shared"],
        "full_decoder": results["decoder_barriers_branch"]["full_decoder"],
    }
    results["branch_comparisons"] = {
        "cls_vs_reg": _compare_barrier_sets("cls", branch_sets["cls"], "reg", branch_sets["reg"]),
        "cls_vs_shared": _compare_barrier_sets("cls", branch_sets["cls"], "shared", branch_sets["shared"]),
        "reg_vs_shared": _compare_barrier_sets("reg", branch_sets["reg"], "shared", branch_sets["shared"]),
        "full_decoder_vs_full_model": _compare_barrier_sets(
            "full_decoder",
            branch_sets["full_decoder"],
            "full",
            branch_sets["full"],
        ),
    }

    # ── 15 backbone-encoder pairs ──────────────────────────
    logger.info("\n--- Backbone-encoder pairs (n=15) ---")
    for i, j in pairs:
        logger.info("Backbone-enc pair (%s, %s)", ingredient_ids[i], ingredient_ids[j])
        res = compute_loss_barrier(
            global_states[i], global_states[j], dataloader, DEVICE
        )
        results["backbone_barriers"].append(
            dict(pair=(ingredient_ids[i], ingredient_ids[j]), **res)
        )
        logger.info("  B = %.5f", res["barrier"])

    results["backbone_barrier_summary"] = _summarize_barriers(results["backbone_barriers"])

    # ── Sharpness ──────────────────────────────────────────
    logger.info("\n--- Sharpness ---")
    for i, ds in enumerate(decoder_states):
        full = merge_subdicts(backbone_enc_state,
                              {k: ds[k] for k in decoder_keys})
        s    = compute_sharpness(full, dataloader, DEVICE)
        results["decoder_sharpness"].append(s)
        logger.info("  Decoder %d sharpness = %.5f", i + 1, s)

    bb_sharp = compute_sharpness(backbone_enc_state, dataloader, DEVICE)
    results["backbone_sharpness"] = [bb_sharp] * len(decoder_states)
    logger.info("  Backbone-encoder sharpness = %.5f", bb_sharp)

    # ── Statistical tests ──────────────────────────────────
    dec_B_vals = [r["barrier"] for r in results["decoder_barriers"]]
    bb_B_vals  = [r["barrier"] for r in results["backbone_barriers"]]

    logger.info("\n=== RQ2/H2a: Mann-Whitney U (decoder B vs backbone B) ===")
    mw = mann_whitney_u_test(dec_B_vals, bb_B_vals)
    results["mann_whitney"] = mw
    logger.info(json.dumps(mw, indent=2))

    logger.info("\n=== RQ2/H2a: Paired Wilcoxon (sharpness) ===")
    wil = wilcoxon_paired(results["decoder_sharpness"],
                          results["backbone_sharpness"])
    results["wilcoxon_sharpness"] = wil
    logger.info(json.dumps(wil, indent=2))

    try:
        with open(f"{RESULTS_DIR}/phase3_soup_results.json") as f:
            p3 = json.load(f)
        baseline_ap  = p3["baseline"]["AP"]
        greedy_ap    = p3["greedy_head"]["AP"]
        map_gains    = [greedy_ap - baseline_ap] * len(dec_B_vals)
        spear        = spearman_r(dec_B_vals, map_gains)
        results["spearman"] = spear
        logger.info("\n=== RQ2/H2b: Spearman r (decoder B vs mAP gain) ===")
        logger.info(json.dumps(spear, indent=2))
    except FileNotFoundError:
        logger.warning("Phase 3 results not found — run Phase 3 first for H2b.")

    out = f"{RESULTS_DIR}/phase4_landscape_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Phase 4 complete → %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()