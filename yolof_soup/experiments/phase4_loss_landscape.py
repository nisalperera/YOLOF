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
from utils import (
    load_state,
    load_states,
    get_decoder_keys,
    get_backbone_encoder_keys,
    merge_subdicts,
    build_eval_dataloader,
    quick_loss,
    mann_whitney_u_test,
    wilcoxon_paired,
    spearman_r,
)

logger = logging.getLogger(__name__)


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
) -> dict:
    """
    B(A, B) = max_{alpha} L(alpha·A + (1-alpha)·B)
              - [(1-alpha)·L(A) + alpha·L(B)]
    """
    alphas = np.linspace(0, 1, n_steps)
    losses = []

    for alpha in alphas:
        interp = {
            k: (alpha * state_A[k].float() + (1 - alpha) * state_B[k].float()
                ).to(state_A[k].dtype)
            for k in state_A
        }
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

    backbone_enc_state = load_state(BACKBONE_ENC_CKPT)
    decoder_states     = load_states(DECODER_CKPT_PATHS)
    decoder_keys       = get_decoder_keys(decoder_states[0])
    global_states      = load_states(GLOBAL_CKPT_PATHS)

    cfg        = build_eval_cfg(SELECTION_DATASET)
    dataloader = build_eval_dataloader(cfg, SELECTION_DATASET)

    results = dict(
        decoder_barriers   = [],
        backbone_barriers  = [],
        decoder_sharpness  = [],
        backbone_sharpness = [],
    )
    pairs = list(itertools.combinations(range(6), 2))

    # ── 15 decoder pairs ──────────────────────────────────
    logger.info("\n--- Decoder pairs (n=15) ---")
    for i, j in pairs:
        logger.info("Decoder pair (%d, %d)", i + 1, j + 1)
        full_A = merge_subdicts(backbone_enc_state,
                                {k: decoder_states[i][k] for k in decoder_keys})
        full_B = merge_subdicts(backbone_enc_state,
                                {k: decoder_states[j][k] for k in decoder_keys})
        res = compute_loss_barrier(full_A, full_B, dataloader, DEVICE)
        results["decoder_barriers"].append(dict(pair=(i+1, j+1), **res))
        logger.info("  B = %.5f", res["barrier"])

    # ── 15 backbone-encoder pairs ──────────────────────────
    logger.info("\n--- Backbone-encoder pairs (n=15) ---")
    for i, j in pairs:
        logger.info("Backbone-enc pair (%d, %d)", i + 1, j + 1)
        res = compute_loss_barrier(
            global_states[i], global_states[j], dataloader, DEVICE
        )
        results["backbone_barriers"].append(dict(pair=(i+1, j+1), **res))
        logger.info("  B = %.5f", res["barrier"])

    # ── Sharpness ──────────────────────────────────────────
    logger.info("\n--- Sharpness ---")
    for i, ds in enumerate(decoder_states):
        full = merge_subdicts(backbone_enc_state,
                              {k: ds[k] for k in decoder_keys})
        s    = compute_sharpness(full, dataloader, DEVICE)
        results["decoder_sharpness"].append(s)
        logger.info("  Decoder %d sharpness = %.5f", i + 1, s)

    bb_sharp = compute_sharpness(backbone_enc_state, dataloader, DEVICE)
    results["backbone_sharpness"] = [bb_sharp] * 6
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