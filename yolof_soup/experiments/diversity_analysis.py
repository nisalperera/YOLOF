"""
RQ3 / H3 – Decoder diversity vs. soup performance analysis.
N=3 vs N=6 ingredients: does more diversity help head-specific soups
more than global soups?

Run: python -m experiments.rq3_diversity_analysis
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import torch

from yolof_soup.config.experiment_config import (
    BACKBONE_ENC_CKPT,
    DECODER_CKPT_PATHS,
    GLOBAL_CKPT_PATHS,
    EVAL_DATASET,
    RESULTS_DIR,
    DEVICE,
    build_eval_cfg,
)
from yolof_soup.utils import (
    load_state,
    load_states,
    get_decoder_keys,
    merge_subdicts,
    compute_anchor,
    compute_task_vectors,
    get_map,
    compare_diversity_gain,
)
from yolof_soup.utils.global_logger import configure_logger


def build_model_with_state(full_state: dict, device):
    from detectron2.modeling import build_model
    cfg   = build_eval_cfg()
    model = build_model(cfg).to(device)
    model.load_state_dict(full_state, strict=True)
    return model


def pairwise_diversity(taus: list) -> float:
    """D = mean pairwise L2 norm over all task vectors."""
    pairs = list(itertools.combinations(range(len(taus)), 2))
    norms = []
    for i, j in pairs:
        diff = torch.cat([(taus[i][k] - taus[j][k]).flatten() for k in taus[0]])
        norms.append(diff.norm().item())
    return float(np.mean(norms)) if norms else 0.0


def uniform_soup_map(
    states: list,
    keys: list,
    backbone_enc_state: dict,
    device,
) -> float:
    """Build uniform soup from states[keys], inject backbone-enc, evaluate."""
    anchor = compute_anchor(states)
    soup   = {k: anchor[k] for k in keys}
    full   = merge_subdicts(backbone_enc_state, soup)
    model  = build_model_with_state(full, device)
    cfg    = build_eval_cfg(EVAL_DATASET)
    return get_map(model, cfg, EVAL_DATASET)


def global_uniform_soup_map(states: list, device) -> float:
    """Uniform soup over full-model states."""
    anchor = compute_anchor(states)
    model  = build_model_with_state(anchor, device)
    cfg    = build_eval_cfg(EVAL_DATASET)
    return get_map(model, cfg, EVAL_DATASET)


def main():

    logger = configure_logger(level=logging.INFO, add_file_handler=True, log_file="rq3_diversity_analysis.log")

    logger.info("=" * 60)
    logger.info("RQ3: Diversity vs. Soup Performance")
    logger.info("=" * 60)

    backbone_enc_state = load_state(BACKBONE_ENC_CKPT)
    decoder_states     = load_states(DECODER_CKPT_PATHS)
    decoder_keys       = get_decoder_keys(decoder_states[0])

    anchor_dec  = compute_anchor(decoder_states)
    taus        = compute_task_vectors(decoder_states, anchor_dec)

    global_states = load_states(GLOBAL_CKPT_PATHS)

    results = {}

    # ── Head-specific: N=3 and N=6 ────────────────────────
    for N in [3, 6]:
        sub_states = decoder_states[:N]
        sub_taus   = taus[:N]
        D   = pairwise_diversity(sub_taus)
        mAP = uniform_soup_map(sub_states, decoder_keys,
                                backbone_enc_state, DEVICE)
        results[f"head_N{N}"] = dict(N=N, diversity=D, mAP=mAP)
        logging.info("Head-specific N=%d  D=%.4f  mAP=%.4f", N, D, mAP)

    # ── Global: N=3 and N=6 ───────────────────────────────
    for N in [3, 6]:
        sub_g   = global_states[:N]
        anch_g  = compute_anchor(sub_g)
        tau_g   = compute_task_vectors(sub_g, anch_g)
        D_g     = pairwise_diversity(tau_g)
        mAP_g   = global_uniform_soup_map(sub_g, DEVICE)
        results[f"global_N{N}"] = dict(N=N, diversity=D_g, mAP=mAP_g)
        logging.info("Global N=%d  D=%.4f  mAP=%.4f", N, D_g, mAP_g)

    # ── H3 directional comparison ─────────────────────────
    delta_head   = results["head_N6"]["mAP"] - results["head_N3"]["mAP"]
    delta_global = results["global_N6"]["mAP"] - results["global_N3"]["mAP"]

    h3 = compare_diversity_gain(delta_head, delta_global)
    results["h3_result"] = h3
    logging.info("\n=== RQ3/H3 ===")
    logging.info(json.dumps(h3, indent=2))

    out = f"{RESULTS_DIR}/rq3_diversity_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("RQ3 complete → %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()