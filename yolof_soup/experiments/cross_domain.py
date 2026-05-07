"""
Phase 5 – Cross-domain evaluation on Pascal VOC 2007.
Tests RQ4/H4: are head-specific soup mAP gains larger cross-domain?

Run: python -m experiments.phase5_cross_domain
"""

from __future__ import annotations

import json
import logging

from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    VOC_DATASET,
    EVAL_DATASET,
    RESULTS_DIR,
    DEVICE,
    build_eval_cfg,
)
from yolof_soup.utils import (
    load_state,
    compute_coco_map,
    bootstrap_ci,
    compare_domain_gains,
)
from yolof_soup.utils.logging_utils import setup_logging


def build_model_with_state(full_state: dict, device):
    from detectron2.modeling import build_model
    cfg   = build_eval_cfg()
    model = build_model(cfg).to(device)
    model.load_state_dict(full_state, strict=True)
    return model


def main():
    setup_logging(level=logging.INFO, filename="phase5_cross_domain.log", use_stdout=True)
    logging.info("=" * 60)
    logging.info("Phase 5: Cross-Domain Evaluation (Pascal VOC 2007)")
    logging.info("=" * 60)

    head_state   = load_state(f"{CHECKPOINT_DIR}/learned_head_soup.pth")
    global_state = load_state(f"{CHECKPOINT_DIR}/global_uniform_soup.pth")

    cfg_coco = build_eval_cfg(EVAL_DATASET)
    cfg_voc  = build_eval_cfg(VOC_DATASET)

    # ── In-domain (COCO held-out eval split) ──────────────
    logging.info("\n--- In-Domain: COCO ---")
    model      = build_model_with_state(head_state, DEVICE)
    coco_head  = compute_coco_map(model, cfg_coco, EVAL_DATASET,
                                   RESULTS_DIR, tag="coco_head")
    logging.info("Head mAP50:95 = %.4f", coco_head["AP"])

    model       = build_model_with_state(global_state, DEVICE)
    coco_global = compute_coco_map(model, cfg_coco, EVAL_DATASET,
                                    RESULTS_DIR, tag="coco_global")
    logging.info("Global mAP50:95 = %.4f", coco_global["AP"])

    delta_coco  = coco_head["AP"] - coco_global["AP"]
    logging.info("Δ_COCO = %+.4f", delta_coco)

    # ── Cross-domain (Pascal VOC 2007 test) ───────────────
    logging.info("\n--- Cross-Domain: Pascal VOC 2007 ---")
    model      = build_model_with_state(head_state, DEVICE)
    voc_head   = compute_coco_map(model, cfg_voc, VOC_DATASET,
                                   RESULTS_DIR, tag="voc_head")
    logging.info("Head mAP50 = %.4f", voc_head.get("AP50", voc_head.get("AP")))

    model       = build_model_with_state(global_state, DEVICE)
    voc_global  = compute_coco_map(model, cfg_voc, VOC_DATASET,
                                    RESULTS_DIR, tag="voc_global")
    logging.info("Global mAP50 = %.4f", voc_global.get("AP50", voc_global.get("AP")))

    voc_ap_key  = "AP50" if "AP50" in voc_head else "AP"
    delta_voc   = voc_head[voc_ap_key] - voc_global[voc_ap_key]
    logging.info("Δ_VOC = %+.4f", delta_voc)

    # ── Bootstrap 95 % CI for head soup VOC mAP ───────────
    logging.info("\n--- Bootstrap CIs ---")
    ci_coco = bootstrap_ci([coco_head["AP"]])
    ci_voc  = bootstrap_ci([voc_head[voc_ap_key]])
    logging.info("COCO head CI: [%.4f, %.4f]", ci_coco["lower"], ci_coco["upper"])
    logging.info("VOC  head CI: [%.4f, %.4f]", ci_voc["lower"],  ci_voc["upper"])

    # ── RQ4/H4 falsification ──────────────────────────────
    h4_result = compare_domain_gains(delta_coco, delta_voc, ci_voc)
    logging.info("\n=== RQ4/H4: %s", h4_result["h4_direction"])
    logging.info(json.dumps(h4_result, indent=2))

    results = dict(
        coco_head_metrics   = coco_head,
        coco_global_metrics = coco_global,
        voc_head_metrics    = voc_head,
        voc_global_metrics  = voc_global,
        delta_coco          = delta_coco,
        delta_voc           = delta_voc,
        ci_coco_head        = ci_coco,
        ci_voc_head         = ci_voc,
        h4_result           = h4_result,
    )
    out = f"{RESULTS_DIR}/phase5_cross_domain_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Phase 5 complete → %s", out)


if __name__ == "__main__":
    main()