#!/usr/bin/env python
"""
Model Soup Builder Script
==========================
Averages backbone+encoder weights from L1-L4 (and optionally C1/C2)
to produce the soup checkpoints needed for D1 / D2 / C3.

Examples
--------
# Standard soup (L1-L4 backbone+encoder)
python tools/build_soup.py \\
  --checkpoints output/L1/model_best.pth output/L2/model_best.pth \\
                output/L3/model_best.pth output/L4/model_best.pth \\
  --labels L1 L2 L3 L4 \\
  --strategy backbone_encoder \\
  --output output/soups/soup_L1-L4_backbone_encoder.pth

# Alternative soup (L1-L4 + C1 backbone+encoder)
python tools/build_soup.py \\
  --checkpoints output/L1/model_best.pth output/L2/model_best.pth \\
                output/L3/model_best.pth output/L4/model_best.pth \\
                output/C1/model_best.pth \\
  --labels L1 L2 L3 L4 C1 \\
  --strategy backbone_encoder \\
  --output output/soups/soup_L1-L4-C1_backbone_encoder.pth

# Full soup (all parameters averaged -- RQ baseline)
python tools/build_soup.py \\
  --checkpoints output/L1/model_best.pth output/L2/model_best.pth \\
                output/L3/model_best.pth output/L4/model_best.pth \\
  --labels L1 L2 L3 L4 \\
  --strategy full \\
  --output output/soups/soup_L1-L4_full.pth
"""

import json
import logging
from pathlib import Path

import torch
from detectron2.engine import default_argument_parser

from yolof.analysis.model_soup import build_soup, save_soup
from yolof_soup.utils.logging_utils import setup_logging

logger = setup_logging(logging.INFO, filename="tools/build_soup.log", use_stdout=True)


def main():
    parser = default_argument_parser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint .pth files to average.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Label per checkpoint (for metadata).")
    parser.add_argument("--strategy", choices=["full", "backbone_encoder"],
                        default="backbone_encoder",
                        help="'backbone_encoder' averages backbone+encoder only; "
                             "'full' averages all parameters.")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-model weights (uniform if omitted).")
    parser.add_argument("--output", required=True,
                        help="Output .pth path for the soup checkpoint.")
    args = parser.parse_args()

    labels = args.labels or [Path(c).stem for c in args.checkpoints]

    logger.info("Building soup from %d checkpoints:", len(args.checkpoints))
    for label, ckpt in zip(labels, args.checkpoints):
        logger.info("  %-8s  %s", label, ckpt)

    soup = build_soup(
        checkpoint_paths=args.checkpoints,
        strategy=args.strategy,
        weights=args.weights,
    )

    metadata = {
        "strategy": args.strategy,
        "sources": [
            {"label": lbl, "path": str(p)}
            for lbl, p in zip(labels, args.checkpoints)
        ],
    }
    save_soup(soup, args.output, metadata=metadata)

    # Write a JSON manifest alongside the soup
    manifest_path = Path(args.output).with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Manifest written -> %s", manifest_path)


if __name__ == "__main__":
    main()
