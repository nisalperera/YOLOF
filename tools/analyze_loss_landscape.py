#!/usr/bin/env python
"""
Loss Landscape Analysis Script
================================
Computes 2-D filter-normalised loss landscapes for one or more trained
YOLOF checkpoints and saves plots + raw data.

Examples
--------
# Single model
python tools/analyze_loss_landscape.py \\
  --config-file configs/yolof_R_50_DC5_1x_thesis_L1.yaml \\
  --checkpoints output/L1/model_best.pth \\
  --labels L1 \\
  --output-dir output/landscape_analysis \\
  --grid-size 21 --radius 1.0 --max-eval-samples 300

# Multiple models in one run (separate landscape per model)
python tools/analyze_loss_landscape.py \\
  --config-file configs/yolof_R_50_DC5_1x_thesis_base.yaml \\
  --checkpoints output/L1/model_best.pth output/L2/model_best.pth \\
                output/C1/model_best.pth \\
  --labels L1 L2 C1 \\
  --output-dir output/landscape_analysis \\
  --grid-size 21 --radius 1.0 --max-eval-samples 300
"""

import json
import logging
from pathlib import Path

import torch
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model

from yolof.config import get_cfg
from yolof.analysis import LossLandscape, load_checkpoint_state_dict
from tools.analyze_connectivity import (
    setup_datasets_and_config,
    build_eval_dataloader,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = default_argument_parser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoint .pth files.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Human-readable label per checkpoint (default: filename stem).")
    parser.add_argument("--output-dir", default="output/landscape_analysis")
    parser.add_argument("--grid-size", type=int, default=21,
                        help="Grid resolution (odd number recommended).")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Filter-norm units for half-axis range.")
    parser.add_argument("--max-eval-samples", type=int, default=300,
                        help="Max val samples per grid point.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for direction sampling.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [Path(c).stem for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise ValueError("--labels count must match --checkpoints count.")

    cfg = setup_datasets_and_config(args.config_file, num_gpus=1)
    dataloader = build_eval_dataloader(cfg)

    for ckpt_path, label in zip(args.checkpoints, labels):
        logger.info("=" * 60)
        logger.info("Processing model: %s  (%s)", label, ckpt_path)
        logger.info("=" * 60)

        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build model and load checkpoint
        model = build_model(cfg)
        model.to(device)
        state_dict = load_checkpoint_state_dict(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Compute landscape
        ll = LossLandscape(
            model=model,
            dataloader=dataloader,
            device=device,
            max_eval_samples=args.max_eval_samples,
        )
        surface = ll.compute(
            grid_size=args.grid_size,
            radius=args.radius,
            seed=args.seed,
        )

        # Save
        LossLandscape.save(surface, out_dir / "landscape.npz")
        LossLandscape.plot(
            surface, out_dir / "landscape.png",
            title=f"Loss Landscape -- {label}",
        )

        # Save summary stats
        Z = surface["Z"]
        stats = {
            "label": label,
            "checkpoint": str(ckpt_path),
            "grid_size": args.grid_size,
            "radius": args.radius,
            "center_loss": float(Z[args.grid_size // 2, args.grid_size // 2]),
            "min_loss": float(Z.min()),
            "max_loss": float(Z.max()),
            "mean_loss": float(Z.mean()),
            "std_loss": float(Z.std()),
            "sharpness": float(Z.max() - Z[args.grid_size // 2, args.grid_size // 2]),
        }
        with open(out_dir / "landscape_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("[%s] center=%.4f  min=%.4f  max=%.4f  sharpness=%.4f",
                    label, stats["center_loss"], stats["min_loss"],
                    stats["max_loss"], stats["sharpness"])

    logger.info("All landscapes complete. Results in %s", output_root)


if __name__ == "__main__":
    main()
