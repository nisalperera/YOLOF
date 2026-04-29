#!/usr/bin/env python
"""
Full Experiment Analysis Orchestrator
======================================
Runs all RQ-relevant LMC pairs AND 2-D loss landscapes, then produces
an aggregate normalised-barrier heatmap.

Experiment groups
-----------------
  rq1_pool_connectivity  -- same-architecture, different-seed pairs
                            (L1 vs L2, L1 vs L3, L1 vs L4, L2 vs L3, ...)
  rq2_long_run           -- long-run models vs pool and vs each other
                            (C1 vs C2, C1 vs L*, C2 vs L*)
  rq3_decoder_finetune   -- decoder-only fine-tune pair (D1 vs D2)
  rq3_decoder_vs_pool    -- decoder-only vs pool models (D1/D2 vs L*)

Usage
-----
Edit MODEL_PATHS at the top of this file to match your output directories,
then run::

    python tools/run_all_analysis.py \\
      --config-file configs/yolof_R_50_DC5_1x_thesis_base.yaml \\
      --output-dir output/full_analysis \\
      --num-alpha-samples 21 \\
      --max-eval-samples 500

Flags
-----
  --skip-landscapes   Skip 2-D landscape computation (faster first pass)
  --skip-lmc          Skip LMC pair computation
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model

from tools.analyze_connectivity import (
    setup_datasets_and_config,
    build_eval_dataloader,
    analyze_pair,
)
from yolof.analysis import (
    LossLandscape,
    build_soup,
    save_soup,
    load_checkpoint_state_dict,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# Configuration -- edit these paths before running
# =========================================================================== #

MODEL_PATHS: Dict[str, str] = {
    "L1": "output/yolof_thesis/L1_seed1_full/model_best.pth",
    "L2": "output/yolof_thesis/L2_seed2_full/model_best.pth",
    "L3": "output/yolof_thesis/L3_seed3_full/model_best.pth",
    "L4": "output/yolof_thesis/L4_seed4_full/model_best.pth",
    "C1": "output/yolof_thesis/C1_seed11_full_long/model_best.pth",
    "C2": "output/yolof_thesis/C2_seed22_full_long/model_best.pth",
    "D1": "output/yolof_thesis/D1_seed7_decoder_only/model_best.pth",
    "D2": "output/yolof_thesis/D2_seed7_decoder_only_altsoup/model_best.pth",
}

# =========================================================================== #


def _define_lmc_pairs() -> List[Tuple[str, str, str]]:
    """
    Return all (model_a, model_b, rq_group) LMC pairs.
    """
    pairs: List[Tuple[str, str, str]] = []

    # RQ1 / pool connectivity -- all unique L x L pairs (6 total)
    l_models = ["L1", "L2", "L3", "L4"]
    for a, b in combinations(l_models, 2):
        pairs.append((a, b, "rq1_pool_connectivity"))

    # RQ2 -- long-run models
    c_models = ["C1", "C2"]
    for c in c_models:
        for l in l_models:
            pairs.append((c, l, "rq2_long_run"))
    pairs.append(("C1", "C2", "rq2_long_run"))

    # RQ3 -- decoder-only fine-tune
    pairs.append(("D1", "D2", "rq3_decoder_finetune"))
    for l in l_models:
        pairs.append(("D1", l, "rq3_decoder_vs_pool"))
        pairs.append(("D2", l, "rq3_decoder_vs_pool"))

    return pairs


def _run_lmc_pairs(
    pairs: List[Tuple[str, str, str]],
    model_paths: Dict[str, str],
    config_file: str,
    output_root: Path,
    num_alpha_samples: int,
    max_eval_samples: int,
    device: Optional[str],
) -> Dict[str, dict]:
    """Run all LMC pairs and collect barrier results."""
    results: Dict[str, dict] = {}

    for i, (a, b, group) in enumerate(pairs):
        if a not in model_paths or b not in model_paths:
            logger.warning("Skipping %s vs %s -- path not configured.", a, b)
            continue

        pair_name = f"{a}_vs_{b}"
        group_dir = output_root / group
        group_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 60)
        logger.info("[%d/%d] LMC  %s  (%s)", i + 1, len(pairs), pair_name, group)

        _, _, success, err = analyze_pair(
            pair_idx=i,
            total_pairs=len(pairs),
            model1_path=model_paths[a],
            model2_path=model_paths[b],
            config_file=config_file,
            output_dir=group_dir,
            num_alpha_samples=num_alpha_samples,
            max_eval_samples=max_eval_samples,
            device=device,
        )

        result_file = group_dir / f"{a}_vs_{b}" / "lmc_results.json"
        if success and result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            results[pair_name] = {
                "group": group,
                "barrier": data["metrics"]["barrier_height"],
                "norm_barrier": data["metrics"]["normalized_barrier"],
                "barrier_cls": data["metrics"].get("barrier_height_cls", float("nan")),
                "barrier_bbox": data["metrics"].get("barrier_height_bbox", float("nan")),
            }
        else:
            logger.error("LMC failed for %s: %s", pair_name, err)
            results[pair_name] = {
                "group": group,
                "barrier": float("nan"),
                "norm_barrier": float("nan"),
            }

    return results


def _run_landscapes(
    model_ids: List[str],
    model_paths: Dict[str, str],
    config_file: str,
    output_root: Path,
    grid_size: int,
    radius: float,
    max_eval_samples: int,
    seed: int,
) -> None:
    """Run 2-D filter-normalised loss landscape for each model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = setup_datasets_and_config(config_file)
    dataloader = build_eval_dataloader(cfg)

    for mid in model_ids:
        if mid not in model_paths:
            logger.warning("Skipping landscape for %s -- path not configured.", mid)
            continue

        logger.info("=" * 60)
        logger.info("[Landscape] %s", mid)

        out_dir = output_root / "landscapes" / mid
        out_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(cfg)
        model.to(device)
        sd = load_checkpoint_state_dict(model_paths[mid])
        model.load_state_dict(sd, strict=False)
        model.eval()

        ll = LossLandscape(
            model=model, dataloader=dataloader, device=device,
            max_eval_samples=max_eval_samples,
        )
        surface = ll.compute(grid_size=grid_size, radius=radius, seed=seed)
        LossLandscape.save(surface, out_dir / "landscape.npz")
        LossLandscape.plot(
            surface, out_dir / "landscape.png",
            title=f"Loss Landscape -- {mid}",
        )

        Z = surface["Z"]
        stats = {
            "model": mid,
            "center_loss": float(Z[grid_size // 2, grid_size // 2]),
            "min_loss": float(Z.min()),
            "max_loss": float(Z.max()),
            "sharpness": float(Z.max() - Z[grid_size // 2, grid_size // 2]),
        }
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("[Landscape] %s -- sharpness=%.4f", mid, stats["sharpness"])


def _plot_barrier_heatmap(
    results: Dict[str, dict],
    model_ids: List[str],
    output_path: Path,
) -> None:
    """Plot a symmetric NxN heatmap of normalised LMC barriers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping heatmap.")
        return

    n = len(model_ids)
    mat = np.full((n, n), np.nan)

    id_idx = {m: i for i, m in enumerate(model_ids)}
    for pair_name, info in results.items():
        parts = pair_name.split("_vs_")
        if len(parts) == 2:
            a, b = parts
            if a in id_idx and b in id_idx:
                i, j = id_idx[a], id_idx[b]
                mat[i, j] = info["norm_barrier"]
                mat[j, i] = info["norm_barrier"]  # symmetric

    np.fill_diagonal(mat, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0)
    plt.colorbar(im, ax=ax, label="Normalised LMC barrier")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_ids, rotation=45, ha="right")
    ax.set_yticklabels(model_ids)
    ax.set_title("Pairwise Normalised LMC Barriers", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[Heatmap] Saved -> %s", output_path)


def main():
    parser = default_argument_parser()
    parser.add_argument("--output-dir", default="output/full_analysis")
    parser.add_argument("--num-alpha-samples", type=int, default=21)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grid-size", type=int, default=21,
                        help="Grid resolution for 2-D loss landscapes.")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Landscape perturbation radius.")
    parser.add_argument(
        "--landscape-models", nargs="*",
        default=["L1", "L2", "L3", "L4", "C1", "C2", "D1", "D2"],
        help="Which models to compute 2-D landscapes for.",
    )
    parser.add_argument("--skip-landscapes", action="store_true",
                        help="Skip 2-D landscape computation.")
    parser.add_argument("--skip-lmc", action="store_true",
                        help="Skip LMC pair computation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. LMC pairs
    all_results: Dict[str, dict] = {}
    if not args.skip_lmc:
        pairs = _define_lmc_pairs()
        logger.info("Running %d LMC pairs ...", len(pairs))
        all_results = _run_lmc_pairs(
            pairs=pairs,
            model_paths=MODEL_PATHS,
            config_file=args.config_file,
            output_root=output_root,
            num_alpha_samples=args.num_alpha_samples,
            max_eval_samples=args.max_eval_samples,
            device=args.device,
        )
        with open(output_root / "lmc_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("LMC summary -> %s/lmc_summary.json", output_root)

    # 2. Barrier heatmap
    if all_results:
        model_order = ["L1", "L2", "L3", "L4", "C1", "C2", "D1", "D2"]
        _plot_barrier_heatmap(
            all_results, model_order,
            output_root / "barrier_heatmap.png",
        )

    # 3. Loss landscapes
    if not args.skip_landscapes:
        _run_landscapes(
            model_ids=args.landscape_models,
            model_paths=MODEL_PATHS,
            config_file=args.config_file,
            output_root=output_root,
            grid_size=args.grid_size,
            radius=args.radius,
            max_eval_samples=args.max_eval_samples,
            seed=args.seed,
        )

    logger.info("=" * 60)
    logger.info("All analysis complete. Results in: %s", output_root)


if __name__ == "__main__":
    main()
