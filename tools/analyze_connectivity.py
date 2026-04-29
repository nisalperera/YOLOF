#!/usr/bin/env python
# Copyright (c) 2024
"""
Linear Mode Connectivity Analysis Script

Analyzes the loss landscape between two trained YOLOF models by:
1. Loading two trained model checkpoints
2. Interpolating their weights at different alpha values
3. Evaluating loss at each interpolation point
4. Computing connectivity metrics and generating visualizations
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from torch.utils.data import DataLoader

from yolof.config import get_cfg as get_yolof_cfg
from yolof.analysis import (
    load_checkpoint_state_dict,
    validate_model_compatibility,
    interpolate_state_dict,
    evaluate_loss_on_dataset,
    compute_connectivity_metrics,
)
from yolof.data import YOLOFDatasetMapper

logger = logging.getLogger(__name__)


def _trivial_batch_collator(batch):
    return batch


def setup_datasets_and_config(config_file: str, num_gpus: int = 1) -> CfgNode:
    """
    Setup datasets and configuration.
    
    Args:
        config_file: Path to config YAML file
        num_gpus: Number of GPUs
        
    Returns:
        Config object
    """
    logger.info("Setting up datasets and configuration...")
    
    root_dir = Path(__file__).resolve().parents[1]
    
    # Try to register datasets if not already registered
    try:
        # Check if datasets are already registered
        DatasetCatalog.get("coco2017_train")
        DatasetCatalog.get("coco2017_val")
        logger.info("Datasets already registered")
    except Exception as e:
        logger.warning(f"Datasets not found, attempting to register: {e}")
        
        # Try to find COCO dataset paths
        coco_paths = [
            (root_dir / "datasets" / "coco"),
            Path("/kaggle/input/2017-2017"),
            Path("/mnt/coco"),
        ]
        
        found_coco = False
        for coco_path in coco_paths:
            train_dir = coco_path / "images" / "train2017"
            val_dir = coco_path / "images" / "val2017"
            anno_dir = coco_path / "annotations"
            
            if train_dir.exists() and val_dir.exists() and anno_dir.exists():
                try:
                    train_ann = anno_dir / "instances_train2017.json"
                    val_ann = anno_dir / "instances_val2017.json"
                    
                    register_coco_instances(
                        "coco2017_train", {}, str(train_ann), str(train_dir)
                    )
                    register_coco_instances(
                        "coco2017_val", {}, str(val_ann), str(val_dir)
                    )
                    found_coco = True
                    logger.info(f"Registered COCO datasets from {coco_path}")
                    break
                except Exception as ex:
                    logger.warning(f"Failed to register from {coco_path}: {ex}")
                    continue
        
        if not found_coco:
            logger.warning(
                "Could not automatically find and register COCO datasets. "
                "Please ensure datasets are registered before running this script."
            )
    
    # Load and setup config
    cfg = get_yolof_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    
    # Ensure datasets are in config
    if "coco2017_val" not in cfg.DATASETS.TEST:
        cfg.DATASETS.TEST = ("coco2017_val",)
    
    cfg.freeze()
    return cfg


def build_eval_dataloader(cfg, dataset_name: str = "coco2017_val"):
    """Build evaluation dataloader."""
    logger.info(f"Building dataloader for {dataset_name}...")
    
    mapper_kwargs = YOLOFDatasetMapper.from_config(cfg, is_train=False)
    mapper = YOLOFDatasetMapper(**mapper_kwargs)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapped_dataset = MapDataset(dataset, mapper)

    dataloader = DataLoader(
        mapped_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_trivial_batch_collator,
    )

    logger.info(f"Dataloader created with {len(dataset_dicts)} samples")
    return dataloader


def analyze_pair(
    pair_idx: int,
    total_pairs: int,
    model1_path: str,
    model2_path: str,
    config_file: str,
    output_dir: Path,
    num_alpha_samples: int = 11,
    max_eval_samples: Optional[int] = 500,
    device: Optional[torch.device | str] = None,
) -> Tuple[int, str, bool, Optional[str]]:
    """
    Analyze a single model pair (for parallel execution).
    
    Args:
        pair_idx: Index of this pair in the list
        total_pairs: Total number of pairs
        model1_path: Path to first model checkpoint
        model2_path: Path to second model checkpoint
        config_file: Path to YOLOF config file
        output_dir: Base output directory
        num_alpha_samples: Number of alpha samples to use
        max_eval_samples: Maximum number of validation samples per alpha
        device: Device to run on (cuda or cpu)
        
    Returns:
        Tuple of (pair_idx, pair_name, success, error_msg)
    """
    # Setup device
    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        torch_device = torch.device(device)
    else:
        torch_device = device
    
    pair_name = f"{Path(model1_path).parent.name}_vs_{Path(model2_path).parent.name}"
    pair_output_dir = output_dir / pair_name
    pair_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"[Worker {pair_idx + 1}/{total_pairs}] Analyzing: {pair_name}")
    logger.info(f"Model 1: {model1_path}")
    logger.info(f"Model 2: {model2_path}")
    logger.info(f"{'='*70}")
    
    try:
        # Setup config and datasets (each worker gets its own instances)
        cfg = setup_datasets_and_config(config_file)
        
        # Build model and dataloader for this worker
        model = build_model(cfg)
        model.to(torch_device)
        dataloader = build_eval_dataloader(cfg)
        
        # Load checkpoints
        state_dict1 = load_checkpoint_state_dict(model1_path)
        state_dict2 = load_checkpoint_state_dict(model2_path)
        
        # Validate compatibility
        validate_model_compatibility(state_dict1, state_dict2)
        
        # Generate alpha values
        alpha_values = np.linspace(0.0, 1.0, num_alpha_samples)
        
        # Evaluate at each alpha
        loss_results = {
            "alpha": [],
            "loss_total": [],
            "loss_cls": [],
            "loss_box_reg": [],
            "num_samples": [],
        }
        
        for alpha_idx, alpha in enumerate(alpha_values):
            logger.info(f"[{pair_name}] Alpha = {alpha:.2f} ({alpha_idx + 1}/{len(alpha_values)})")
            
            # Interpolate models
            interpolated_state_dict = interpolate_state_dict(state_dict1, state_dict2, float(alpha))
            
            # Load interpolated weights into model
            model.load_state_dict(interpolated_state_dict)
            model.to(torch_device)
            model.eval()
            
            # Evaluate loss
            loss_dict = evaluate_loss_on_dataset(
                model,
                dataloader,
                torch_device,
                return_val_loss=True,
                max_samples=max_eval_samples,
            )
            
            # Store results
            loss_results["alpha"].append(float(alpha))
            loss_results["loss_total"].append(loss_dict["loss_total"])
            loss_results["loss_cls"].append(loss_dict["loss_cls"])
            loss_results["loss_box_reg"].append(loss_dict["loss_box_reg"])
            loss_results["num_samples"].append(loss_dict["num_samples"])
        
        # Compute connectivity metrics
        alpha_array = np.array(loss_results["alpha"])
        loss_array = np.array(loss_results["loss_total"])
        loss_cls_array = np.array(loss_results["loss_cls"])
        loss_reg_array = np.array(loss_results["loss_box_reg"])
        
        metrics = compute_connectivity_metrics(
            alpha_array,
            loss_array,
            loss_cls_array,
            loss_bbox_curve=loss_reg_array,
        )
        
        # Save results
        results_file = pair_output_dir / "lmc_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "model_pair": {
                        "model1": str(model1_path),
                        "model2": str(model2_path),
                    },
                    "loss_curve": loss_results,
                    "metrics": metrics,
                    "evaluation": {
                        "max_eval_samples": max_eval_samples,
                        "num_alpha_samples": num_alpha_samples,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"[{pair_name}] Results saved to {results_file}")
        
        # Log metrics
        logger.info(f"[{pair_name}] Connectivity Metrics:")
        logger.info(f"[{pair_name}]   Total barrier: {metrics['barrier_height']:.6f} (normalized {metrics['normalized_barrier']:.6f})")
        logger.info(f"[{pair_name}]   Cls barrier:   {metrics.get('barrier_height_cls', float('nan')):.6f} (normalized {metrics.get('normalized_barrier_cls', float('nan')):.6f})")
        logger.info(f"[{pair_name}]   BBox barrier:  {metrics.get('barrier_height_bbox', float('nan')):.6f} (normalized {metrics.get('normalized_barrier_bbox', float('nan')):.6f})")
        
        # Optional: generate visualization
        try:
            generate_visualization(loss_results, metrics, pair_output_dir)
        except Exception as e:
            logger.warning(f"[{pair_name}] Failed to generate visualization: {e}")
        
        return pair_idx, pair_name, True, None
    
    except Exception as e:
        error_msg = f"Error analyzing pair: {e}"
        logger.error(f"[{pair_name}] {error_msg}", exc_info=True)
        return pair_idx, pair_name, False, error_msg


def analyze_connectivity(
    config_file: str,
    model_pairs: List[Tuple[str, str]],
    output_dir: str = "output/lmc_analysis",
    num_alpha_samples: int = 11,
    max_eval_samples: Optional[int] = 500,
    device: Optional[torch.device | str] = None,
    num_workers: int = 5,
) -> None:
    """
    Main function to analyze linear mode connectivity between model pairs (parallel).
    
    Args:
        config_file: Path to YOLOF config file
        model_pairs: List of (model1_path, model2_path) tuples
        output_dir: Directory to save analysis results
        num_alpha_samples: Number of alpha samples to use
        max_eval_samples: Maximum number of validation samples per alpha
        device: Device to run on (cuda or cpu)
        num_workers: Number of parallel workers (default: 5)
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Using {num_workers} parallel workers")
    logger.info(f"Analyzing {len(model_pairs)} model pair(s)")
    
    # Use ThreadPoolExecutor to parallelize analysis
    completed_pairs = []
    failed_pairs = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for pair_idx, (model1_path, model2_path) in enumerate(model_pairs):
            future = executor.submit(
                analyze_pair,
                pair_idx,
                len(model_pairs),
                model1_path,
                model2_path,
                config_file,
                output_path,
                num_alpha_samples,
                max_eval_samples,
                device,
            )
            future_to_idx[future] = pair_idx
        
        # Process completed futures as they finish
        for future in as_completed(future_to_idx):
            try:
                pair_idx, pair_name, success, error_msg = future.result()
                if success:
                    completed_pairs.append(pair_name)
                    logger.info(f"✓ Completed: {pair_name}")
                else:
                    failed_pairs.append((pair_name, error_msg))
                    logger.error(f"✗ Failed: {pair_name} - {error_msg}")
            except Exception as e:
                logger.error(f"Exception in worker thread: {e}", exc_info=True)
                failed_pairs.append((f"Task-{future_to_idx[future]}", str(e)))
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("Analysis Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Completed: {len(completed_pairs)}/{len(model_pairs)}")
    if completed_pairs:
        for pair_name in completed_pairs:
            logger.info(f"  ✓ {pair_name}")
    if failed_pairs:
        logger.warning(f"Failed: {len(failed_pairs)}")
        for pair_name, error_msg in failed_pairs:
            logger.warning(f"  ✗ {pair_name}")
            if error_msg:
                logger.warning(f"    Error: {error_msg}")
    logger.info(f"Results saved to {output_path}")


def generate_visualization(
    loss_results: Dict,
    metrics: Dict,
    output_dir: Path,
) -> None:
    """
    Generate visualization plots for loss landscape.
    
    Args:
        loss_results: Dict with loss curves
        metrics: Dict with computed metrics
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    logger.info("Generating visualization...")
    
    alpha = np.array(loss_results["alpha"])
    loss_total = np.array(loss_results["loss_total"])
    loss_cls = np.array(loss_results["loss_cls"])
    loss_box_reg = np.array(loss_results["loss_box_reg"])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Linear Mode Connectivity Analysis", fontsize=16, fontweight="bold")
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(alpha, loss_total, "b-o", linewidth=2, markersize=6)
    ax.axhline(y=metrics["endpoint_loss_max"], color="r", linestyle="--", label="Endpoint max")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss Landscape")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"barrier={metrics['barrier_height']:.4f}\nrel={metrics['normalized_barrier']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
    )
    ax.legend()
    
    # Classification loss
    ax = axes[0, 1]
    ax.plot(alpha, loss_cls, "g-s", linewidth=2, markersize=6)
    ax.axhline(y=metrics.get("endpoint_loss_max_cls", loss_cls[[0, -1]].max()), color="r", linestyle="--", label="Endpoint max")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Classification Loss")
    ax.set_title("Classification Loss")
    ax.grid(True, alpha=0.3)
    if "barrier_height_cls" in metrics:
        ax.text(
            0.02,
            0.95,
            f"barrier={metrics['barrier_height_cls']:.4f}\nrel={metrics['normalized_barrier_cls']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
        )
    ax.legend()
    
    # Regression loss
    ax = axes[1, 0]
    ax.plot(alpha, loss_box_reg, "r-^", linewidth=2, markersize=6)
    ax.axhline(y=metrics.get("endpoint_loss_max_bbox", loss_box_reg[[0, -1]].max()), color="r", linestyle="--", label="Endpoint max")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("BBox Regression Loss")
    ax.set_title("BBox Regression Loss")
    ax.grid(True, alpha=0.3)
    if "barrier_height_bbox" in metrics:
        ax.text(
            0.02,
            0.95,
            f"barrier={metrics['barrier_height_bbox']:.4f}\nrel={metrics['normalized_barrier_bbox']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
        )
    ax.legend()
    
    # Combined view
    ax = axes[1, 1]
    ax.plot(alpha, loss_total, "b-o", linewidth=2, markersize=6, label="Total")
    ax.plot(alpha, loss_cls, "g-s", linewidth=2, markersize=6, label="Classification")
    ax.plot(alpha, loss_box_reg, "r-^", linewidth=2, markersize=6, label="Regression")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Loss")
    ax.set_title("All Losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    plot_path = output_dir / "lmc_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Visualization saved to {plot_path}")


def main():
    """Main entry point."""
    parser = default_argument_parser()
    parser.add_argument(
        "--model-pairs",
        type=str,
        nargs="+",
        required=True,
        help="Model pairs to analyze. Format: 'model1_path:model2_path' or separate args",
    )
    parser.add_argument(
        "--num-alpha-samples",
        type=int,
        default=11,
        help="Number of alpha samples (default: 11)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/lmc_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help="Maximum number of validation samples to use per alpha (default: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    
    # Parse model pairs
    model_pairs = []
    for pair_str in args.model_pairs:
        if ":" in pair_str:
            # Format: "model1:model2"
            model1, model2 = pair_str.split(":", 1)
            model_pairs.append((model1.strip(), model2.strip()))
        else:
            # Assume it's a pair and next arg is the second model
            # This is handled by nargs="+"
            logger.warning(f"Could not parse model pair: {pair_str}")
    
    if not model_pairs:
        logger.error("No valid model pairs provided")
        parser.print_help()
        return
    
    logger.info(f"Analyzing {len(model_pairs)} model pair(s)")
    
    # Run analysis
    analyze_connectivity(
        config_file=args.config_file,
        model_pairs=model_pairs,
        output_dir=args.output_dir,
        num_alpha_samples=args.num_alpha_samples,
        max_eval_samples=args.max_eval_samples,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
