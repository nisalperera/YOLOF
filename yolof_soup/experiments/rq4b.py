from __future__ import annotations

from datetime import datetime
import re
import os
import time
import json
import random
import logging
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch.multiprocessing as mp

import torch


from yolof.utils import _format_duration

from yolof_soup.config.experiment_config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    CALIB_DATASET,
    PHASE2_OUTPUT_DIR,
    _register_datasets,
    build_eval_cfg,
)
from yolof_soup.config.experiment_registry import get_run_specs
from yolof_soup.utils.checkpoint_utils import load_states
from yolof_soup.utils.eval_utils import build_eval_dataloader
from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor
from yolof_soup.utils.global_logger import get_logger
from yolof_soup.experiments.soup_construction import build_tri_head_learned_soup

import warnings

# Suppress the specific torch.meshgrid warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*?torch.meshgrid.*?")


logger = None

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (can be adjusted)
# ─────────────────────────────────────────────────────────────────────────────

#: Whether to use Hessian (expensive) or L2 proxy for Fisher weights
USE_HESSIAN_FOR_FISHER: bool = True

#: Learned Soup hyperparameters (Condition 5)
LEARNED_SOUP_LR: float = 0.0005
LEARNED_SOUP_EPOCHS: int = 20
LEARNED_SOUP_PATIENCE: int = 5
LEARNED_SOUP_BATCH_SIZE: int = 16

# NEW: Regularization hyperparameters (academic literature)
LEARNED_SOUP_ENTROPY_WEIGHT: float = 0.1      # KL penalty toward uniform (prevent collapse)
LEARNED_SOUP_TEMP_MIN: float = 0.5            # Min temperature β (Guo et al., 2017)
LEARNED_SOUP_TEMP_MAX: float = 2.0            # Max temperature β  
LEARNED_SOUP_GRAD_CLIP: float = 1.0           # Gradient clipping to prevent divergence
LEARNED_SOUP_ALPHA_THRESHOLD: float = 0.05    # Warn if ingredient weight < 5% (ingredient filtering)

CORE_GROUPS = [
    list(range(0, 4)),    # Model 0 → cores 0-3
    list(range(4, 8)),    # Model 1 → cores 4-7
    list(range(8, 12)),   # Model 2 → cores 8-11
    list(range(12, 16)),  # Model 3 → cores 12-15
    list(range(16, 20)),  # Model 4 → cores 16-19
    list(range(20, 24)),  # Model 5 → cores 20-23
]


def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducible experiments.
    """
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set Python built-in random generator
    random.seed(seed)
    
    # 3. Set NumPy random generator
    np.random.seed(seed)
    
    # 4. Set PyTorch random generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # 5. Configure CuDNN for determinism (Note: This may slightly reduce training speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def setup_logger(verbose: bool = True):
    """Set up a global logger for the module."""
    global logger
    logger = get_logger(
        level=logging.DEBUG if verbose else logging.INFO,
        add_file_handler=True,
    )


def get_latest_json_file(directory: str, pattern: str = r"phase3_soup_results_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json") -> Path:
    """Return the most recent file matching the timestamped pattern, parsed from the filename itself."""
    dir_path = Path(directory)
    candidates = []
    regex = re.compile(pattern)

    for file_path in dir_path.glob("phase3_soup_results_*.json"):
        match = regex.match(file_path.name)
        if match:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
            candidates.append((timestamp, file_path))

    if not candidates:
        raise FileNotFoundError(f"No files matching pattern found in {directory}")

    candidates.sort(key=lambda item: item[0])
    latest_timestamp, latest_path = candidates[-1]
    return latest_path


def load_latest_soup_results(directory: str) -> tuple[dict, Path]:
    latest_path = get_latest_json_file(directory)
    with open(latest_path) as handle:
        data = json.load(handle)
    return data, latest_path

def run(verbose: bool = True, seed: int = 42) -> bool:
    """
    Main Phase 3 entry point.

    Args:
        verbose:            Whether to log debug-level progress.

    Returns:
        Dict with results for all 5 conditions + metadata.
    """

    start_time = time.perf_counter()
    try:

        setup_logger(verbose=verbose)


        logger.info("=" * 90)
        logger.info("PHASE 3: SOUP CONSTRUCTION & EVALUATION")
        logger.info("=" * 90)

        results_dir = Path(RESULTS_DIR);    results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(CHECKPOINT_DIR); checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ingredients_dir = Path(PHASE2_OUTPUT_DIR)

        # ── Load ingredient checkpoints ───────────────────────────────────────
        logger.info("\n[1/4] Loading ingredient checkpoints...")
        run_registry = get_run_specs()
        ingredient_runs = [r for r in run_registry if r.role == "ingredient"]
        ingredient_paths = []

        for run_spec in ingredient_runs:
            ckpt_path = Path(ingredients_dir) / f"{run_spec.run_name}/model_best.pth"
            ingredient_paths.append(ckpt_path)

        missing = [p for p in ingredient_paths if not p.exists()]
        if missing:
            logger.error("Missing checkpoints: %s", missing)
            raise FileNotFoundError(
                f"Missing phase 2 checkpoints. Expected at: {ingredient_paths[0].parent}/"
            )

        ingredient_states = load_states(ingredient_paths)
        logger.info("  ✓ Loaded %d ingredients", len(ingredient_states))

        # ── Build config ──────────────────────────────────────────────────────
        logger.info("\n[2/4] Building Detectron2 config...")
        cfg = build_eval_cfg()
        logger.info("  ✓ Config ready")

        # ── Build dataloaders ─────────────────────────────────────────────────
        logger.info("\n[3/4] Building dataloaders...")
        # NEW: Use CALIB_DATASET for learned soup optimization (validation-based learning)
        # This provides better generalization than training on CALIB_DATASET
        calib_dataloader = build_eval_dataloader(
            cfg, CALIB_DATASET, batch_size=LEARNED_SOUP_BATCH_SIZE
        )

        if hasattr(calib_dataloader.dataset, "sampler"):
            calib_dataset_size = calib_dataloader.dataset.sampler._size
        else:
            calib_dataset_size = len(calib_dataloader.dataset._dataset)

        logger.info(
            "  ✓ Dataloaders ready — Calibration: %d batches",
            # int(train_dataset_size / train_dataloader.batch_size),
            int(calib_dataset_size / calib_dataloader.batch_size),
        )

        soup_results, latest_json_path = load_latest_soup_results(str(RESULTS_DIR))
        condition_6_record = soup_results["condition_6"]
        logger.info("  ✓ Loaded latest condition_6 record from %s", latest_json_path)

        beta_replicates = []
        alpha_beta_replicates_raw = []

        for i in range(10):
            seed_i = seed + 100 + (i * 1000)
            logger.info("\n[4/4] Building learned soup replicate %d (seed=%d)...", i + 1, seed_i)
            set_seed(seed_i)  # Reset seed for reproducibility
            _, metadata = build_tri_head_learned_soup(ingredient_states, cfg, calib_dataloader, logger=logger)

            for key, value in metadata.items():
                logger.info("  ✓ %s: %s", key.replace("_", " ").title(), value)

            beta_replicates.append({
                "cls": float(metadata["beta_cls"]),
                "bbox": float(metadata["beta_bbox"]),
                "obj": float(metadata["beta_obj"]),
            })
            alpha_beta_replicates_raw.append({
                "seed": seed_i,
                **metadata,
            })

        # Attach to your condition_6 record before writing phase output
        condition_6_record["beta_replicates"] = beta_replicates
        condition_6_record["alpha_beta_replicates_raw"] = alpha_beta_replicates_raw  # optional, full audit trail

        soup_results["condition_6"] = condition_6_record
        with open(latest_json_path, "w") as handle:
            json.dump(condition_6_record, handle, indent=2)

        logger.info("Saved beta replicates (n=%d) to %s", len(beta_replicates), latest_json_path)

        logger.info("  ✓ Conditions 1-6 built")
        logger.info("  ✓ All 6 conditions built")

        
        logger.info("\n" + "=" * 90)
        logger.info("PHASE 3 COMPLETE")
        logger.info("=" * 90)

        total_elapsed = time.perf_counter() - start_time
        logger.info("Total elapsed time: %s", _format_duration(total_elapsed))
        return True

    except Exception:
        if logger:
            logger.exception("Phase 3 failed.")

        total_elapsed = time.perf_counter() - start_time
        logger.info("Total elapsed time: %s", _format_duration(total_elapsed))
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

parsed_args = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Soup Construction & Evaluation")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parsed_args = parser.parse_args()

    _register_datasets()
    gpu_monitor = GPUMemoryMonitor(interval=30, verbose=parsed_args.verbose)
    gpu_monitor.start()

    run(verbose=True, seed=parsed_args.seed)