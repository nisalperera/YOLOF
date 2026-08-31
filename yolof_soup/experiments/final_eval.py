from __future__ import annotations

from datetime import datetime
import os
import math
import copy
import json
import time
import random
import logging
import itertools
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

import numpy as np

from tabulate import tabulate

from yolof.utils import _format_duration

from yolof_soup.config.experiment_config import (
    DEBUG,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    COCO_EVAL_DATASET,
    OBJECTS365_DATASET,
    PHASE2_OUTPUT_DIR,
    _register_datasets,
    build_eval_cfg,
)

from yolof_soup.utils.checkpoint_utils import load_state, load_states
from yolof_soup.config.experiment_registry import get_run_specs

from yolof_soup.utils.gpu_memory_usage import GPUMemoryMonitor
from yolof_soup.utils.global_logger import get_logger

from yolof_soup.experiments.soup_construction import MERGE_CONDITIONS, evaluate_condition

import warnings

# Suppress the specific torch.meshgrid warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*?torch.meshgrid.*?")


logger = None  # Will be initialized in main()

def setup_logger(verbose: bool = True):
    """Set up a global logger for the module."""
    global logger
    logger = get_logger(
        level=logging.DEBUG if verbose else logging.INFO,
        add_file_handler=True,
    )

logger = setup_logger(DEBUG)

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


def evaluate_merged_conditions(checkpoint_dir, cfg, N_COLS, dataset_name, batch_size):
    checkpoints = {}
    
    checkpoints["condition_1_state"] = load_state(checkpoint_dir / "global_uniform_soup.pth")
    checkpoints["condition_2_state"] = load_state(checkpoint_dir / "branch_uniform_soup.pth")
    checkpoints["condition_3_state"] = load_state(checkpoint_dir / "dirichlet_soup.pth")
    checkpoints["condition_4_state"] = load_state(checkpoint_dir / "fisher_soup.pth")
    checkpoints["condition_5_state"] = load_state(checkpoint_dir / "uniform_learned_soup.pth")
    checkpoints["condition_6_state"] = load_state(checkpoint_dir / "learned_tri_head_soup.pth")

    logger.info("  ✓ Loaded all 6 conditions from cached checkpoints")
    logger.info("\n[6/7] Evaluating conditions...")
    map_results = {}
    for i, condition in enumerate(list(MERGE_CONDITIONS.keys())):
        logger.info(f"Evaluating {condition} soup")
        checkpoint_name = f"condition_{i + 1}_state"
        if checkpoint_name not in checkpoints:
            logger.info(f"  → Checkpoint for {condition} not found — skipping evaluation")
            continue

        start_eval = time.perf_counter()
        condition_name = f"condition_{i + 1}"
        map_results[f"results_cond{i+1}"] = evaluate_condition(checkpoints[checkpoint_name], cfg, f"final_eval/{condition_name}", 
                                                            dataset_name=dataset_name, logger=logger, batch_size=batch_size)

        logger.info("\nResults summary:")
        logger.info("  Condition %i (%s):  mAP50:95=%.4f", i + 1, condition.capitalize(), map_results[f"results_cond{i+1}"]["map50_95"])

        # Log per-class AP table for best condition
        results_flatten = list(itertools.chain(*map_results[f"results_cond{i+1}"]["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d, tablefmt="pipe", floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2), numalign="left",
        )
        logger.info("\nCondition %i (%s) per-class AP (sample):\n%s", i + 1, condition.capitalize(), table)
        logger.info("%s Evaluation completed in: %s", condition.capitalize(), _format_duration(time.perf_counter() - start_eval))

    # ── Save results JSON ─────────────────────────────────────────────────
    results_summary = {
        "condition_1": map_results["results_cond1"],
        "condition_2": map_results["results_cond2"],
        "condition_3": map_results["results_cond3"],
        "condition_4": map_results["results_cond4"],
        "condition_5": map_results["results_cond5"],
        "condition_6": map_results["results_cond6"]
    }

    results_path = Path(RESULTS_DIR) / f"phase3_soup_final_eval_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    logger.info("  ✓ Results saved → %s", results_path)


def evaluate_ingredients(cfg, N_COLS, dataset_name, batch_size, role="ingredient"):
    ingredients_dir = Path(PHASE2_OUTPUT_DIR)
    
    map_results = {}

    # ── Load ingredient checkpoints ───────────────────────────────────────
    logger.info("\n[1/2] Loading evaluation specs...")
    run_registry = get_run_specs()
    runs = [r for r in run_registry if r.role == role]

    logger.info("\n[2/2] Starting Evaluation...")
    for i, run in enumerate(runs):
        logger.info(f"Evaluating ingredient: {run.run_name}")

        start_eval = time.perf_counter()
        if not run.checkpoint_path.exists():
            logger.error("Missing checkpoint for %s at %s", run.run_name, run.checkpoint_path)
            continue

        state_dict = load_state(run.checkpoint_path)
        map_results[run.run_name] = evaluate_condition(state_dict, cfg, f"final_eval/{run.run_name}", 
                                                            dataset_name=dataset_name, logger=logger, batch_size=batch_size)

        logger.info("\nResults summary:")
        logger.info("  %s %i (%s):  mAP50:95=%.4f", role.replace("_", " ").title(), i + 1, run.run_name.capitalize(), map_results[run.run_name]["map50_95"])

        # Log per-class AP table for best condition
        results_flatten = list(itertools.chain(*map_results[run.run_name]["per_class_ap"]))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d, tablefmt="pipe", floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2), numalign="left",
        )
        logger.info("%s %i (%s) per-class AP (sample):\n%s", role.replace("_", " ").title(), i+1, run.run_name.capitalize(), table)
        logger.info("%s Evaluation completed in: %s", run.run_name.capitalize(), _format_duration(time.perf_counter() - start_eval))

    # ── Save results JSON ─────────────────────────────────────────────────
    results_path = Path(RESULTS_DIR) / f"{role}_final_eval_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(results_path, "w") as f:
        json.dump(map_results, f, indent=2, default=str)
    logger.info("  ✓ Results saved → %s", results_path)

# ── Evaluate conditions ─────────────────────────────────────────
def final_eval(merged: bool = True, ingredient: bool = True, decoder_finetune: bool = True, dataset: str = COCO_EVAL_DATASET, batch_size: int = 16):

    results_dir = Path(RESULTS_DIR);    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(CHECKPOINT_DIR); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_eval_cfg()
    N_COLS = min(6, cfg.MODEL.YOLOF.DECODER.NUM_CLASSES * 2)

    if merged:
        evaluate_merged_conditions(checkpoint_dir, cfg, N_COLS, dataset, batch_size)

    if ingredient:
        evaluate_ingredients(cfg, N_COLS, dataset, batch_size, "ingredient")

    if decoder_finetune:
        evaluate_ingredients(cfg, N_COLS, dataset, batch_size, "decoder_finetune")
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Soup Construction & Evaluation")
    parser.add_argument("--merged", action="store_true", default=False, help="Evaluate merged conditions (M1-M6)")
    parser.add_argument("--ingredient", action="store_true", default=False, help="Evaluate ingredient models (L1-L4, R1-R2)")
    parser.add_argument("--decoder_finetune", action="store_true", default=False, help="Evaluate Decoder Finetune runs (D1-D2, C3)")
    parser.add_argument("--dataset", type=str, default=COCO_EVAL_DATASET, choices=[COCO_EVAL_DATASET, OBJECTS365_DATASET], help="Dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parsed_args = parser.parse_args()

    _register_datasets()
    gpu_monitor = GPUMemoryMonitor(interval=30, verbose=parsed_args.verbose)
    set_seed(parsed_args.seed)
    DEBUG = parsed_args.verbose
    setup_logger(verbose=parsed_args.verbose)
    
    final_eval(False, False, True, parsed_args.dataset, parsed_args.batch_size)