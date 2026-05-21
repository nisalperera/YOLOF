"""
Linear Mode Connectivity Analysis Module

Provides utilities for analyzing the loss landscape between two trained YOLOF models
by interpolating their weights and evaluating loss at interpolated points.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from yolof.utils import _format_duration
from yolof_soup.utils.logging_utils import setup_logging

logger = setup_logging(level=logging.INFO, filename="mode_connectivity.log", use_stdout=True)


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _batch_size(batch) -> int:
    if isinstance(batch, (list, tuple)):
        return len(batch)
    return 1


def _extract_loss_value(losses: Dict, key: str) -> float:
    if key not in losses:
        return 0.0
    value = losses[key]
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def _curve_metrics(loss_curve: np.ndarray) -> Dict[str, float]:
    loss_curve = np.asarray(loss_curve, dtype=float)

    max_loss = float(np.max(loss_curve))
    min_loss = float(np.min(loss_curve))
    mean_loss = float(np.mean(loss_curve))
    std_loss = float(np.std(loss_curve))

    endpoint_start = float(loss_curve[0])
    endpoint_end = float(loss_curve[-1])
    endpoint_loss_avg = (endpoint_start + endpoint_end) / 2.0
    endpoint_loss_max = max(endpoint_start, endpoint_end)
    barrier_height = max_loss - endpoint_loss_max
    normalized_barrier = barrier_height / max(endpoint_loss_max, 1e-12)

    return {
        "max_loss": max_loss,
        "min_loss": min_loss,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "endpoint_loss_avg": float(endpoint_loss_avg),
        "endpoint_loss_max": float(endpoint_loss_max),
        "barrier_height": float(barrier_height),
        "normalized_barrier": float(normalized_barrier),
        "peak_index": int(np.argmax(loss_curve)),
    }


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a YOLOF checkpoint and return its state dict."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Invalid checkpoint format: expected dict, got {type(checkpoint)}"
        )

    logger.info("Loaded state dict with %d parameters", len(state_dict))
    return state_dict


def validate_model_compatibility(state_dict1: Dict, state_dict2: Dict) -> bool:
    """Validate that two model state dicts have compatible architectures."""
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        error_msg = "Model architectures are incompatible:\n"
        if missing_in_2:
            error_msg += f"  Missing in model 2: {missing_in_2}\n"
        if missing_in_1:
            error_msg += f"  Missing in model 1: {missing_in_1}\n"
        raise ValueError(error_msg)

    for key in keys1:
        shape1 = state_dict1[key].shape
        shape2 = state_dict2[key].shape
        if shape1 != shape2:
            raise ValueError(
                f"Shape mismatch for parameter '{key}': {shape1} vs {shape2}"
            )

    logger.info("Model architectures validated successfully")
    return True


def interpolate_state_dict(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    alpha: float,
    include_buffers: bool = True,
) -> Dict[str, torch.Tensor]:
    """Linearly interpolate two state dicts using (1 - alpha) * a + alpha * b."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    validate_model_compatibility(state_dict_a, state_dict_b)

    interpolated_state_dict = {}
    for key in state_dict_a.keys():
        param_a = state_dict_a[key]
        param_b = state_dict_b[key]

        if not isinstance(param_a, torch.Tensor) or not isinstance(param_b, torch.Tensor):
            interpolated_state_dict[key] = param_a
            continue

        if param_a.shape != param_b.shape:
            raise ValueError(
                f"Shape mismatch for parameter '{key}': {param_a.shape} vs {param_b.shape}"
            )

        if torch.is_floating_point(param_a) and torch.is_floating_point(param_b):
            interpolated_state_dict[key] = (1.0 - alpha) * param_a + alpha * param_b
        else:
            interpolated_state_dict[key] = param_a

    return interpolated_state_dict


def interpolate_models(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Backward-compatible wrapper around interpolate_state_dict."""
    return interpolate_state_dict(state_dict1, state_dict2, alpha)


def evaluate_loss_on_dataset(
    model: nn.Module,
    dataloader,
    device: torch.device,
    return_val_loss: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate YOLOF losses on a dataset subset."""
    model.eval()

    total_loss_cls = 0.0
    total_loss_box_reg = 0.0
    total_loss = 0.0
    num_samples = 0

    logger.info("Starting loss evaluation on dataset...")

    eval_start = time.perf_counter()
    chunk_start = time.perf_counter()  # timer for each 1000-batch chunk

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_to_device(batch, device)
            batch_size = _batch_size(batch)

            outputs = model(batch, return_val_loss=return_val_loss)

            batch_losses = None
            if outputs is not None and isinstance(outputs, dict):
                batch_losses = outputs.get("losses", outputs)
            elif isinstance(outputs, list) and outputs:
                first_output = outputs[0]
                if isinstance(first_output, dict):
                    batch_losses = first_output.get("losses", first_output) if return_val_loss else None
            else:
                logger.warning("Unexpected output format from model: %s", type(outputs))

            if batch_losses is not None:
                loss_cls = _extract_loss_value(batch_losses, "loss_cls")
                loss_box_reg = _extract_loss_value(batch_losses, "loss_box_reg")
                total_loss_cls += loss_cls * batch_size
                total_loss_box_reg += loss_box_reg * batch_size
                total_loss += (loss_cls + loss_box_reg) * batch_size

            num_samples += batch_size
            if max_samples is not None and num_samples >= max_samples:
                # Log final partial chunk before breaking
                chunk_elapsed = time.perf_counter() - chunk_start
                logger.info(
                    "Processed %d batches... (chunk: %s, total: %s)",
                    batch_idx + 1,
                    _format_duration(chunk_elapsed),
                    _format_duration(time.perf_counter() - eval_start),
                )
                break

            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(dataloader.dataset._dataset) // dataloader.batch_size:
                chunk_elapsed = time.perf_counter() - chunk_start
                total_elapsed = time.perf_counter() - eval_start
                batches_done = batch_idx + 1
                batches_total = len(dataloader.dataset._dataset) // dataloader.batch_size
                # Estimate remaining time based on average batch time so far
                avg_per_batch = total_elapsed / batches_done
                remaining_batches = batches_total - batches_done
                eta = avg_per_batch * remaining_batches

                logger.info(
                    "Processed %d/%d batches | chunk: %s | elapsed: %s | ETA: %s",
                    batches_done,
                    batches_total,
                    _format_duration(chunk_elapsed),
                    _format_duration(total_elapsed),
                    _format_duration(eta),
                )
                chunk_start = time.perf_counter()  # reset chunk timer

    total_elapsed = time.perf_counter() - eval_start
    avg_loss_cls = total_loss_cls / max(1, num_samples)
    avg_loss_box_reg = total_loss_box_reg / max(1, num_samples)
    avg_loss_total = total_loss / max(1, num_samples)

    results = {
        "loss_cls": avg_loss_cls,
        "loss_box_reg": avg_loss_box_reg,
        "loss_total": avg_loss_total,
        "num_samples": num_samples,
        "eval_time_seconds": total_elapsed,
    }

    logger.info("Loss evaluation complete in %s. Samples: %d", _format_duration(total_elapsed), num_samples)
    logger.info("  Classification loss: %.6f", avg_loss_cls)
    logger.info("  Regression loss:     %.6f", avg_loss_box_reg)
    logger.info("  Total loss:          %.6f", avg_loss_total)

    return results


def compute_connectivity_metrics(
    alpha_values: np.ndarray,
    loss_curve: np.ndarray,
    loss_cls_curve: Optional[np.ndarray] = None,
    loss_bbox_curve: Optional[np.ndarray] = None,
    loss_reg_curve: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute total, cls, and bbox LMC barriers for sampled alpha values."""
    if loss_bbox_curve is None:
        loss_bbox_curve = loss_reg_curve

    metrics = _curve_metrics(loss_curve)
    metrics["loss_range"] = metrics["max_loss"] - metrics["min_loss"]

    if len(alpha_values) >= 3:
        first_deriv = np.diff(loss_curve) / np.diff(alpha_values)
        second_deriv = np.diff(first_deriv) / np.diff(alpha_values[:-1])
        metrics["curvature"] = float(np.mean(np.abs(second_deriv)))
    else:
        metrics["curvature"] = 0.0

    if len(alpha_values) == len(loss_curve):
        metrics["peak_alpha"] = float(alpha_values[int(metrics["peak_index"])])
    else:
        metrics["peak_alpha"] = float("nan")

    if loss_cls_curve is not None:
        cls_metrics = _curve_metrics(loss_cls_curve)
        metrics.update(
            {
                "max_loss_cls": cls_metrics["max_loss"],
                "min_loss_cls": cls_metrics["min_loss"],
                "mean_loss_cls": cls_metrics["mean_loss"],
                "std_loss_cls": cls_metrics["std_loss"],
                "endpoint_loss_avg_cls": cls_metrics["endpoint_loss_avg"],
                "endpoint_loss_max_cls": cls_metrics["endpoint_loss_max"],
                "barrier_height_cls": cls_metrics["barrier_height"],
                "normalized_barrier_cls": cls_metrics["normalized_barrier"],
                "peak_index_cls": cls_metrics["peak_index"],
            }
        )
        if len(alpha_values) == len(loss_cls_curve):
            metrics["peak_alpha_cls"] = float(alpha_values[int(cls_metrics["peak_index"])])

    if loss_bbox_curve is not None:
        bbox_metrics = _curve_metrics(loss_bbox_curve)
        metrics.update(
            {
                "max_loss_bbox": bbox_metrics["max_loss"],
                "min_loss_bbox": bbox_metrics["min_loss"],
                "mean_loss_bbox": bbox_metrics["mean_loss"],
                "std_loss_bbox": bbox_metrics["std_loss"],
                "endpoint_loss_avg_bbox": bbox_metrics["endpoint_loss_avg"],
                "endpoint_loss_max_bbox": bbox_metrics["endpoint_loss_max"],
                "barrier_height_bbox": bbox_metrics["barrier_height"],
                "normalized_barrier_bbox": bbox_metrics["normalized_barrier"],
                "peak_index_bbox": bbox_metrics["peak_index"],
                "max_loss_reg": bbox_metrics["max_loss"],
                "min_loss_reg": bbox_metrics["min_loss"],
                "mean_loss_reg": bbox_metrics["mean_loss"],
                "std_loss_reg": bbox_metrics["std_loss"],
                "endpoint_loss_avg_reg": bbox_metrics["endpoint_loss_avg"],
                "endpoint_loss_max_reg": bbox_metrics["endpoint_loss_max"],
                "barrier_height_reg": bbox_metrics["barrier_height"],
                "normalized_barrier_reg": bbox_metrics["normalized_barrier"],
                "peak_index_reg": bbox_metrics["peak_index"],
            }
        )
        if len(alpha_values) == len(loss_bbox_curve):
            peak_alpha = float(alpha_values[int(bbox_metrics["peak_index"])])
            metrics["peak_alpha_bbox"] = peak_alpha
            metrics["peak_alpha_reg"] = peak_alpha

    return metrics
