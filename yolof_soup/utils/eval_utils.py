"""
utils/eval_utils.py
====================
COCO evaluation and quick loss proxy, wired to nisalperera/YOLOF.

Public API:
  build_eval_dataloader(cfg, dataset_name)  → DataLoader
  compute_coco_map(model, cfg, ...)         → {AP, AP50, AP75, APs, APm, APl}
  get_map(model, cfg, ...)                  → scalar AP (mAP50:95)
  quick_loss(model, dataloader, device)     → scalar mean total loss

Wraps:
  yolof.evaluation.coco_ar_ap.COCOEvaluatorWithAPandAR
  yolof.analysis.mode_connectivity.evaluate_loss_on_dataset
  yolof.data.YOLOFDatasetMapper
  yolof.data.samplers.EvenlyDistributedInferenceSampler

Fix log:
  - extract_per_class_ap: corrected the 80-class COCO category list;
    the previous list contained two placeholder "??" entries at the end
    caused by a category-index off-by-one, resulting in wrong per-class
    AP extraction for 'hair drier' (idx 78) and 'toothbrush' (idx 79).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

from yolof.analysis.mode_connectivity import evaluate_loss_on_dataset
from yolof.data import YOLOFDatasetMapper
from yolof.data.samplers import EvenlyDistributedInferenceSampler
from yolof.evaluation.coco_ar_ap import COCOEvaluatorWithAPandAR
from yolof_soup.utils.global_logger import get_logger

logger = get_logger()

# Canonical 80-class COCO category list (indices 0–79, contiguous).
# Source: COCO API categories sorted by contiguous id.
_COCO_CATEGORIES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
assert len(_COCO_CATEGORIES) == 80, "COCO category list must have exactly 80 entries."


def build_eval_dataloader(cfg, dataset_name: Optional[str] = None):
    """
    Build a Detectron2 test DataLoader using YOLOFDatasetMapper.

    Args:
        cfg:          Detectron2 CfgNode (fully merged).
        dataset_name: Override; defaults to cfg.DATASETS.TEST[0].

    Returns:
        DataLoader compatible with both compute_coco_map and quick_loss.
    """
    name    = dataset_name or cfg.DATASETS.TEST[0]
    mapper  = YOLOFDatasetMapper(cfg, is_train=False)
    sampler = EvenlyDistributedInferenceSampler(len(DatasetCatalog.get(name)))
    return build_detection_test_loader(cfg, name, mapper=mapper, sampler=sampler)


def compute_coco_map(
    model,
    cfg,
    dataset_name: str,
    output_dir: str | Path,
    tag: str = "eval",
) -> Dict[str, float]:
    """
    Full COCO detection evaluation via COCOEvaluatorWithAPandAR.

    Results are written to  output_dir/inference/<tag>/.

    Returns:
        Dict with keys AP, AP50, AP75, APs, APm, APl (values in [0, 100]).
    """
    eval_dir = Path(output_dir) / "inference" / tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    loader    = build_eval_dataloader(cfg, dataset_name)
    evaluator = COCOEvaluatorWithAPandAR(dataset_name, output_dir=str(eval_dir))

    model.eval()
    results = inference_on_dataset(model, loader, evaluator)
    bbox    = results.get("bbox", {})
    logger.info("[compute_coco_map] tag=%-28s  AP=%.4f", tag, bbox.get("AP", float("nan")))
    return bbox


def get_map(
    model: torch.nn.Module,
    cfg,
    dataset_name: str,
    output_dir: str | Path = "/tmp/eval",
    tag: str = "quick_map",
) -> float:
    """
    Convenience wrapper: return scalar mAP50:95 (the 'AP' key).
    """
    return float(compute_coco_map(model, cfg, dataset_name, output_dir, tag).get("AP", 0.0))


def extract_per_class_ap(
    results_dict: Dict[str, float],
    categories: Optional[List[str]] = None,
) -> List[float]:
    """
    Extract per-class AP values from COCO evaluator results dict.

    The evaluator stores results as "AP-{class_name}" for each of the
    80 contiguous COCO classes.  This function maps them back to a
    length-80 list in canonical category order.

    Args:
        results_dict: Dict returned from compute_coco_map().
        categories:   Optional override for the category list.  Defaults
                      to the canonical 80-class COCO list.

    Returns:
        List[float] of length 80; missing classes are filled with 0.0.
    """
    cats = categories if categories is not None else _COCO_CATEGORIES
    per_class_ap: List[float] = []

    for class_name in cats:
        key = f"AP-{class_name}"
        per_class_ap.append(float(results_dict.get(key, 0.0)))

    if all(v == 0.0 for v in per_class_ap):
        # Fallback: some evaluators use integer class-index keys
        logger.debug(
            "No AP-{class_name} keys found in results; "
            "falling back to AP-{class_idx} keys."
        )
        per_class_ap = [
            float(results_dict.get(f"AP-{idx}", 0.0))
            for idx in range(len(cats))
        ]

    return per_class_ap


def quick_loss(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_samples: Optional[int] = 1000,
) -> float:
    """
    Mean total detection loss (loss_cls + loss_box_reg) on a data subset.

    Delegates to yolof.analysis.mode_connectivity.evaluate_loss_on_dataset,
    which handles the model's return_val_loss flag and loss key extraction.

    Args:
        model:       YOLOF model; will be moved to *device*.
        dataloader:  Pre-built eval DataLoader.
        device:      Torch device.
        max_samples: Hard cap on images processed (for speed).

    Returns:
        Scalar mean loss (lower = better).
    """
    model.to(device)
    r = evaluate_loss_on_dataset(
        model, dataloader, device,
        return_val_loss=True,
        max_samples=max_samples,
    )
    total = float(r.get("loss_total", float("inf")))
    logger.debug("quick_loss=%.5f  (n=%d)", total, r.get("num_samples", -1))
    return total
