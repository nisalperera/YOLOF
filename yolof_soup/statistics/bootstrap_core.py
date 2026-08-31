#!/usr/bin/env python3
"""
Shared paired image-level bootstrap engine for COCO object-detection
model comparisons.

Design principle
-----------------
The independent resampling unit is one held-out evaluation IMAGE, including
all of its ground-truth annotations and all stored detections for every
compared model. COCO categories, per-class AP values, and individual
bounding boxes are NOT treated as independent observations, because:

  - category-level AP values are correlated components of one aggregate
    mAP estimate,
  - multiple objects in the same image share scene context and detector
    failure modes,
  - class imbalance means categories are not exchangeable units.

Every script under yolof_soup/statistics that produces a formal
confirmatory p-value or confidence interval for a headline mAP50:95
comparison should import from this module rather than running
scipy.stats.ttest_rel / wilcoxon / f_oneway directly on per-class AP
arrays.

Usage pattern
-------------
    from yolof_soup.statistics.bootstrap_core import (
        load_json,
        run_comparison,
        holm_adjust,
    )

    ground_truth = load_json(gt_path)
    bootstrap_image_ids = np.load(manifest_path)

    result = run_comparison(
        name="H1-A: Condition 1 vs Condition 6 (M6)",
        model_a_name="Condition 1",
        model_a_predictions_path=Path("results/predictions/condition_1.json"),
        model_b_name="M6",
        model_b_predictions_path=Path("results/predictions/condition_6.json"),
        ground_truth=ground_truth,
        ground_truth_path=gt_path,
        bootstrap_image_ids=bootstrap_image_ids,
    )

Bootstrap draws must be generated once via generate_coco_bootstrap_manifest.py
and reused across every H1/H3/H4a/H4c comparison so that all pairwise
contrasts share the same resampling scheme (paired design).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_MAX_DETS = [1, 10, 100]


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def index_records_by_image(
    records: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        indexed[record["image_id"]].append(record)
    return indexed


def validate_dataset(
    ground_truth: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], list[int]]:
    required_keys = {"images", "annotations", "categories"}
    missing_keys = required_keys - set(ground_truth)
    if missing_keys:
        raise ValueError(f"Ground truth is missing required COCO keys: {sorted(missing_keys)}")

    image_by_id = {image["id"]: image for image in ground_truth["images"]}
    if len(image_by_id) != len(ground_truth["images"]):
        raise ValueError("Ground truth contains duplicate image IDs.")

    annotations_by_image = index_records_by_image(ground_truth["annotations"])

    invalid_annotation_ids = {
        annotation["image_id"]
        for annotation in ground_truth["annotations"]
        if annotation["image_id"] not in image_by_id
    }
    if invalid_annotation_ids:
        raise ValueError(
            "Annotations reference unknown image IDs: "
            f"{sorted(invalid_annotation_ids)[:20]}"
        )

    return image_by_id, annotations_by_image, sorted(image_by_id.keys())


def validate_predictions(
    prediction_path: Path,
    valid_image_ids: set[int],
) -> list[dict[str, Any]]:
    predictions = load_json(prediction_path)
    if not isinstance(predictions, list):
        raise TypeError(f"Predictions must be a COCO result list: {prediction_path}")

    required_keys = {"image_id", "category_id", "bbox", "score"}
    unknown_image_ids: set[int] = set()

    for index, prediction in enumerate(predictions):
        missing = required_keys - set(prediction)
        if missing:
            raise ValueError(
                f"Prediction {index} in {prediction_path} is missing keys: {sorted(missing)}"
            )
        if prediction["image_id"] not in valid_image_ids:
            unknown_image_ids.add(prediction["image_id"])

    if unknown_image_ids:
        raise ValueError(
            f"{prediction_path} includes detections for image IDs outside the "
            f"held-out ground truth. Examples: {sorted(unknown_image_ids)[:20]}"
        )

    return predictions


def build_synthetic_resample(
    sampled_original_image_ids: np.ndarray,
    image_by_id: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    predictions_a_by_image: dict[int, list[dict[str, Any]]],
    predictions_b_by_image: dict[int, list[dict[str, Any]]],
    ground_truth_template: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build one bootstrap-replicate COCO dataset.

    Repeated original images receive distinct synthetic image IDs because
    COCOeval collapses duplicated image IDs otherwise. Ground truth and
    every model's saved detections are duplicated identically for each
    sampled occurrence, preserving the paired design.
    """
    synthetic_images = []
    synthetic_annotations = []
    synthetic_predictions_a = []
    synthetic_predictions_b = []

    next_image_id = 1
    next_annotation_id = 1

    for original_image_id in sampled_original_image_ids:
        original_image_id = int(original_image_id)
        synthetic_image_id = next_image_id
        next_image_id += 1

        synthetic_image = dict(image_by_id[original_image_id])
        synthetic_image["id"] = synthetic_image_id
        synthetic_images.append(synthetic_image)

        for annotation in annotations_by_image.get(original_image_id, []):
            copied = dict(annotation)
            copied["id"] = next_annotation_id
            copied["image_id"] = synthetic_image_id
            synthetic_annotations.append(copied)
            next_annotation_id += 1

        for prediction in predictions_a_by_image.get(original_image_id, []):
            copied = dict(prediction)
            copied["image_id"] = synthetic_image_id
            synthetic_predictions_a.append(copied)

        for prediction in predictions_b_by_image.get(original_image_id, []):
            copied = dict(prediction)
            copied["image_id"] = synthetic_image_id
            synthetic_predictions_b.append(copied)

    synthetic_ground_truth = {
        "info": ground_truth_template.get("info", {}),
        "licenses": ground_truth_template.get("licenses", []),
        "categories": ground_truth_template["categories"],
        "images": synthetic_images,
        "annotations": synthetic_annotations,
    }

    return synthetic_ground_truth, synthetic_predictions_a, synthetic_predictions_b


def evaluate_map50_95(
    ground_truth_path: Path,
    predictions_path: Path,
    max_dets: list[int] = DEFAULT_MAX_DETS,
) -> float:
    """Compute standard COCO bbox AP50:95, on a 0-100 scale."""
    coco_gt = COCO(str(ground_truth_path))
    coco_dt = coco_gt.loadRes(str(predictions_path))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.maxDets = max_dets
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return float(evaluator.stats[0] * 100.0)


_WORKER_CONTEXT: dict[str, Any] = {}


def _initialise_worker(
    ground_truth: dict[str, Any],
    predictions_a: list[dict[str, Any]],
    predictions_b: list[dict[str, Any]],
    max_dets: list[int],
) -> None:
    global _WORKER_CONTEXT
    image_by_id, annotations_by_image, _ = validate_dataset(ground_truth)
    _WORKER_CONTEXT = {
        "ground_truth": ground_truth,
        "image_by_id": image_by_id,
        "annotations_by_image": annotations_by_image,
        "predictions_a_by_image": index_records_by_image(predictions_a),
        "predictions_b_by_image": index_records_by_image(predictions_b),
        "max_dets": max_dets,
    }


def _evaluate_one_replicate(task: tuple[int, np.ndarray]) -> tuple[int, float, float]:
    replicate_index, sampled_original_image_ids = task
    context = _WORKER_CONTEXT

    synthetic_ground_truth, synthetic_predictions_a, synthetic_predictions_b = (
        build_synthetic_resample(
            sampled_original_image_ids=sampled_original_image_ids,
            image_by_id=context["image_by_id"],
            annotations_by_image=context["annotations_by_image"],
            predictions_a_by_image=context["predictions_a_by_image"],
            predictions_b_by_image=context["predictions_b_by_image"],
            ground_truth_template=context["ground_truth"],
        )
    )

    with tempfile.TemporaryDirectory(
        prefix=f"coco_bootstrap_{replicate_index:05d}_"
    ) as temp_dir_string:
        temp_dir = Path(temp_dir_string)
        gt_path = temp_dir / "ground_truth.json"
        pred_a_path = temp_dir / "predictions_a.json"
        pred_b_path = temp_dir / "predictions_b.json"

        with gt_path.open("w", encoding="utf-8") as f:
            json.dump(synthetic_ground_truth, f)
        with pred_a_path.open("w", encoding="utf-8") as f:
            json.dump(synthetic_predictions_a, f)
        with pred_b_path.open("w", encoding="utf-8") as f:
            json.dump(synthetic_predictions_b, f)

        map_a = evaluate_map50_95(gt_path, pred_a_path, context["max_dets"])
        map_b = evaluate_map50_95(gt_path, pred_b_path, context["max_dets"])

    return replicate_index, map_a, map_b


def bootstrap_p_values(differences: np.ndarray) -> tuple[float, float]:
    """
    Return (one-sided p for H_A: model_b > model_a, two-sided p).
    """
    b = len(differences)
    p_lower = (1.0 + float(np.sum(differences <= 0.0))) / (b + 1.0)
    p_upper = (1.0 + float(np.sum(differences >= 0.0))) / (b + 1.0)
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))
    return p_lower, p_two_sided


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        adjusted_value = (m - rank) * p_values[original_index]
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.zeros(m, dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def run_comparison(
    name: str,
    model_a_name: str,
    model_a_predictions_path: Path,
    model_b_name: str,
    model_b_predictions_path: Path,
    ground_truth: dict[str, Any],
    ground_truth_path: Path,
    bootstrap_image_ids: np.ndarray,
    max_dets: list[int] = DEFAULT_MAX_DETS,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Run one paired image-level bootstrap comparison: model_b vs model_a.

    Positive `observed_difference_ap_points` means model_b outperforms
    model_a on the full held-out set.
    """
    model_a_predictions_path = Path(model_a_predictions_path)
    model_b_predictions_path = Path(model_b_predictions_path)

    if not model_a_predictions_path.is_file():
        raise FileNotFoundError(f"Predictions not found for {model_a_name}: {model_a_predictions_path}")
    if not model_b_predictions_path.is_file():
        raise FileNotFoundError(f"Predictions not found for {model_b_name}: {model_b_predictions_path}")

    _, _, valid_image_ids_sorted = validate_dataset(ground_truth)
    valid_image_ids = set(valid_image_ids_sorted)

    predictions_a = validate_predictions(model_a_predictions_path, valid_image_ids)
    predictions_b = validate_predictions(model_b_predictions_path, valid_image_ids)

    observed_map_a = evaluate_map50_95(ground_truth_path, model_a_predictions_path, max_dets)
    observed_map_b = evaluate_map50_95(ground_truth_path, model_b_predictions_path, max_dets)
    observed_difference = observed_map_b - observed_map_a

    tasks = [
        (replicate_index, bootstrap_image_ids[replicate_index])
        for replicate_index in range(len(bootstrap_image_ids))
    ]

    if workers == 1:
        _initialise_worker(ground_truth, predictions_a, predictions_b, max_dets)
        replicate_results = [_evaluate_one_replicate(task) for task in tasks]
    else:
        start_method = "fork" if os.name != "nt" else "spawn"
        context = mp.get_context(start_method)
        with context.Pool(
            processes=workers,
            initializer=_initialise_worker,
            initargs=(ground_truth, predictions_a, predictions_b, max_dets),
        ) as pool:
            replicate_results = list(
                pool.imap_unordered(_evaluate_one_replicate, tasks, chunksize=1)
            )

    replicate_results.sort(key=lambda item: item[0])
    bootstrap_map_a = np.array([r[1] for r in replicate_results], dtype=np.float64)
    bootstrap_map_b = np.array([r[2] for r in replicate_results], dtype=np.float64)
    bootstrap_differences = bootstrap_map_b - bootstrap_map_a

    ci_lower, ci_upper = np.percentile(bootstrap_differences, [2.5, 97.5])
    p_one_sided, p_two_sided = bootstrap_p_values(bootstrap_differences)

    return {
        "name": name,
        "model_a": {
            "name": model_a_name,
            "prediction_file": str(model_a_predictions_path),
            "prediction_sha256": sha256_file(model_a_predictions_path),
            "observed_ap50_95": observed_map_a,
        },
        "model_b": {
            "name": model_b_name,
            "prediction_file": str(model_b_predictions_path),
            "prediction_sha256": sha256_file(model_b_predictions_path),
            "observed_ap50_95": observed_map_b,
        },
        "direction": f"{model_b_name} > {model_a_name}",
        "observed_difference_ap_points": observed_difference,
        "bootstrap": {
            "replicates": len(bootstrap_differences),
            "mean_difference_ap_points": float(bootstrap_differences.mean()),
            "median_difference_ap_points": float(np.median(bootstrap_differences)),
            "std_difference_ap_points": float(bootstrap_differences.std(ddof=1)),
            "ci_95_percentile_lower": float(ci_lower),
            "ci_95_percentile_upper": float(ci_upper),
            "p_one_sided_model_b_greater": float(p_one_sided),
            "p_two_sided": float(p_two_sided),
            "proportion_differences_positive": float(np.mean(bootstrap_differences > 0.0)),
        },
        "_bootstrap_differences": bootstrap_differences,
    }


def apply_decision_rule(
    result: dict[str, Any],
    p_holm_adjusted: float,
    alpha: float = 0.05,
    practical_threshold_ap_points: float = 0.5,
) -> dict[str, Any]:
    """Attach a Holm-adjusted decision to a run_comparison() result."""
    ci_lower = result["bootstrap"]["ci_95_percentile_lower"]
    observed_difference = result["observed_difference_ap_points"]

    statistically_significant = p_holm_adjusted < alpha
    ci_positive = ci_lower > 0.0
    practically_meaningful = observed_difference >= practical_threshold_ap_points

    result["bootstrap"]["p_one_sided_holm_adjusted"] = p_holm_adjusted
    result["decision"] = {
        "alpha_familywise": alpha,
        "practical_threshold_ap_points": practical_threshold_ap_points,
        "statistically_significant_holm_adjusted": statistically_significant,
        "ci_lower_bound_positive": ci_positive,
        "practically_meaningful": practically_meaningful,
        "support_directional_superiority": bool(
            statistically_significant and ci_positive and practically_meaningful
        ),
    }
    return result


def load_bootstrap_manifest_inputs(
    ground_truth_path: Path,
    bootstrap_image_ids_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Load and cross-validate the held-out ground truth and the bootstrap
    draws generated once by generate_coco_bootstrap_manifest.py.
    """
    ground_truth = load_json(ground_truth_path)
    _, _, image_ids_sorted = validate_dataset(ground_truth)

    bootstrap_image_ids = np.load(bootstrap_image_ids_path)
    if bootstrap_image_ids.ndim != 2:
        raise ValueError("bootstrap_image_ids.npy must have shape (B, N).")

    _, sample_size = bootstrap_image_ids.shape
    if sample_size != len(image_ids_sorted):
        raise ValueError(
            "Bootstrap sample size does not match held-out image count: "
            f"bootstrap N = {sample_size}, ground-truth N = {len(image_ids_sorted)}."
        )

    unknown_ids = set(np.unique(bootstrap_image_ids).tolist()) - set(image_ids_sorted)
    if unknown_ids:
        raise ValueError(
            f"Bootstrap draws reference unknown image IDs: {sorted(unknown_ids)[:20]}"
        )

    return ground_truth, bootstrap_image_ids


def save_result_artifacts(
    result: dict[str, Any],
    output_dir: Path,
    file_stem: str,
) -> dict[str, Any]:
    """Persist bootstrap differences and strip them from the JSON-safe dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    differences_path = output_dir / f"{file_stem}_bootstrap_differences.npy"
    np.save(differences_path, result["_bootstrap_differences"])
    result["bootstrap"]["differences_file"] = differences_path.name
    result.pop("_bootstrap_differences", None)
    return result
