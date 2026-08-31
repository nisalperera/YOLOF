#!/usr/bin/env python3
"""
Generate paired image-level bootstrap resample specifications for a
COCO-format held-out evaluation dataset.

This script does not rerun model inference and does not calculate AP.
It creates reproducible image-level bootstrap draws that can be applied
to every model's existing COCO-format predictions.

Inputs:
    1. Ground-truth COCO annotation JSON for the held-out evaluation subset.
    2. One or more COCO detection-result JSON files, used only for validation.

Outputs:
    bootstrap_draws.npy
        Shape: (B, N)
        Each row contains indices into the original sorted image-ID array.

    bootstrap_image_ids.npy
        Shape: (B, N)
        Each row contains sampled original COCO image IDs.

    bootstrap_manifest.json
        Dataset, prediction coverage, seed, and checksum metadata.

    bootstrap_summary.csv
        Per-replicate diagnostics: unique images, duplicates, etc.

Important:
    - The resampling unit is one complete image.
    - Sampling is with replacement.
    - The same bootstrap draws must be used for all compared models.
    - The generated draws do not alter or reorder model predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired image-level bootstrap manifests for "
            "COCO-format object-detection evaluation."
        )
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help=(
            "COCO-format ground-truth annotation JSON for the final "
            "held-out COCO validation subset."
        ),
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more standard COCO detection-result JSON files. "
            "These are validated for image-ID compatibility but are not "
            "modified by this script."
        ),
    )

    parser.add_argument(
        "--prediction-names",
        nargs="+",
        default=None,
        help=(
            "Optional model names in the same order as --predictions, "
            "for example: M6 M1 BestIngredient M5 C3."
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of bootstrap replicates B. Default: 5000.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed NumPy random seed. Default: 42.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bootstrap_manifest"),
        help="Directory for generated bootstrap files.",
    )

    parser.add_argument(
        "--write-first-replicate",
        action="store_true",
        help=(
            "Write first_replicate_image_ids.txt for quick manual "
            "inspection. This is useful for debugging only."
        ),
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return SHA-256 checksum for reproducibility tracking."""
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_ground_truth(ground_truth: dict[str, Any]) -> np.ndarray:
    """Validate required COCO ground-truth fields and return sorted IDs."""
    required_fields = {"images", "annotations", "categories"}
    missing_fields = required_fields - set(ground_truth)

    if missing_fields:
        raise ValueError(
            "Ground-truth JSON is missing required COCO fields: "
            f"{sorted(missing_fields)}"
        )

    image_records = ground_truth["images"]
    annotation_records = ground_truth["annotations"]
    category_records = ground_truth["categories"]

    if not image_records:
        raise ValueError("Ground-truth JSON contains no images.")

    if not category_records:
        raise ValueError("Ground-truth JSON contains no categories.")

    image_ids = [image["id"] for image in image_records]

    if len(image_ids) != len(set(image_ids)):
        duplicates = [
            image_id
            for image_id, count in Counter(image_ids).items()
            if count > 1
        ]
        raise ValueError(
            "Ground-truth JSON contains duplicate image IDs. "
            f"Examples: {duplicates[:10]}"
        )

    image_id_set = set(image_ids)
    invalid_annotation_image_ids = sorted(
        {
            annotation["image_id"]
            for annotation in annotation_records
            if annotation["image_id"] not in image_id_set
        }
    )

    if invalid_annotation_image_ids:
        raise ValueError(
            "Some annotations reference image IDs absent from images. "
            f"Examples: {invalid_annotation_image_ids[:20]}"
        )

    return np.array(sorted(image_ids), dtype=np.int64)


def validate_prediction_file(
    prediction_path: Path,
    valid_image_ids: set[int],
) -> dict[str, Any]:
    """
    Validate one COCO detection-result JSON file.

    COCO result JSON is normally a list of dictionaries:
    {
        "image_id": ...,
        "category_id": ...,
        "bbox": [x, y, w, h],
        "score": ...
    }
    """
    predictions = load_json(prediction_path)

    if not isinstance(predictions, list):
        raise ValueError(
            f"{prediction_path} is not a standard COCO result list."
        )

    prediction_image_ids = set()
    records_missing_required_fields = 0
    unknown_image_ids = set()

    for prediction in predictions:
        required_fields = {"image_id", "category_id", "bbox", "score"}

        if not required_fields.issubset(prediction):
            records_missing_required_fields += 1
            continue

        image_id = prediction["image_id"]
        prediction_image_ids.add(image_id)

        if image_id not in valid_image_ids:
            unknown_image_ids.add(image_id)

    if records_missing_required_fields:
        raise ValueError(
            f"{prediction_path} contains "
            f"{records_missing_required_fields} records missing at least one "
            "of: image_id, category_id, bbox, score."
        )

    if unknown_image_ids:
        raise ValueError(
            f"{prediction_path} contains detections for image IDs not present "
            f"in the held-out ground truth. Examples: "
            f"{sorted(unknown_image_ids)[:20]}"
        )

    return {
        "path": str(prediction_path.resolve()),
        "sha256": sha256_file(prediction_path),
        "detection_records": len(predictions),
        "images_with_at_least_one_prediction": len(prediction_image_ids),
        "images_without_predictions": len(valid_image_ids - prediction_image_ids),
        "unknown_prediction_image_ids": len(unknown_image_ids),
    }


def write_summary_csv(
    output_path: Path,
    bootstrap_image_ids: np.ndarray,
) -> None:
    """Write diagnostics for each bootstrap replicate."""
    original_n = bootstrap_image_ids.shape[1]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "replicate",
                "draws",
                "unique_images",
                "duplicate_draws",
                "unique_image_fraction",
            ],
        )
        writer.writeheader()

        for replicate_index, sampled_ids in enumerate(
            bootstrap_image_ids,
            start=1,
        ):
            unique_images = len(np.unique(sampled_ids))

            writer.writerow(
                {
                    "replicate": replicate_index,
                    "draws": original_n,
                    "unique_images": unique_images,
                    "duplicate_draws": original_n - unique_images,
                    "unique_image_fraction": (
                        f"{unique_images / original_n:.8f}"
                    ),
                }
            )


def main() -> None:
    args = parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero.")

    if not args.ground_truth.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {args.ground_truth}"
        )

    for prediction_path in args.predictions:
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Prediction file not found: {prediction_path}"
            )

    if (
        args.prediction_names is not None
        and len(args.prediction_names) != len(args.predictions)
    ):
        raise ValueError(
            "--prediction-names must contain exactly one name for each "
            "--predictions file."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_json(args.ground_truth)
    image_ids = validate_ground_truth(ground_truth)

    n_images = len(image_ids)
    image_id_set = set(image_ids.tolist())

    print(f"Ground-truth file: {args.ground_truth}")
    print(f"Held-out evaluation images: {n_images}")
    print(f"Bootstrap replicates B: {args.iterations}")
    print(f"Random seed: {args.seed}")

    prediction_metadata = {}

    for index, prediction_path in enumerate(args.predictions):
        model_name = (
            args.prediction_names[index]
            if args.prediction_names is not None
            else prediction_path.stem
        )

        prediction_info = validate_prediction_file(
            prediction_path=prediction_path,
            valid_image_ids=image_id_set,
        )

        prediction_metadata[model_name] = prediction_info

        print(
            f"Validated predictions for {model_name}: "
            f"{prediction_info['detection_records']} detections, "
            f"{prediction_info['images_with_at_least_one_prediction']} "
            "images with at least one prediction."
        )

    rng = np.random.default_rng(args.seed)

    # Each row contains N draws from {0, ..., N-1}, with replacement.
    # The indices refer to the sorted `image_ids` vector.
    bootstrap_draws = rng.integers(
        low=0,
        high=n_images,
        size=(args.iterations, n_images),
        dtype=np.int32,
    )

    # Convert index positions into original COCO image IDs.
    bootstrap_image_ids = image_ids[bootstrap_draws]

    draws_path = args.output_dir / "bootstrap_draws.npy"
    image_ids_path = args.output_dir / "bootstrap_image_ids.npy"
    summary_path = args.output_dir / "bootstrap_summary.csv"
    manifest_path = args.output_dir / "bootstrap_manifest.json"
    original_ids_path = args.output_dir / "original_heldout_image_ids.txt"

    np.save(draws_path, bootstrap_draws)
    np.save(image_ids_path, bootstrap_image_ids)

    with original_ids_path.open("w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(f"{int(image_id)}\n")

    write_summary_csv(
        output_path=summary_path,
        bootstrap_image_ids=bootstrap_image_ids,
    )

    first_replicate_unique = len(np.unique(bootstrap_image_ids[0]))

    manifest = {
        "method": (
            "Paired non-parametric image-level bootstrap resampling "
            "with replacement"
        ),
        "resampling_unit": (
            "One complete held-out COCO validation image, including all "
            "ground-truth annotations and all saved detections for each model"
        ),
        "paired_design": (
            "The same bootstrap image-ID draws must be applied to every "
            "model compared within a pairwise contrast."
        ),
        "ground_truth": {
            "path": str(args.ground_truth.resolve()),
            "sha256": sha256_file(args.ground_truth),
            "image_count": n_images,
            "annotation_count": len(ground_truth["annotations"]),
            "category_count": len(ground_truth["categories"]),
        },
        "bootstrap": {
            "iterations_B": args.iterations,
            "sample_size_per_replicate_N": n_images,
            "sampling_with_replacement": True,
            "random_generator": "numpy.random.default_rng",
            "random_seed": args.seed,
            "draws_file": draws_path.name,
            "sampled_original_image_ids_file": image_ids_path.name,
            "original_sorted_image_ids_file": original_ids_path.name,
            "summary_file": summary_path.name,
            "first_replicate_unique_images": first_replicate_unique,
            "first_replicate_duplicate_draws": (
                n_images - first_replicate_unique
            ),
        },
        "prediction_files_validated": prediction_metadata,
        "implementation_note": (
            "During COCO metric calculation, repeated image draws must be "
            "represented with distinct synthetic image IDs. A repeated "
            "original image must carry duplicated ground-truth annotations "
            "and duplicated detections under the same synthetic image ID "
            "for every paired model."
        ),
        "do_not_treat_as_independent_units": [
            "COCO categories",
            "per-class AP values",
            "individual bounding boxes",
            "class-image quota assignments",
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if args.write_first_replicate:
        first_replicate_path = (
            args.output_dir / "first_replicate_image_ids.txt"
        )

        with first_replicate_path.open("w", encoding="utf-8") as f:
            for draw_position, image_id in enumerate(
                bootstrap_image_ids[0],
                start=1,
            ):
                f.write(f"{draw_position}\t{int(image_id)}\n")

    print("\nBootstrap manifest generation completed.")
    print(f"Bootstrap draw indices: {draws_path}")
    print(f"Bootstrap image IDs: {image_ids_path}")
    print(f"Original image IDs: {original_ids_path}")
    print(f"Bootstrap diagnostics: {summary_path}")
    print(f"Reproducibility manifest: {manifest_path}")

    if args.write_first_replicate:
        print(
            "First replicate inspection file: "
            f"{args.output_dir / 'first_replicate_image_ids.txt'}"
        )

    print("\nExpected bootstrap behavior:")
    print(f"  - Each replicate contains {n_images} image draws.")
    print(
        "  - Each replicate contains roughly 63.2% distinct original "
        "images on average."
    )
    print(
        "  - Remaining draws are duplicate occurrences of sampled images."
    )
    print(
        "  - A different subset of original images is omitted in each "
        "replicate."
    )


if __name__ == "__main__":
    main()