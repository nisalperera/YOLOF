# import fiftyone as fo

# # Path to your data
# data_path = "/home/nisalperera/YOLOF/datasets/coco_oi/images"
# labels_path = "/home/nisalperera/YOLOF/datasets/coco_oi/annotations/coco_oi_instances_eval.json"

# # Import the dataset
# dataset = fo.Dataset.from_dir(
#     dataset_dir=data_path,
#     dataset_type=fo.types.COCODetectionDataset,
#     labels_path=labels_path,
# )

# # Launch the App to visualize
# session = fo.launch_app(dataset)


import fiftyone as fo
import fiftyone.types as fot
import json
import os

# ─────────────────────────────────────────────
# Configuration — adjust to your folder path
# ─────────────────────────────────────────────
DATASET_ROOT   = "/home/nisalperera/YOLOF/datasets/coco_oi"
IMAGES_DIR     = os.path.join(DATASET_ROOT, "images")
ANNOTATIONS    = os.path.join(DATASET_ROOT, "annotations", "coco_oi_instances_eval.json")  # note: typo in your folder name
DATASET_NAME   = "coco_oi_dataset"

# ─────────────────────────────────────────────
# Load annotations
# ─────────────────────────────────────────────
with open(ANNOTATIONS, "r") as f:
    coco_data = json.load(f)

# Build lookup maps
categories   = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
images_info  = {img["id"]: img for img in coco_data.get("images", [])}

# Group annotations by image_id
annotations_map: dict = {}
for ann in coco_data.get("annotations", []):
    img_id = ann["image_id"]
    annotations_map.setdefault(img_id, []).append(ann)

# ─────────────────────────────────────────────
# Create FiftyOne dataset
# ─────────────────────────────────────────────
# Delete existing dataset with same name if it exists
if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)

dataset = fo.Dataset(DATASET_NAME)
samples = []

for img_id, img_meta in images_info.items():
    file_name  = img_meta["file_name"]          # e.g. "1.jpg", "2.jpg"
    img_width  = img_meta.get("width", None)
    img_height = img_meta.get("height", None)
    img_path   = os.path.join(IMAGES_DIR, file_name)

    # Build FiftyOne sample
    sample = fo.Sample(filepath=img_path)

    # Attach detections (bounding boxes)
    detections = []
    for ann in annotations_map.get(img_id, []):
        label     = categories.get(ann["category_id"], str(ann["category_id"]))
        x, y, w, h = ann["bbox"]                # COCO: [x_min, y_min, width, height] in pixels

        # FiftyOne expects [x, y, w, h] normalised to [0, 1]
        if img_width and img_height:
            bounding_box = [
                x / img_width,
                y / img_height,
                w / img_width,
                h / img_height,
            ]
        else:
            # Fallback: use pixel values (FiftyOne will still accept them,
            # but normalised coordinates are strongly preferred)
            bounding_box = [x, y, w, h]

        detection = fo.Detection(
            label=label,
            bounding_box=bounding_box,
            confidence=ann.get("score", None),   # optional — present if predictions file
        )

        # Optionally carry over segmentation mask (polygon) if present
        if "segmentation" in ann and ann["segmentation"]:
            detection["segmentation"] = ann["segmentation"]

        detections.append(detection)

    sample["ground_truth"] = fo.Detections(detections=detections)

    # Store raw COCO metadata as a tag for easy filtering
    sample.tags = [file_name]
    samples.append(sample)

dataset.add_samples(samples)
dataset.persistent = True       # survives Python session restarts

print(f"\n✅ Dataset '{DATASET_NAME}' created with {len(dataset)} samples.")

# ─────────────────────────────────────────────
# Launch the FiftyOne App in your browser
# ─────────────────────────────────────────────
session = fo.launch_app(dataset)
session.wait()                  # blocks until you close the browser tab