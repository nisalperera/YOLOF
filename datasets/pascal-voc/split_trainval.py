import json
import random
import argparse
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
# Set to an integer (e.g., 42) for reproducible splits, or None for random
SEED = 42

# Split ratio (0.9 = 90% train, 10% val)
TRAIN_RATIO = 0.9

# ============================================================================
# Main
# ============================================================================
def split_coco_json(coco_json_path, train_output, val_output):
    """Split COCO JSON into train (90%) and val (10%) sets."""
    
    # Set random seed if specified
    if SEED is not None:
        random.seed(SEED)
        print(f"Using reproducible seed: {SEED}")
    else:
        print("Using random seed (no reproducibility)")
    
    # Load the full COCO JSON
    print(f"Loading {coco_json_path}...")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]
    
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    
    # Shuffle images for random split
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)
    
    # Calculate split point
    split_idx = int(len(shuffled_images) * TRAIN_RATIO)
    train_images = shuffled_images[:split_idx]
    val_images = shuffled_images[split_idx:]
    
    # Create sets of image IDs for quick lookup
    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}
    
    # Filter annotations for each split (preserve original IDs)
    train_annotations = [ann for ann in annotations if ann["image_id"] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]
    
    # Create train COCO dataset
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    
    # Create val COCO dataset
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }
    
    # Write train JSON
    print(f"Writing {train_output}...")
    with open(train_output, "w") as f:
        json.dump(train_coco, f)
    
    # Write val JSON
    print(f"Writing {val_output}...")
    with open(val_output, "w") as f:
        json.dump(val_coco, f)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Split Statistics:")
    print("=" * 60)
    print(f"Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Val:   {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"Split ratio: {len(train_images) / len(images):.2%} train / {len(val_images) / len(images):.2%} val")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split COCO JSON dataset into 90% train and 10% validation sets."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parent / "trainval_2007+2012.json"),
        help="Path to the input COCO JSON file (default: trainval_2007+2012.json in script directory)",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default=str(Path(__file__).parent / "trainval_2007+2012_train.json"),
        help="Path to output train JSON file (default: trainval_2007+2012_train.json in script directory)",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default=str(Path(__file__).parent / "trainval_2007+2012_val.json"),
        help="Path to output validation JSON file (default: trainval_2007+2012_val.json in script directory)",
    )
    
    args = parser.parse_args()
    split_coco_json(args.input, args.train_output, args.val_output)
