import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
COCO_VAL_JSON   = "/home/nisalperera/YOLOF/datasets/coco/annotations/instances_val2017.json"   # path to annotation file
COCO_VAL_IMAGES = "/home/nisalperera/YOLOF/datasets/coco/images/val2017/"                              # path to validation images folder
OUTPUT_JSON     = "/home/nisalperera/YOLOF/datasets/coco/annotations/subset_val2017.json"                  # output annotation file
OUTPUT_IMAGES   = "/home/nisalperera/YOLOF/datasets/coco/images/subset_val2017/"                      # output image folder (optional copy)
MIN_IMAGES_PER_CAT = 100                                 # minimum images per category
COPY_IMAGES     = True                                   # set False to skip copying image files
# ─────────────────────────────────────────────────────────────────────────────


def create_coco_subset(
    annotation_file: str,
    image_dir: str,
    output_json: str,
    output_image_dir: str,
    min_per_category: int = 100,
    copy_images: bool = True,
):
    print("Loading annotations...")
    with open(annotation_file, "r") as f:
        coco = json.load(f)

    # ── Build lookup maps ────────────────────────────────────────────────────
    # image_id → list of annotation dicts
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    # category_id → set of image_ids that contain it
    cat_to_imgs = defaultdict(set)
    for ann in coco["annotations"]:
        cat_to_imgs[ann["category_id"]].add(ann["image_id"])

    # image_id → image dict
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # ── Greedy selection: for each category, add images until quota met ──────
    selected_image_ids: set[int] = set()

    # Sort categories by how many covering images they already have in selected
    # (process rarest categories first — maximises sharing of images across cats)
    cat_ids_sorted = sorted(cat_to_imgs.keys(), key=lambda c: len(cat_to_imgs[c]))

    for cat_id in cat_ids_sorted:
        available = cat_to_imgs[cat_id]
        # Count how many of this category's images are already selected
        already = len(available & selected_image_ids)
        needed  = max(0, min_per_category - already)

        if needed == 0:
            continue

        # Prefer images that also cover other under-represented categories
        # (simple heuristic: sort by number of distinct categories in the image)
        candidates = sorted(
            available - selected_image_ids,
            key=lambda iid: len({a["category_id"] for a in img_to_anns[iid]}),
            reverse=True,   # images with more categories first
        )

        for iid in candidates[:needed]:
            selected_image_ids.add(iid)

    print(f"Selected {len(selected_image_ids)} images covering all categories "
          f"with ≥{min_per_category} images each.")

    # ── Verify coverage ──────────────────────────────────────────────────────
    cat_counts: dict[int, int] = defaultdict(int)
    for iid in selected_image_ids:
        for ann in img_to_anns[iid]:
            cat_counts[ann["category_id"]] += 1

    shortfall = {c: cat_counts.get(c, 0) for c in cat_to_imgs
                 if cat_counts.get(c, 0) < min_per_category}
    if shortfall:
        print(f"WARNING: {len(shortfall)} categories still below threshold "
              f"(not enough val images exist for them): {shortfall}")
    else:
        print("✓ All categories meet the minimum threshold.")

    # ── Build subset annotation dict ─────────────────────────────────────────
    selected_image_ids_sorted = sorted(selected_image_ids)
    subset_images = [img_id_to_info[iid] for iid in selected_image_ids_sorted]
    subset_anns   = [ann for iid in selected_image_ids_sorted
                     for ann in img_to_anns[iid]]

    subset_cat_ids = {ann["category_id"] for ann in subset_anns}
    subset_cats    = [c for c in coco["categories"] if c["id"] in subset_cat_ids]

    subset_coco = {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  subset_cats,
        "images":      subset_images,
        "annotations": subset_anns,
    }

    # ── Save annotation JSON ─────────────────────────────────────────────────
    with open(output_json, "w") as f:
        json.dump(subset_coco, f)
    print(f"Saved subset annotation → {output_json}")

    # ── Optionally copy images ───────────────────────────────────────────────
    if copy_images:
        os.makedirs(output_image_dir, exist_ok=True)
        missing = 0
        for img_info in subset_images:
            src  = Path(image_dir) / img_info["file_name"]
            dest = Path(output_image_dir) / img_info["file_name"]
            if src.exists():
                shutil.copy2(src, dest)
            else:
                missing += 1
        if missing:
            print(f"WARNING: {missing} image files not found in '{image_dir}'.")
        print(f"Copied images → {output_image_dir}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n── Category coverage summary ──────────────────────────────")
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    print(f"{'Category':<25} {'Images':>8}")
    print("─" * 36)
    for cat in sorted(subset_cats, key=lambda c: c["name"]):
        print(f"{cat['name']:<25} {cat_counts.get(cat['id'], 0):>8}")
    print("─" * 36)
    print(f"{'TOTAL images':<25} {len(subset_images):>8}")
    print(f"{'TOTAL annotations':<25} {len(subset_anns):>8}")


if __name__ == "__main__":
    create_coco_subset(
        annotation_file   = COCO_VAL_JSON,
        image_dir         = COCO_VAL_IMAGES,
        output_json       = OUTPUT_JSON,
        output_image_dir  = OUTPUT_IMAGES,
        min_per_category  = MIN_IMAGES_PER_CAT,
        copy_images       = COPY_IMAGES,
    )