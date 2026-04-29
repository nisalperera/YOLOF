#!/usr/bin/env python3
"""Smoke test for all 6 augmentation groups."""
import numpy as np
import tempfile
import os
import json
import shutil
import cv2

# Create a synthetic COCO-like dataset with 20 dummy images
tmpdir = tempfile.mkdtemp()
img_dir = os.path.join(tmpdir, "images")
os.makedirs(img_dir)

annotations = []
images_list = []
ann_id = 1
for i in range(20):
    img_path = os.path.join(img_dir, f"img_{i:03d}.jpg")
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    images_list.append({
        "id": i, "file_name": img_path, "width": 100, "height": 100
    })
    for _ in range(2):
        x, y = np.random.randint(0, 60, 2)
        w, h = np.random.randint(10, 40, 2)
        annotations.append({
            "id": ann_id, "image_id": i, "category_id": 1,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(w * h), "iscrowd": 0
        })
        ann_id += 1

coco = {
    "images": images_list,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "obj"}]
}
ann_file = os.path.join(tmpdir, "ann.json")
with open(ann_file, "w") as f:
    json.dump(coco, f)

# Register dataset
from detectron2.data.datasets.coco import register_coco_instances
register_coco_instances("smoke_test", {}, ann_file, img_dir)

from yolof.config import get_cfg
from yolof.data import YOLOFDatasetMapper
from detectron2.data import DatasetCatalog

dicts = DatasetCatalog.get("smoke_test")

# ---- Test non-mosaic groups ----
for group in ["minimal", "mixup", "cutmix", "autoaugment"]:
    cfg = get_cfg()
    cfg.INPUT.AUG_GROUP = group
    cfg.INPUT.COLOR_JITTER.ENABLED = True
    cfg.INPUT.MIXUP.ENABLED = (group == "mixup")
    cfg.INPUT.CUTMIX.ENABLED = (group == "cutmix")
    cfg.INPUT.AUTOAUGMENT.ENABLED = (group == "autoaugment")
    cfg.MODEL.YOLOF.RETURN_VAL_LOSS = True
    mapper = YOLOFDatasetMapper(cfg, True)

    results = []
    for d in dicts:
        out = mapper(d)
        results.append(out)

    last = results[-1]
    assert "image" in last, f"{group}: no image tensor"
    assert last["image"].dim() == 3, f"{group}: wrong image dims"
    assert "instances" in last, f"{group}: no instances"
    print(
        f"  {group}: image={tuple(last['image'].shape)}, "
        f"num_instances={len(last['instances'])}"
    )

print("Non-mosaic groups passed!")

# ---- Test mosaic groups ----
for group in ["mosaic", "mosaic_color"]:
    cfg = get_cfg()
    cfg.INPUT.AUG_GROUP = group
    cfg.INPUT.RESIZE.ENABLED = True
    cfg.INPUT.JITTER_CROP.ENABLED = True
    cfg.INPUT.DISTORTION.ENABLED = (group == "mosaic_color")
    cfg.INPUT.MOSAIC.ENABLED = True
    cfg.MODEL.YOLOF.RETURN_VAL_LOSS = True
    mapper = YOLOFDatasetMapper(cfg, True)

    results = []
    for d in dicts:
        out = mapper(d)
        results.append(out)

    last = results[-1]
    assert "image" in last, f"{group}: no image tensor"
    assert last["image"].dim() == 3, f"{group}: wrong image dims"
    assert "instances" in last, f"{group}: no instances"
    print(
        f"  {group}: image={tuple(last['image'].shape)}, "
        f"num_instances={len(last['instances'])}"
    )

print("Mosaic groups passed!")

# Cleanup
shutil.rmtree(tmpdir)
print("\nAll 6 augmentation groups passed the smoke test!")
