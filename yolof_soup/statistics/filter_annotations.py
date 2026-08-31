import os
import json

from glob import glob

from yolof_soup.config.experiment_config import COCO_VAL_ANN


with open(COCO_VAL_ANN) as f:
    gt_data = json.load(f)


pred_jsons = glob("/home/nisalperera/YOLOF/results/inference/final_eval/*/coco_instances_results.json")

for pred_json in pred_jsons:
    print(f"Processing saved predictions file: {pred_json.replace(os.getenv('HOME'), '')}")
    with open(pred_json) as r:
        predictions = json.load(r)

    image_ids = [img["id"] for img in gt_data["images"]]
    filtered_predictions = []
    for pred in predictions:
        if pred["image_id"] in image_ids:
            filtered_predictions.append(pred)
        else:
            print(f"[Warning] image_id: {pred['image_id']} not in the dataset.")

    print(f"Filtered out {len(predictions) - len(filtered_predictions)} because those images were not in the dataset.")

    # with open(pred_json, "w") as w:
    #     json.dump(filtered_predictions, w)