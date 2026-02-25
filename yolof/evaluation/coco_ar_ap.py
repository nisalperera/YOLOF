import os
import sys
import copy
import itertools

from collections import OrderedDict, defaultdict

import numpy as np
import torch.distributed as dist
import detectron2.utils.comm as comm

import torch

from tabulate import tabulate

from detectron2.layers import cat
from detectron2.structures import Boxes
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table
from detectron2.data.detection_utils import annotations_to_instances
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from fvcore.nn import sigmoid_focal_loss_jit, giou_loss

from torchvision.ops.boxes import box_iou
from torchvision.ops import complete_box_iou_loss as ciou_loss, distance_box_iou_loss as diou_loss


class COCOEvaluatorWithAPandAR(COCOEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            if "losses" in output:
                prediction["losses"] = defaultdict(dict)
                # Move losses to CPU for logging
                for key, value in output["losses"].items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(self._cpu_device)
                    prediction["losses"][key] = value

            if len(prediction) > 1:
                self._predictions.append(prediction)
        

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("COCOEvaluator Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        if "losses" in predictions[0]:
            # Aggregate losses across all predictions
            self._aggregate_losses(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _aggregate_losses(self, predictions):
        """
        Aggregate losses from predictions.
        Args:
            predictions: a list of predictions, each containing 'losses' key.
        Returns:
            A dictionary with aggregated losses.
        """
        aggregated_losses = defaultdict(list)
        for prediction in predictions:
            for key, value in prediction["losses"].items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                aggregated_losses[key].append(value)
        losses = {f"val_{key}": sum(values) / len(values) for key, values in aggregated_losses.items()}
        losses["val_loss_total"] = sum(losses.values())
        return self._results.update(losses)
    
    
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR-maxDets=1", "AR-maxDets=10", "AR-maxDets=100", "ARs-maxDets=100", "ARm-maxDets=100", "ARl-maxDets=100"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        recalls = coco_eval.eval["recall"]
        # recall has dims (iou, cls, area range, max dets)
        assert len(class_names) == recalls.shape[1]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")

            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")
            
            results_per_category.append(("{}".format(name), float(ap * 100), float(ar * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP & AR: \n".format(iou_type) + table)

        for name, ap, ar in results_per_category:
            results.update({"AP-"+name: ap, "AR-"+name: ar})

        return results
    