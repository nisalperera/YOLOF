
import sys

import datetime
import logging
import time
import copy
from collections import abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from itertools import chain
from torch import nn
from torch.nn.functional import one_hot

import detectron2.utils.comm as comm
from detectron2.structures import BoxMode
# from detectron2.utils.events import EventStorage
# from detectron2.utils.comm import get_world_size, synchronize
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators

from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
from torchvision.ops import complete_box_iou_loss as ciou_loss, distance_box_iou_loss as diou_loss


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    comm.synchronize()
    logger.info("Evaluation results for {} on rank {}".format(evaluator.__class__.__name__, comm.get_rank()))
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


class EvalLossProcessor(DatasetEvaluator):
    """
    A DatasetEvaluator that computes the average loss during evaluation.
    It is used for debugging purposes.
    """

    def __init__(self, bbox_loss, focal_loss_alpha, focal_loss_gamma, enabled=False, distributed=True):
        self._zero_loss = torch.tensor(0.0, device=torch.device("cpu")).round(decimals=3)
        self._cpu_device = torch.device("cpu")
        # self._val_loss_cls = self._zero_loss
        # self._val_loss_box_reg = self._zero_loss
        self._val_loss_cls_list = []
        self._val_loss_box_reg_list = []
        self._total_samples = 0
        self._enabled = enabled

        self._focal_loss_alpha = focal_loss_alpha
        self._focal_loss_gamma = focal_loss_gamma
        self.box_reg_loss = getattr(sys.modules[__name__], bbox_loss + "_loss")  # e.g., "giou_loss", "smooth_l1_loss"

        self._distributed = distributed

        self._logger = logging.getLogger("detectron2.evaluation.evaluator")

    def reset(self):
        attribs = vars(self)
        for key in attribs:
            if "val_loss" in key and "list" in key:
                setattr(self, key, [])

    def process(self, inputs, outputs):
        """
        Process a batch of inputs and outputs to compute the loss.
        """

        if not self._enabled:
            return

        self._total_samples += len(outputs)

        for input, output in zip(inputs, outputs):
            gt_boxes = []
            gt_classes = []
            for annot in input.get("annotations", []):
                if "bbox" in annot:
                    gt_boxes.append(torch.tensor(BoxMode.convert(annot["bbox"], annot["bbox_mode"], BoxMode.XYXY_ABS), device=self._cpu_device))
                
                if "category_id" in annot:
                    gt_classes.append(one_hot(torch.tensor(annot["category_id"]), 16))
                
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
            
                if len(gt_classes) > len(instances):
                    # If there are more ground truth classes than predictions, pad the predictions
                    pred_classes = [torch.zeros(16, device=self._cpu_device)] * (len(gt_classes) - len(instances))
                else:
                    pred_classes = []
                    gt_classes.extend([torch.zeros(16, device=self._cpu_device)] * (len(instances) - len(gt_classes)))
                
                pred_classes.extend([one_hot(torch.tensor(cat), 16) for cat in instances.pred_classes])
                pred_classes = torch.stack(pred_classes, dim=0) if pred_classes else torch.zeros((len(instances), 16), device=self._cpu_device)
                gt_classes = torch.stack(gt_classes, dim=0) if gt_classes else torch.zeros((len(instances), 16), device=self._cpu_device)
                class_loss = sigmoid_focal_loss_jit(
                    pred_classes,
                    gt_classes,
                    alpha=self._focal_loss_alpha,
                    gamma=self._focal_loss_gamma,
                    reduction="sum",
                )
                pred_boxes = []
                if instances.pred_boxes.tensor.size(0) == 0:
                    pred_boxes.append(torch.tensor([0, 0, 10, 10], device=self._cpu_device))

                if len(gt_boxes) > len(pred_boxes):
                    # If there are more ground truth boxes than predictions, pad the predictions
                    pred_boxes.extend([torch.tensor([0, 0, 10, 10], device=self._cpu_device)] * (len(gt_boxes) - len(pred_boxes)))
                else:
                    gt_boxes.extend([torch.tensor([0, 0, 10, 10], device=self._cpu_device)] * (len(pred_boxes) - len(gt_boxes)))

                pred_boxes = torch.stack(pred_boxes, dim=0) if pred_boxes else torch.zeros((len(instances), 4), device=self._cpu_device)
                gt_boxes = torch.stack(gt_boxes, dim=0) if gt_boxes else torch.zeros((len(instances), 4), device=self._cpu_device)
                box_loss = self.box_reg_loss(
                    pred_boxes,
                    gt_boxes,
                    reduction="sum",
                )

                # self._val_loss_cls += class_loss.round(decimals=3)
                # self._val_loss_box_reg += box_loss.round(decimals=3)
                self._val_loss_cls_list.append(class_loss.round(decimals=3))
                self._val_loss_box_reg_list.append(box_loss.round(decimals=3))
                # prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

            # if len(prediction) > 1:
            #     self._predictions.append(prediction)

    def evaluate(self):
        """
        Return the average loss.
        """

        if self._total_samples == 0 or not self._enabled:
            return {
                "val_loss_total": float("nan"), 
                "val_loss_cls": float("nan"), 
                "val_loss_box_reg": float("nan")
            }

        self._val_loss_cls = torch.tensor(self._val_loss_cls_list).mean() if len(self._val_loss_cls_list) else self._zero_loss
        self._val_loss_box_reg = torch.tensor(self._val_loss_box_reg_list).mean() if len(self._val_loss_box_reg_list) else self._zero_loss
        
        if self._distributed:
            # Gather the results from all processes
            comm.synchronize()
            print(
                "COCOEvaluator Gathering losses from {} rank ... \n".format(
                    comm.get_rank()
                )
            )
            val_loss_cls = comm.gather(self._val_loss_cls, dst=0)
            val_loss_box_reg = comm.gather(self._val_loss_box_reg, dst=0)

            if not comm.is_main_process():
                return {}
            
            # Average the losses across all processes
            self._val_loss_cls = torch.tensor(val_loss_cls).mean()
            self._val_loss_box_reg = torch.tensor(val_loss_box_reg).mean()
            self._total_loss  = self._val_loss_cls + self._val_loss_box_reg

        else:
            # If not distributed, just use the local values
            self._total_loss = self._val_loss_cls + self._val_loss_box_reg
        
        return {
            "val_loss_cls": round(copy.deepcopy(self._val_loss_cls).item(), 3),
            "val_loss_box_reg": round(copy.deepcopy(self._val_loss_box_reg).item(), 3),
            "val_loss_total": round(copy.deepcopy(self._total_loss).item(), 3)
        }


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
