#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other
customizations.
"""

import logging
import os
import json
import math

from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.nn as nn

import detectron2.utils.comm as comm
from detectron2.data import (
    MetadataCatalog, build_detection_train_loader, build_detection_test_loader
)
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, hooks, launch
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import register_coco_instances

from detectron2.utils.events import EventStorage
from detectron2.modeling import GeneralizedRCNNWithTTA

from yolof.checkpoint import YOLOFCheckpointer
from yolof.config import get_cfg
from yolof.data import YOLOFDtasetMapper
from yolof.checkpoint import YOLOFCheckpointer
from yolof.hooks import GradCAMHook, BestCheckpointerAPARF1, EarlyStoppingHook
from yolof.evaluation.coco_ar_ap import COCOEvaluatorWithAPandAR
from yolof.data.samplers import EvenlyDistributedInferenceSampler

import wandb

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        if comm.is_main_process():
            wandb.init(entity="nisalperera", project="Thesis", config=cfg, sync_tensorboard=True)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset.
        For your own dataset, you can simply create an evaluator manually in
        your script and do not have to worry about the hacky if-else logic
        here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(
                COCOEvaluatorWithAPandAR(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(
                COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently " \
               "do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently " \
               "do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "YOLOF" == cfg.MODEL.META_ARCHITECTURE:
            mapper = YOLOFDtasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if "YOLOF" == cfg.MODEL.META_ARCHITECTURE:
            mapper = YOLOFDtasetMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        norm_module_types = (
            nn.BatchNorm2d,
            nn.SyncBatchNorm,
            nn.GroupNorm
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "backbone" in name:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                params += [
                    {"params": [value], "lr": lr, "weight_decay": weight_decay}
                ]

        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name,
                output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    def build_hooks(self):
        # Add hooks for saving best model based on bbox AP in evalutation
        cfg = self.cfg.clone()
        hooks = super().build_hooks()
        # hooks.insert(-1, EarlyStoppingHook(log_dir=cfg.OUTPUT_DIR, 
        #                                    es_metric=("bbox/AP", "bbox/AR-maxDets=100"), 
        #                                    eval_period=cfg.TEST.EVAL_PERIOD, mode="min"))
        hooks.insert(-1, BestCheckpointerAPARF1(cfg.TEST.EVAL_PERIOD,
                                          YOLOFCheckpointer(self.model, cfg.OUTPUT_DIR),
                                          ("bbox/AP",), "max"))
        return hooks

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Starting training from iteration {self.start_iter}. Evaluation will happen every {self.cfg.TEST.EVAL_PERIOD} iterations.")

        self.stop_training = False

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    if self.stop_training:
                        logger.info("Training is stopped due to EarlyStopping.")
                        break
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


def setup(args):
    """
    Create configs and perform basic setups.
    """

    logger = logging.getLogger("detectron2.setup")
    # root_dir = Path(__file__).resolve().parents[1]

    # Define paths for your datasets (assuming they were created in previous steps)
    TRAIN_ANN_FILE = f'/kaggle/input/2017-2017/annotations_trainval2017/annotations/instances_train2017.json'
    TRAIN_IMG_DIR = f'/kaggle/input/2017-2017/train2017/train2017'
    VAL_ANN_FILE = f'/kaggle/input/2017-2017/annotations_trainval2017/annotations/instances_val2017.json'
    VAL_IMG_DIR = f'/kaggle/input/2017-2017/val2017/val2017'

    with open(TRAIN_ANN_FILE, "r") as r:
        thing_classes = [cat['name'] for cat in json.load(r)["categories"]]

    register_coco_instances("vehicle_train", {}, TRAIN_ANN_FILE, TRAIN_IMG_DIR)
    register_coco_instances("vehicle_val", {}, VAL_ANN_FILE, VAL_IMG_DIR)

    MetadataCatalog.get("vehicle_train").set(thing_classes=thing_classes)
    MetadataCatalog.get("vehicle_val").set(thing_classes=thing_classes)

    logger.info("Datasets registered successfully!")
    logger.info("Available datasets: {}".format(DatasetCatalog.list()))

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # iters2epoch = math.floor(len(DatasetCatalog.get("vehicle_train")) / (cfg.SOLVER.IMS_PER_BATCH * args.num_gpus))
    # max_iter = cfg.SOLVER.MAX_ITER * iters2epoch
    # warmup_iters = cfg.SOLVER.WARMUP_ITERS * iters2epoch
    # steps = cfg.SOLVER.STEPS

    cfg.MODEL.YOLOF.DECODER.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.YOLOF.RETURN_VAL_LOSS = True

    cfg.DATASETS.TRAIN = ("vehicle_train",)
    cfg.DATASETS.TEST = ("vehicle_val",)
    # cfg.SOLVER.MAX_ITER = max_iter
    # cfg.SOLVER.WARMUP_ITERS = warmup_iters
    # cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.OUTPUT_DIR = "./output"

    # cfg.SOLVER.STEPS = tuple([int(max_iter * step) for step in steps])
    # cfg.SOLVER.IMS_PER_BATCH = args.num_gpus * cfg.SOLVER.IMS_PER_BATCH
    # cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * args.num_gpus
    # cfg.SOLVER.CHECKPOINT_PERIOD = int(iters2epoch * cfg.SOLVER.CHECKPOINT_PERIOD) 
    # cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        YOLOFCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0,
                            lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    # Change the layer of your choice
    trainer.register_hooks([
        # GradCAMHook("decoder.cls_subnet", cfg=cfg),
        hooks.PeriodicCheckpointer(YOLOFCheckpointer(trainer.model, cfg.OUTPUT_DIR, save_to_disk=True), cfg.SOLVER.CHECKPOINT_PERIOD, trainer.max_iter)
    ])

    return trainer.train()


if __name__ == "__main__":
    
    os.environ["OMP_NUM_THREADS"] = "1"
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )