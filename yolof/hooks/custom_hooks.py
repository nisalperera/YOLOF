import math
import random
import operator

from copy import deepcopy
from collections import defaultdict

import cv2
import torch

import numpy as np

import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog
from detectron2.engine.hooks import HookBase
from detectron2.checkpoint import Checkpointer
from detectron2.utils.logger import setup_logger
from detectron2.engine.hooks import BestCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.comm import is_main_process, synchronize

from yolof.cam.gradcam import GradCAM


class GradCAMHook(HookBase):
    """
    Custom Detectron2 Hook to compute Grad-CAM after each training iteration
    and log the visualizations to TensorBoard.
    """
    def __init__(self, target_layer_name, cfg):
        """
        Args:
            model: Detectron2 GeneralizedRCNN model.
            target_layer_name: Name of the convolutional layer for Grad-CAM.
            log_dir: Directory to save TensorBoard logs.
        """

        self.cfg = cfg
        self.target_layer_name = target_layer_name
        self.eval_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.logger = setup_logger(output=cfg.OUTPUT_DIR, name="d2.yolof.grad_cam")

    def _gen_input(self, img_path):
        image = read_image(img_path, format="BGR").copy()
        image_height, image_width = image.shape[:2]
        transform_gen = T.ResizeShortestEdge(
        	[self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(image).apply_image(image)
        input_tensor = torch.as_tensor(transformed_img.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

        return image, image_height, image_width, input_tensor

    def _generate_cam(self):
        """
        Called after each training step to compute and log Grad-CAM visualizations.
        """
        
        for image_obj in self.eval_dataloader:
            idx = image_obj["image_id"]
            self.logger.info(f"Iteration: {self.trainer.iter}, Generating CAM for Image: id-{idx}")

            filename = image_obj["file_name"]
            
            # Prepare input for Grad-CAM (assuming a single image batch)
            input_image_np, image_height, image_width, input_tensor = self._gen_input(filename)
            input_dict = {
                "image": input_tensor,
                "height": image_height,
                "width": image_width,
                "annotations": image_obj.get("annotations", []),
                "instances": None,  # No instances for Grad-CAM
            }

            output = self.grad_cam(input_dict)["instances"]
            
            heatmap = None
            for target_instance in range(output.pred_classes.detach().cpu().numpy().shape[0]):
                pred_box = output.pred_boxes[target_instance].tensor[0]
                pred_class = output.pred_classes[target_instance].detach().cpu().numpy().item()

                x1 = int(pred_box.detach().cpu().numpy()[0])
                y1 = int(pred_box.detach().cpu().numpy()[1])
                x2 = int(pred_box.detach().cpu().numpy()[2])
                y2 = int(pred_box.detach().cpu().numpy()[3])

                cv2.rectangle(input_image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(input_image_np, str(pred_class + 1), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Compute Grad-CAM heatmap and overlay
                with self.grad_cam as cam:
                    cam_heatmap = cam.get_cam(output, target_instance)
                
                if cam_heatmap.size != 0:

                    # Convert heatmap to RGB format for visualization
                    if heatmap is not None:
                        heatmap = cv2.addWeighted(cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET), 1, heatmap, 1, 0)
                    else:
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)

                if len(image_obj["annotations"]) > 0 and len(image_obj["annotations"]) > target_instance:
                    annotation = image_obj["annotations"][target_instance]

                    x1 = int(annotation["bbox"][0])
                    y1 = int(annotation["bbox"][1])
                    x2 = int(annotation["bbox"][2] + x1)
                    y2 = int(annotation["bbox"][3] + y1)

                    cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(heatmap, str(annotation["category_id"] + 1), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
            if heatmap is not None:
                input_image_np = cv2.resize(input_image_np, (heatmap.shape[1], heatmap.shape[0]))
                heatmap = cv2.cvtColor(cv2.addWeighted(input_image_np, 0.5, heatmap, 0.5, 0), cv2.COLOR_RGB2BGR)
                self.trainer.storage.put_image(f"Heatmap - Img ID: {idx}", torch.from_numpy(heatmap.transpose(2, 0, 1)))
            else:
                self.logger.warn(f"Class Activation Maps are not found. Outputs: {len(output)}. Image: id-{idx}")

    def _initiate_grad_cam(self):
        """
        Initializes the Grad-CAM instance with the model and target layer.
        """
        if isinstance(self.trainer.model, torch.nn.parallel.DistributedDataParallel):
            self.logger.info("Converting DistributedDataParallel model to Non-Distributed model for Grad-CAM.")
            self.model = deepcopy(self.trainer.model.module)
            self.model.eval()
        else:
            self.logger.info("Using Non-Distributed model for Grad-CAM.")
            self.model = self.trainer.model

        self.grad_cam = GradCAM(self.model, self.target_layer_name, logger=self.logger)
        self.logger.info(f"Grad-CAM initialized for layer: {self.target_layer_name}")
        
        if self.model.training:
            self.model.eval()

        if self.model.return_val_loss:
            self.model.return_val_loss = False

    def after_step(self):
        synchronize()
        next_iter = self.trainer.iter + 1
        
        if not self.eval_period == 0 and next_iter % self.eval_period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter and is_main_process():
                self.logger.info(f"{self.trainer.iter}: Starting to generate CAM on {len(self.eval_dataloader)} images...")

                # Initialize Grad-CAM instance
                self._initiate_grad_cam()

                self._generate_cam()
        
        
    def before_step(self):
        """
        Close the TensorBoard FileWriter after training.
        """
        synchronize()

        if hasattr(self, "grad_cam"):
            delattr(self, "grad_cam")

        if hasattr(self, "model"):
            delattr(self, "model")
        
        if not self.trainer.model.training:
            self.trainer.model.train()

        self.trainer.model.return_val_loss = self.cfg.MODEL.YOLOF.RETURN_VAL_LOSS

    def before_train(self):
        synchronize()
        if is_main_process():            
            eval_dataset = DatasetCatalog.get(self.trainer.cfg.DATASETS.TEST[0])

            annotations = [(i, obj["annotations"]) for i, obj in enumerate(eval_dataset)]

            selected_classes = defaultdict(list)
            for i, annots in annotations:
                image_obj = eval_dataset[i]
                for annot in annots:
                    selected_classes[annot["category_id"]].append({image_obj["image_id"]: image_obj})

            new_eval_dataset = {}
            for cls_id, image_objs in selected_classes.items():
                if len(image_objs) > 1:
                    new_eval_dataset[cls_id] = image_objs[random.randint(0, len(image_objs) - 1)]
                else:
                    new_eval_dataset[cls_id] = image_objs[0]

            self.eval_dataloader = [list(obj.values())[0] for obj in new_eval_dataset.values()]
            self.logger.info(f"Selected {len(self.eval_dataloader)} images for CAM visualization (one image per class).")

    def after_train(self):
        synchronize()
        # do the last eval in after_train
        if is_main_process():
            self.logger.info(f"Trainig Completed: Starting to generate CAM on {len(self.eval_dataloader)} images...")

            # Initialize Grad-CAM instance
            self._initiate_grad_cam()

            self._generate_cam()


class BestCheckpointerAPARF1(BestCheckpointer):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: tuple,
        mode: str = "max",
        file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (tuple): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """

        super().__init__(eval_period, checkpointer, None, mode, file_prefix)

        if not isinstance(val_metric, tuple):
            val_metric = (val_metric)
        elif isinstance(val_metric, list):
            val_metric = tuple(val_metric)

        self._val_metric = val_metric

    def _update_best(self, vals, iteration):
        if all([math.isnan(val) for val in vals]) or all([math.isinf(val) for val in vals]):
            return False
        self.best_metric = vals
        self.best_iter = iteration
        return True

    def _best_checking(self):
        latest_metric = []
        val_metrics = list(self._val_metric)
        metric_iter = 0
        for val_metric in val_metrics:
            if "F1" in val_metric:
                continue

            metric_tuple = self.trainer.storage.latest().get(val_metric)
            if metric_tuple is None:
                self._logger.warning(
                    f"Given val metric {val_metric} does not seem to be computed/stored."
                    "Will not be checkpointing based on it."
                )
            else:
                latest_metric.append(metric_tuple[0])
                metric_iter = metric_tuple[1]

        # This condtions assumes that the first 2 elements are precision and recall (order does not matter)
        if len(latest_metric) >= 2:
            # Calculate F1 score from precision and Recall
            if np.isnan(latest_metric[0]) or np.isnan(latest_metric[1]):
                self._logger.warning(
                    f"Metrices {val_metrics[0]} and {val_metrics[1]} values  are NaN. Skipping Evaluation."
                )
                return
            f1 = (2 * (latest_metric[0] * latest_metric[1]) / sum(latest_metric)) if sum(latest_metric) > 0 else 0
            latest_metric.append(f1)
        
        elif len(latest_metric) == 0 or any([math.isnan(metric) for metric in latest_metric]) or \
                any([math.isinf(metric) for metric in latest_metric]):
            self._logger.warning(
                    f"Either of the given val metrices {val_metrics} do not seem to be computed/stored. "
                    "Checkpointing will not be done."
                )
            return

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                self._logger.info(
                    f"Saved first model at {[round(best_metric, 4) for best_metric in self.best_metric]} @ {self.best_iter} steps"
                )
        elif all([self._compare(metric, best_metric) for metric, best_metric in zip(latest_metric, self.best_metric)]):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)
            self._logger.info(
                f"Saved best model as latest eval score for {val_metrics} are "
                f"{[round(metric, 4) for metric in latest_metric]}, better than last best score "
                f"{[round(best_metric, 4) for best_metric in self.best_metric]} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)
        else:
            self._logger.info(
                f"Not saving as latest eval score for {val_metrics} are {[round(metric, 4) for metric in latest_metric]}, "
                f"not better than best score {[round(best_metric, 4) for best_metric in self.best_metric]} @ iteration {self.best_iter}."
            )


class EarlyStoppingHook(HookBase):
    def __init__(self, log_dir, es_metric, patience=5, eval_period=1, mode: str = "max",):
        self._patience = patience
        self._eval_period = eval_period
        self.no_improvement_count = 0

        self.logger = setup_logger(output=log_dir, name="d2.yolof.early_stopping")
        
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `EarlyStopping` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt

        if isinstance(es_metric, str):
            self.es_metric = (es_metric)
        elif isinstance(es_metric, (list, tuple)):
            self.es_metric = tuple(es_metric)
        else:
            self.logger.warning(
                f"Given es_metric {es_metric} is not supported. Only tuple, list or string is supported."
            )
        self.best_metrics = {metric: np.inf if mode == "min" else -np.inf for metric in self.es_metric}

    def _early_stopping_check(self):
        current_metrics = {metric: np.nan for metric in self.es_metric}
        es_metrics = list(self.es_metric)
        for es_metric in es_metrics:
            if "f1" in es_metric.lower():
                self.logger.warning(
                    f"Given metric {es_metric} does not supported at the moment. "
                    "Will not be based on it."
                )

            metric_tuple = self.trainer.storage.latest().get(es_metric)
            if metric_tuple is None:
                self.logger.warning(
                    f"Given metric {es_metric} does not seem to be computed/stored. "
                    "Will not be based on it."
                )
            else:
                current_metrics[es_metric] = metric_tuple[0]

        for k in self.es_metric:
            if np.isnan(current_metrics[k]) or np.isinf(current_metrics[k]):
                self.logger.warning(
                    f"Metric {k} has NaN or Inf value. Skipping Early Stopping Check."
                )
                return
            
        if all([self._compare(current_metrics[metric], self.best_metrics[metric]) for metric in self.es_metric]):
            self.best_metrics = current_metrics
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            self.logger.info(
                f"No improvement in metrics {self.es_metric} for {self.no_improvement_count} iterations. "
                f"Current metrics: {[round(current_metrics[metric], 4) for metric in self.es_metric]}, "
                f"Best metrics: {[round(self.best_metrics[metric], 4) for metric in self.es_metric]}."
            )
            
        if self.no_improvement_count >= self._patience:
            self.trainer.stop_training = True  # Custom flag to stop training
            self.logger.info(
                f"Early stopping triggered after {self.no_improvement_count} iterations without improvement at iter {self.trainer.iter}. "
            )
        
    def after_step(self):
        synchronize()
        # do the last eval in after_train
        if is_main_process():
            next_iter = self.trainer.iter + 1
            if not self._eval_period == 0 and next_iter % self._eval_period == 0:
                self._early_stopping_check()



class ValidationLoss(HookBase):
    """
    Custom Detectron2 Hook to compute validation loss after each training iteration
    and log the visualizations to TensorBoard.
    """
    def __init__(self, eval_period: int = 100):
        self.eval_period = eval_period
        self.logger = setup_logger(output=None, name="d2.yolof.validation_loss")

    def after_step(self):
        synchronize()
        next_iter = self.trainer.iter + 1
        
        if not self.eval_period == 0 and next_iter % self.eval_period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter and is_main_process():
                self.logger.info(f"{self.trainer.iter}: Starting to compute validation loss...")

                # Compute validation loss
                val_losses = self.trainer.storage.latest().get("val_losses")
                if val_losses is not None:
                    for k, v in val_losses.items():
                        self.logger.info(f"Validation Loss - {k}: {v:.4f}")
