from collections import deque
import copy
import logging
from typing import Optional, List, Union

import cv2
import numpy as np
import torch

from detectron2.utils.logger import setup_logger
from detectron2.config import configurable, CfgNode
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import BoxMode

from .detection_utils import (
    build_augmentation,
    transform_instance_annotations,
    filter_small_boxes,
)


class YOLOFDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by YOLOF.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Add a queue for saving previous image infos in mosaic transformation
    2. Applies cropping/geometric transforms to the image and annotations
    3. Optionally applies Mixup / CutMix using a pool of prior images
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        mosaic_trans: Optional[CfgNode],
        mixup_cfg: Optional[CfgNode] = None,
        cutmix_cfg: Optional[CfgNode] = None,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
        add_meta_infos: bool = False,
        return_val_loss: bool = False,
    ):
        """
        Args:
            augmentations: a list of augmentations or deterministic
                transforms to apply
            image_format: an image format supported by
                :func:`detection_utils.read_image`.
            mosaic_trans: a CfgNode for Mosaic transformation.
            mixup_cfg: a CfgNode for Mixup augmentation.
            cutmix_cfg: a CfgNode for CutMix augmentation.
            use_instance_mask: whether to process instance segmentation
                annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process
                instance segmentation masks into this format.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask
                annotations.
            add_meta_infos: whether to add `meta_infos` field
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.recompute_boxes = recompute_boxes
        self.add_meta_infos = add_meta_infos
        self.return_val_loss = return_val_loss
        # fmt: on
        logger = logging.getLogger("detectron2.data")
        mode = "training" if is_train else "inference"
        if not is_train:
            logger.info(
                f"[YoloFDatasetMapper] Constructing a YOLOF DatasetMapper for "
                f"with return_val_loss={self.return_val_loss} in {mode} mode."
            )
        logger.info(
            f"[YoloFDatasetMapper] Augmentations used in {mode}: {augmentations}"
        )

        # --- Mosaic pool ---
        self.mosaic_trans = mosaic_trans
        if self.mosaic_trans is not None and self.mosaic_trans.ENABLED:
            self.mosaic_pool = deque(maxlen=self.mosaic_trans.POOL_CAPACITY)

        # --- Mixup pool ---
        self.mixup_cfg = mixup_cfg
        if self.mixup_cfg is not None and self.mixup_cfg.ENABLED:
            self.mixup_pool = deque(maxlen=self.mixup_cfg.POOL_CAPACITY)
            logger.info(
                f"[YoloFDatasetMapper] Mixup enabled: alpha={self.mixup_cfg.ALPHA}, "
                f"prob={self.mixup_cfg.PROB}"
            )

        # --- CutMix pool ---
        self.cutmix_cfg = cutmix_cfg
        if self.cutmix_cfg is not None and self.cutmix_cfg.ENABLED:
            self.cutmix_pool = deque(maxlen=self.cutmix_cfg.POOL_CAPACITY)
            logger.info(
                f"[YoloFDatasetMapper] CutMix enabled: alpha={self.cutmix_cfg.ALPHA}, "
                f"prob={self.cutmix_cfg.PROB}"
            )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # use a local `build_augmentation` instead
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        elif cfg.INPUT.JITTER_CROP.ENABLED and is_train:
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        # Determine which multi-image augmentations are active based on
        # AUG_GROUP.  Only one of mosaic / mixup / cutmix can be active.
        aug_group = cfg.INPUT.AUG_GROUP
        mosaic_cfg = copy.deepcopy(cfg.INPUT.MOSAIC)
        mixup_cfg = copy.deepcopy(cfg.INPUT.MIXUP)
        cutmix_cfg = copy.deepcopy(cfg.INPUT.CUTMIX)

        if aug_group == "mosaic" or aug_group == "mosaic_color":
            mosaic_cfg.defrost()
            mosaic_cfg.ENABLED = True
            mosaic_cfg.freeze()
            mixup_cfg.defrost()
            mixup_cfg.ENABLED = False
            mixup_cfg.freeze()
            cutmix_cfg.defrost()
            cutmix_cfg.ENABLED = False
            cutmix_cfg.freeze()
        elif aug_group == "mixup":
            mosaic_cfg.defrost()
            mosaic_cfg.ENABLED = False
            mosaic_cfg.freeze()
            mixup_cfg.defrost()
            mixup_cfg.ENABLED = True
            mixup_cfg.freeze()
            cutmix_cfg.defrost()
            cutmix_cfg.ENABLED = False
            cutmix_cfg.freeze()
        elif aug_group == "cutmix":
            mosaic_cfg.defrost()
            mosaic_cfg.ENABLED = False
            mosaic_cfg.freeze()
            mixup_cfg.defrost()
            mixup_cfg.ENABLED = False
            mixup_cfg.freeze()
            cutmix_cfg.defrost()
            cutmix_cfg.ENABLED = True
            cutmix_cfg.freeze()
        elif aug_group in ("minimal", "autoaugment", "none"):
            mosaic_cfg.defrost()
            mosaic_cfg.ENABLED = False
            mosaic_cfg.freeze()
            mixup_cfg.defrost()
            mixup_cfg.ENABLED = False
            mixup_cfg.freeze()
            cutmix_cfg.defrost()
            cutmix_cfg.ENABLED = False
            cutmix_cfg.freeze()
        # else: respect whatever the individual flags say

        # Mosaic groups need meta_infos from JitterCrop
        need_meta = cfg.INPUT.JITTER_CROP.ENABLED or aug_group in (
            "mosaic", "mosaic_color"
        )

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "mosaic_trans": mosaic_cfg,
            "mixup_cfg": mixup_cfg,
            "cutmix_cfg": cutmix_cfg,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "add_meta_infos": need_meta,
            "return_val_loss": cfg.MODEL.YOLOF.RETURN_VAL_LOSS,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset
                format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # ----------------------------------------------------------------
        # Pool management: add current sample to the relevant pool *before*
        # deciding whether to trigger the multi-image augmentation.
        # ----------------------------------------------------------------
        mosaic_flag = 0
        mosaic_samples = None
        if (
            self.mosaic_trans is not None
            and self.mosaic_trans.ENABLED
            and self.is_train
        ):
            if len(self.mosaic_pool) > self.mosaic_trans.NUM_IMAGES:
                mosaic_flag = np.random.randint(2)
                if mosaic_flag == 1:
                    mosaic_samples = np.random.choice(
                        self.mosaic_pool, self.mosaic_trans.NUM_IMAGES - 1
                    )
            self.mosaic_pool.append(copy.deepcopy(dataset_dict))

        mixup_flag = False
        mixup_sample = None
        if (
            self.mixup_cfg is not None
            and self.mixup_cfg.ENABLED
            and self.is_train
        ):
            self.mixup_pool.append(copy.deepcopy(dataset_dict))
            if (
                len(self.mixup_pool) > 2
                and np.random.random() < self.mixup_cfg.PROB
            ):
                mixup_flag = True
                mixup_sample = np.random.choice(self.mixup_pool)

        cutmix_flag = False
        cutmix_sample = None
        if (
            self.cutmix_cfg is not None
            and self.cutmix_cfg.ENABLED
            and self.is_train
        ):
            self.cutmix_pool.append(copy.deepcopy(dataset_dict))
            if (
                len(self.cutmix_pool) > 2
                and np.random.random() < self.cutmix_cfg.PROB
            ):
                cutmix_flag = True
                cutmix_sample = np.random.choice(self.cutmix_pool)

        # ----------------------------------------------------------------
        # Load + augment the primary image
        # ----------------------------------------------------------------
        image, annos = self._load_image_with_annos(dataset_dict)

        # ----------------------------------------------------------------
        # Mosaic (existing logic, unchanged)
        # ----------------------------------------------------------------
        if self.is_train and mosaic_flag == 1 and mosaic_samples is not None:
            image, annos = self._apply_mosaic(image, annos, dataset_dict, mosaic_samples)

        # ----------------------------------------------------------------
        # Mixup
        # ----------------------------------------------------------------
        if self.is_train and mixup_flag and mixup_sample is not None:
            image, annos = self._apply_mixup(image, annos, mixup_sample)

        # ----------------------------------------------------------------
        # CutMix
        # ----------------------------------------------------------------
        if self.is_train and cutmix_flag and cutmix_sample is not None:
            image, annos = self._apply_cutmix(image, annos, cutmix_sample)

        # ----------------------------------------------------------------
        # Convert to Instances
        # ----------------------------------------------------------------
        if annos is not None:
            image_shape = image.shape[:2]  # h, w
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        return dataset_dict

    # ====================================================================
    # Mixup
    # ====================================================================
    def _apply_mixup(self, image1, annos1, second_dict):
        """
        Blend two images via Mixup and take the union of their annotations.

        Args:
            image1 (ndarray): primary image (H, W, C), already augmented.
            annos1 (list[dict]): primary annotations.
            second_dict (dict): dataset_dict for the second image.

        Returns:
            (ndarray, list[dict]): blended image and merged annotations.
        """
        second_dict = copy.deepcopy(second_dict)
        image2, annos2 = self._load_image_with_annos(second_dict)

        if annos2 is None:
            return image1, annos1

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Resize second image to match the primary's size
        if (h1, w1) != (h2, w2):
            image2 = cv2.resize(
                image2.astype(np.float32), (w1, h1),
                interpolation=cv2.INTER_LINEAR,
            )
            # Rescale boxes of annos2
            sx, sy = w1 / max(w2, 1), h1 / max(h2, 1)
            for ann in annos2:
                bbox = ann["bbox"]
                ann["bbox"] = np.array(
                    [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy],
                    dtype=np.float32,
                )

        # Sample lambda
        alpha = self.mixup_cfg.ALPHA
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

        # Blend images
        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)
        blended = (lam * image1 + (1 - lam) * image2).clip(0, 255).astype(np.float32)

        # Union of annotations
        merged_annos = list(annos1) + list(annos2)

        # Remove tiny boxes
        min_area = self.mixup_cfg.MIN_BOX_AREA
        merged_annos = filter_small_boxes(merged_annos, min_area=min_area)

        return blended, merged_annos

    # ====================================================================
    # CutMix
    # ====================================================================
    def _apply_cutmix(self, image1, annos1, second_dict):
        """
        Paste a random rectangular patch from a second image into the primary
        image.  Annotations are adjusted accordingly.

        Args:
            image1 (ndarray): primary image (H, W, C), already augmented.
            annos1 (list[dict]): primary annotations.
            second_dict (dict): dataset_dict for the second image.

        Returns:
            (ndarray, list[dict]): composited image and adjusted annotations.
        """
        second_dict = copy.deepcopy(second_dict)
        image2, annos2 = self._load_image_with_annos(second_dict)

        if annos2 is None:
            return image1, annos1

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Resize second image to match primary
        if (h1, w1) != (h2, w2):
            image2 = cv2.resize(
                image2.astype(np.float32), (w1, h1),
                interpolation=cv2.INTER_LINEAR,
            )
            sx, sy = w1 / max(w2, 1), h1 / max(h2, 1)
            for ann in annos2:
                bbox = ann["bbox"]
                ann["bbox"] = np.array(
                    [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy],
                    dtype=np.float32,
                )

        # Sample lambda and compute cut region
        alpha = self.cutmix_cfg.ALPHA
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h1 * cut_ratio)
        cut_w = int(w1 * cut_ratio)

        # Random center
        cx = np.random.randint(0, w1)
        cy = np.random.randint(0, h1)

        # Clip to image bounds
        x1 = int(np.clip(cx - cut_w // 2, 0, w1))
        y1 = int(np.clip(cy - cut_h // 2, 0, h1))
        x2 = int(np.clip(cx + cut_w // 2, 0, w1))
        y2 = int(np.clip(cy + cut_h // 2, 0, h1))

        if x2 <= x1 or y2 <= y1:
            return image1, annos1

        # Paste patch from image2 into image1
        image1 = image1.astype(np.float32).copy()
        image1[y1:y2, x1:x2] = image2.astype(np.float32)[y1:y2, x1:x2]

        # Keep annos1 boxes that retain >=50% area outside the cut region
        min_area = self.cutmix_cfg.MIN_BOX_AREA
        kept_annos1 = []
        for ann in annos1:
            bbox = ann["bbox"]  # XYXY_ABS
            orig_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            if orig_area <= 0:
                continue
            # Area of overlap with cut region
            ix1 = max(bbox[0], x1)
            iy1 = max(bbox[1], y1)
            ix2 = min(bbox[2], x2)
            iy2 = min(bbox[3], y2)
            overlap_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            remaining_frac = 1.0 - overlap_area / orig_area
            if remaining_frac >= 0.5:
                kept_annos1.append(ann)

        # Add annos2 boxes clipped to the cut region
        kept_annos2 = []
        for ann in annos2:
            bbox = ann["bbox"]  # XYXY_ABS
            # Clip box to cut region
            clipped = np.array([
                max(bbox[0], x1), max(bbox[1], y1),
                min(bbox[2], x2), min(bbox[3], y2),
            ], dtype=np.float32)
            clipped_area = (
                max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
            )
            orig_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            # Keep if significant overlap with cut region
            if orig_area > 0 and clipped_area / orig_area >= 0.3:
                ann_copy = copy.deepcopy(ann)
                ann_copy["bbox"] = clipped
                kept_annos2.append(ann_copy)

        merged = kept_annos1 + kept_annos2
        merged = filter_small_boxes(merged, min_area=min_area)

        return image1, merged

    # ====================================================================
    # Mosaic (refactored from inline code)
    # ====================================================================
    def _apply_mosaic(self, image, annos, dataset_dict, mosaic_samples):
        """
        Apply Mosaic augmentation (4-image composition).
        """
        min_offset = self.mosaic_trans.MIN_OFFSET
        mosaic_width = self.mosaic_trans.MOSAIC_WIDTH
        mosaic_height = self.mosaic_trans.MOSAIC_HEIGHT
        cut_x = np.random.randint(
            int(mosaic_width * min_offset), int(mosaic_width * (1 - min_offset))
        )
        cut_y = np.random.randint(
            int(mosaic_height * min_offset), int(mosaic_height * (1 - min_offset))
        )
        out_image = np.zeros([mosaic_height, mosaic_width, 3], dtype=image.dtype)
        out_annos = []

        for m_idx in range(self.mosaic_trans.NUM_IMAGES):
            if m_idx != 0:
                dataset_dict = copy.deepcopy(mosaic_samples[m_idx - 1])
                image, annos = self._load_image_with_annos(dataset_dict)

            image_size = image.shape[:2]  # h, w
            meta_infos = annos[0].pop("meta_infos")
            pleft = meta_infos.get("jitter_pad_left", 0)
            pright = meta_infos.get("jitter_pad_right", 0)
            ptop = meta_infos.get("jitter_pad_top", 0)
            pbot = meta_infos.get("jitter_pad_bot", 0)
            left_shift = min(cut_x, max(0, -int(pleft)))
            top_shift = min(cut_y, max(0, -int(ptop)))
            right_shift = min(image_size[1] - cut_x, max(0, -int(pright)))
            bot_shift = min(image_size[0] - cut_y, max(0, -int(pbot)))
            out_image, cur_annos = self._blend_moasic(
                cut_x,
                cut_y,
                out_image,
                image,
                copy.deepcopy(annos),
                (mosaic_height, mosaic_width),
                m_idx,
                (left_shift, top_shift, right_shift, bot_shift),
            )
            out_annos.extend(cur_annos)

        return out_image, out_annos

    def _load_image_with_annos(self, dataset_dict):
        """
        Load the image and annotations given a dataset_dict.
        """
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            if not self.return_val_loss:
                dataset_dict.pop("sem_seg_file_name", None)
                return image, None

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other
            # types of data
            # apply meta_infos for mosaic transformation
            annos = [
                transform_instance_annotations(
                    obj, transforms, image_shape, add_meta_infos=self.add_meta_infos
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = None
        return image, annos

    def _apply_boxes(
        self,
        annotations,
        left_shift,
        top_shift,
        cut_width,
        cut_height,
        cut_start_x,
        cut_start_y,
    ):
        """
        Modify the boxes' coordinates according to shifts and cut_starts.
        """
        for annotation in annotations:
            bboxes = BoxMode.convert(
                annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS
            )
            bboxes = np.asarray(bboxes)
            bboxes[0::2] -= left_shift
            bboxes[1::2] -= top_shift

            bboxes[0::2] = np.clip(bboxes[0::2], 0, cut_width)
            bboxes[1::2] = np.clip(bboxes[1::2], 0, cut_height)
            bboxes[0::2] += cut_start_x
            bboxes[1::2] += cut_start_y
            annotation["bbox"] = bboxes
            annotation["bbox_mode"] = BoxMode.XYXY_ABS
        return annotations

    def _blend_moasic(
        self, cut_x, cut_y, target_img, img, annos, img_size, blend_index, four_shifts
    ):
        """
        Blend the images and annotations in Mosaic transform.
        """
        h, w = img_size
        img_h, img_w = img.shape[:2]
        left_shift = min(four_shifts[0], img_w - cut_x)
        top_shift = min(four_shifts[1], img_h - cut_y)
        right_shift = min(four_shifts[2], img_w - (w - cut_x))
        bot_shift = min(four_shifts[3], img_h - (h - cut_y))

        if blend_index == 0:
            annos = self._apply_boxes(annos, left_shift, top_shift, cut_x, cut_y, 0, 0)
            target_img[:cut_y, :cut_x] = img[
                top_shift : top_shift + cut_y, left_shift : left_shift + cut_x
            ]
        if blend_index == 1:
            annos = self._apply_boxes(
                annos,
                img_w + cut_x - w - right_shift,
                top_shift,
                w - cut_x,
                cut_y,
                cut_x,
                0,
            )
            target_img[:cut_y, cut_x:] = img[
                top_shift : top_shift + cut_y,
                img_w + cut_x - w - right_shift : img_w - right_shift,
            ]
        if blend_index == 2:
            annos = self._apply_boxes(
                annos,
                left_shift,
                img_h + cut_y - h - bot_shift,
                cut_x,
                h - cut_y,
                0,
                cut_y,
            )
            target_img[cut_y:, :cut_x] = img[
                img_h + cut_y - h - bot_shift : img_h - bot_shift,
                left_shift : left_shift + cut_x,
            ]
        if blend_index == 3:
            annos = self._apply_boxes(
                annos,
                img_w + cut_x - w - right_shift,
                img_h + cut_y - h - bot_shift,
                w - cut_x,
                h - cut_y,
                cut_x,
                cut_y,
            )
            target_img[cut_y:, cut_x:] = img[
                img_h + cut_y - h - bot_shift : img_h - bot_shift,
                img_w + cut_x - w - right_shift : img_w - right_shift,
            ]
        return target_img, annos
