import logging
import numpy as np

import detectron2.data.transforms as T
from detectron2.structures import BoxMode

from .augmentation_impl import (
    YOLOFJitterCrop,
    YOLOFResize,
    YOLOFRandomDistortion,
    RandomFlip,
    YOLOFRandomShift,
    ColorJitter,
    DetectionAutoAugment,
)

logger = logging.getLogger(__name__)

# Multi-scale short-side sizes used by various groups
_WIDE_MULTI_SCALE = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    If ``cfg.INPUT.AUG_GROUP`` is set to a value other than ``"none"``,
    the group-specific pipeline is built instead.

    Returns:
        list[Augmentation]
    """
    aug_group = cfg.INPUT.AUG_GROUP
    if aug_group != "none":
        return build_group_augmentation(cfg, is_train)

    is_normal_aug = not cfg.INPUT.RESIZE.ENABLED
    if is_normal_aug:
        augmentation = build_normal_augmentation(cfg, is_train)
    else:
        augmentation = build_yolo_augmentation(cfg, is_train)
    if is_train and cfg.INPUT.SHIFT.ENABLED:
        augmentation.append(YOLOFRandomShift(max_shifts=cfg.INPUT.SHIFT.SHIFT_PIXELS))
    return augmentation


def build_normal_augmentation(cfg, is_train):
    """
    Train Augmentations:
        - ResizeShortestEdge
        - RandomFlip (not for test)
        - Optional: YOLOFRandomDistortion (even if RESIZE is False)
    Test:
        - ResizeShortestEdge
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    # Add color distortion (DISTORTION) even if RESIZE is disabled
    if is_train and cfg.INPUT.DISTORTION.ENABLED:
        from .augmentation_impl import YOLOFRandomDistortion
        augmentation.append(
            YOLOFRandomDistortion(
                hue=cfg.INPUT.DISTORTION.HUE,
                saturation=cfg.INPUT.DISTORTION.SATURATION,
                exposure=cfg.INPUT.DISTORTION.EXPOSURE,
            )
        )
        
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation


def build_yolo_augmentation(cfg, is_train):
    """
    Train Augmentations:
        - YOLOFJitterCrop
        - YOLOFResize
        - YOLOFRandomDistortion
        - RandomFlip
    Test:
        - YOLOFResize
    """
    augmentation = []
    if is_train:
        if cfg.INPUT.JITTER_CROP.ENABLED:
            augmentation.append(YOLOFJitterCrop(cfg.INPUT.JITTER_CROP.JITTER_RATIO))
        augmentation.append(
            YOLOFResize(
                shape=cfg.INPUT.RESIZE.SHAPE, scale_jitter=cfg.INPUT.RESIZE.SCALE_JITTER
            )
        )
        if cfg.INPUT.DISTORTION.ENABLED:
            augmentation.append(
                YOLOFRandomDistortion(
                    hue=cfg.INPUT.DISTORTION.HUE,
                    saturation=cfg.INPUT.DISTORTION.SATURATION,
                    exposure=cfg.INPUT.DISTORTION.EXPOSURE,
                )
            )
        if cfg.INPUT.RANDOM_FLIP != "none":
            # The difference between `T.RandomFlip` and `RandomFlip` is that
            # we register a new method `apply_meta_infos` in `RandomFlip`
            augmentation.append(
                RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
    else:
        augmentation.append(
            YOLOFResize(shape=cfg.INPUT.RESIZE.TEST_SHAPE, scale_jitter=None)
        )
    return augmentation


def transform_instance_annotations(
    annotation, transforms, image_size, *, add_meta_infos=False
):
    """
    Apply transforms to box and meta_infos annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        add_meta_infos (bool): Whether to apply meta_infos.

    Returns:
        dict:
            the same input dict with fields "bbox", "meta_infos"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS
    )
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    # add meta_infos
    if add_meta_infos:
        meta_infos = dict()
        meta_infos = transforms.apply_meta_infos(meta_infos)
        annotation["meta_infos"] = meta_infos
    return annotation


# ============================================================================
# Group-based augmentation builder
# ============================================================================

_VALID_GROUPS = {
    "none", "minimal", "mixup", "cutmix", "mosaic", "mosaic_color", "autoaugment",
}


def build_group_augmentation(cfg, is_train):
    """
    Build the per-image augmentation list driven by ``cfg.INPUT.AUG_GROUP``.

    Multi-image operations (Mixup, CutMix, Mosaic) are handled inside the
    dataset mapper, **not** here.  This function only builds the per-image
    geometric + photometric chain.

    Returns:
        list[Augmentation]
    """
    group = cfg.INPUT.AUG_GROUP
    if group not in _VALID_GROUPS:
        raise ValueError(
            f"Unknown AUG_GROUP '{group}'. Must be one of {_VALID_GROUPS}"
        )

    if not is_train:
        # Inference always uses the same pipeline regardless of group.
        if cfg.INPUT.RESIZE.ENABLED:
            return [YOLOFResize(shape=cfg.INPUT.RESIZE.TEST_SHAPE, scale_jitter=None)]
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        return [T.ResizeShortestEdge(min_size, max_size, "choice")]

    if group == "minimal":
        return _build_minimal(cfg)
    elif group == "mixup":
        return _build_mixup(cfg)
    elif group == "cutmix":
        return _build_cutmix(cfg)
    elif group == "mosaic":
        return _build_mosaic(cfg, with_color=False)
    elif group == "mosaic_color":
        return _build_mosaic(cfg, with_color=True)
    elif group == "autoaugment":
        return _build_autoaugment(cfg)
    else:
        # "none" — shouldn't reach here; caller already handled it.
        return build_normal_augmentation(cfg, is_train=True)


def _build_minimal(cfg):
    """Minimal: narrow multi-scale resize + flip + optional tiny brightness jitter."""
    minimal_cfg = cfg.INPUT.MINIMAL
    min_sizes = tuple(minimal_cfg.MIN_SIZE_TRAIN)
    max_size = minimal_cfg.MAX_SIZE_TRAIN
    flip_prob = float(minimal_cfg.FLIP_PROB)
    brightness_jitter = float(minimal_cfg.BRIGHTNESS_JITTER)

    augmentation = [
        T.ResizeShortestEdge(min_sizes, max_size, "choice"),
        T.RandomFlip(horizontal=True, vertical=False, prob=flip_prob),
    ]

    if brightness_jitter > 0:
        augmentation.append(
            ColorJitter(
                brightness=brightness_jitter,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            )
        )

    return augmentation


def _build_mixup(cfg):
    """Mixup group per-image chain: wide multi-scale + light color jitter + flip."""
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    cj = cfg.INPUT.COLOR_JITTER
    augmentation = [
        T.ResizeShortestEdge(_WIDE_MULTI_SCALE, max_size, "choice"),
        ColorJitter(
            brightness=cj.BRIGHTNESS,
            contrast=cj.CONTRAST,
            saturation=cj.SATURATION,
            hue=cj.HUE,
        ),
        T.RandomFlip(horizontal=True, vertical=False),
    ]
    return augmentation


def _build_cutmix(cfg):
    """CutMix group per-image chain: wide multi-scale + light color jitter + flip."""
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    cj = cfg.INPUT.COLOR_JITTER
    augmentation = [
        T.ResizeShortestEdge(_WIDE_MULTI_SCALE, max_size, "choice"),
        ColorJitter(
            brightness=cj.BRIGHTNESS,
            contrast=cj.CONTRAST,
            saturation=cj.SATURATION,
            hue=cj.HUE,
        ),
        T.RandomFlip(horizontal=True, vertical=False),
    ]
    return augmentation


def _build_mosaic(cfg, with_color: bool):
    """
    Mosaic groups use the YOLO augmentation path (JitterCrop + YOLOFResize)
    because Mosaic depends on ``meta_infos`` from JitterCrop.
    """
    augmentation = []
    if cfg.INPUT.JITTER_CROP.ENABLED:
        augmentation.append(YOLOFJitterCrop(cfg.INPUT.JITTER_CROP.JITTER_RATIO))
    augmentation.append(
        YOLOFResize(
            shape=cfg.INPUT.RESIZE.SHAPE,
            scale_jitter=cfg.INPUT.RESIZE.SCALE_JITTER,
        )
    )
    if with_color and cfg.INPUT.DISTORTION.ENABLED:
        augmentation.append(
            YOLOFRandomDistortion(
                hue=cfg.INPUT.DISTORTION.HUE,
                saturation=cfg.INPUT.DISTORTION.SATURATION,
                exposure=cfg.INPUT.DISTORTION.EXPOSURE,
            )
        )
    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    if cfg.INPUT.SHIFT.ENABLED:
        augmentation.append(YOLOFRandomShift(max_shifts=cfg.INPUT.SHIFT.SHIFT_PIXELS))
    return augmentation


def _build_autoaugment(cfg):
    """AutoAugment group: wide multi-scale + AutoAugment policy + flip."""
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    num_policies = cfg.INPUT.AUTOAUGMENT.NUM_POLICIES
    augmentation = [
        T.ResizeShortestEdge(_WIDE_MULTI_SCALE, max_size, "choice"),
        DetectionAutoAugment(num_policies=num_policies),
        T.RandomFlip(horizontal=True, vertical=False),
    ]
    return augmentation


# ============================================================================
# Box filtering utility
# ============================================================================

def filter_small_boxes(annotations, min_area=16):
    """
    Remove annotations whose bounding box area is below *min_area*.

    Args:
        annotations (list[dict]): each must have ``bbox`` in XYXY_ABS mode.
        min_area (float): minimum area threshold.

    Returns:
        list[dict]: filtered annotations.
    """
    filtered = []
    for ann in annotations:
        bbox = ann["bbox"]
        area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
        if area >= min_area:
            filtered.append(ann)
    return filtered
 