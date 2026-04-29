import numpy as np
from PIL import Image

from detectron2.data.transforms import Augmentation, NoOpTransform, TransformList

from .transform import (
    YOLOFJitterCropTransform,
    YOLOFDistortTransform,
    HFlipTransform,
    VFlipTransform,
    ResizeTransform,
    YOLOFShiftTransform,
    ColorJitterTransform,
    EqualizeTransform,
    PosterizeTransform,
    SolarizeTransform,
    AutoContrastTransform,
    SharpnessTransform,
    SmallAngleRotateTransform,
)

__all__ = [
    "YOLOFJitterCrop",
    "YOLOFResize",
    "YOLOFRandomDistortion",
    "RandomFlip",
    "YOLOFRandomShift",
    "ColorJitter",
    "DetectionAutoAugment",
]


class YOLOFJitterCrop(Augmentation):
    """Jitter and crop the image and box."""

    def __init__(self, jitter_ratio):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        oh, ow = image.shape[:2]
        dw = int(ow * self.jitter_ratio)
        dh = int(oh * self.jitter_ratio)
        pleft = np.random.randint(-dw, dw)
        pright = np.random.randint(-dw, dw)
        ptop = np.random.randint(-dh, dh)
        pbot = np.random.randint(-dh, dh)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot
        return YOLOFJitterCropTransform(
            pleft=pleft,
            pright=pright,
            ptop=ptop,
            pbot=pbot,
            output_size=(swidth, sheight),
        )


class YOLOFResize(Augmentation):
    """
    Resize image to a target size
    """

    def __init__(self, shape, interp=Image.BILINEAR, scale_jitter=None):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
            scale_jitter (optional, tuple[float, float]): None or (0.8, 1.2)
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        elif (isinstance(shape, tuple) and len(shape) == 1):
            shape = (shape[0], shape[0])
        shape = tuple(shape)
        assert scale_jitter is None or isinstance(scale_jitter, tuple)
        self._init(locals())

    def get_transform(self, image):
        if self.scale_jitter is not None:
            if len(self.scale_jitter) > 2:
                assert isinstance(self.scale_jitter[0], tuple)
                idx = np.random.choice(range(len(self.scale_jitter)))
                shape = self.scale_jitter[idx]
            else:
                jitter = np.random.uniform(self.scale_jitter[0], self.scale_jitter[1])
                shape = (int(self.shape[0] * jitter), int(self.shape[1] * jitter))
        else:
            shape = self.shape
        return ResizeTransform(
            image.shape[0], image.shape[1], shape[0], shape[1], self.interp
        )


class YOLOFRandomDistortion(Augmentation):
    """
    Random distort image's hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        """
        RandomDistortion Initialization.
        Args:
            hue (float): value of hue
            saturation (float): value of saturation
            exposure (float): value of exposure
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        return YOLOFDistortTransform(self.hue, self.saturation, self.exposure)


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class YOLOFRandomShift(Augmentation):
    """
    Shift the image and box given shift pixels and probability.
    """

    def __init__(self, prob=0.5, max_shifts=32):
        """
        Args:
            prob (float): probability of shifts.
            max_shifts (int): the max pixels for shifting.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            shift_x = np.random.randint(low=-self.max_shifts, high=self.max_shifts)
            shift_y = np.random.randint(low=-self.max_shifts, high=self.max_shifts)
            return YOLOFShiftTransform(shift_x, shift_y)
        else:
            return NoOpTransform()


class ColorJitter(Augmentation):
    """
    PIL-style color jitter: randomly adjusts brightness, contrast,
    saturation, and hue in RGB space.
    """

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        """
        Args:
            brightness (float): max deviation from 1.0, range [0, 1).
            contrast (float): max deviation from 1.0, range [0, 1).
            saturation (float): max deviation from 1.0, range [0, 1).
            hue (float): max absolute shift in hue, range [0, 0.5).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        return ColorJitterTransform(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


class DetectionAutoAugment(Augmentation):
    """
    Detection-aware AutoAugment.  For each image, randomly selects one of
    several sub-policies.  Each sub-policy is a sequence of 2 operations,
    each applied with an independent probability.

    Policies:
        A: ColorJitter(0.3) -> Equalize
        B: ColorJitter(0.2) -> Rotate(±8°, bbox-safe)
        C: Posterize(bits=4-6) -> Equalize
        D: Sharpness(0.3-1.7) -> Brightness(0.2)
        E: Solarize(128) or AutoContrast -> ColorJitter(0.1)
    """

    def __init__(self, num_policies: int = 5):
        super().__init__()
        self._init(locals())

    # Each policy is a list of (transform_factory, probability) pairs.
    # transform_factory is called to get a Transform instance.

    @staticmethod
    def _policy_A():
        """ColorJitter(0.3) -> Equalize"""
        transforms = []
        if np.random.random() < 0.8:
            transforms.append(
                ColorJitterTransform(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0)
            )
        if np.random.random() < 0.6:
            transforms.append(EqualizeTransform())
        return transforms

    @staticmethod
    def _policy_B(image):
        """ColorJitter(0.2) -> Rotate(±8°, bbox-safe)"""
        transforms = []
        if np.random.random() < 0.7:
            transforms.append(
                ColorJitterTransform(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
            )
        if np.random.random() < 0.5:
            angle = np.random.uniform(-8, 8)
            h, w = image.shape[:2]
            transforms.append(SmallAngleRotateTransform(angle=angle, h=h, w=w))
        return transforms

    @staticmethod
    def _policy_C():
        """Posterize(bits=4-6) -> Equalize"""
        transforms = []
        if np.random.random() < 0.7:
            bits = np.random.randint(4, 7)  # 4, 5, or 6
            transforms.append(PosterizeTransform(bits=bits))
        if np.random.random() < 0.6:
            transforms.append(EqualizeTransform())
        return transforms

    @staticmethod
    def _policy_D():
        """Sharpness(0.3-1.7) -> Brightness(0.2)"""
        transforms = []
        if np.random.random() < 0.7:
            factor = np.random.uniform(0.3, 1.7)
            transforms.append(SharpnessTransform(factor=factor))
        if np.random.random() < 0.6:
            transforms.append(
                ColorJitterTransform(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0)
            )
        return transforms

    @staticmethod
    def _policy_E():
        """Solarize(128) or AutoContrast -> ColorJitter(0.1)"""
        transforms = []
        if np.random.random() < 0.6:
            if np.random.random() < 0.5:
                transforms.append(SolarizeTransform(threshold=128))
            else:
                transforms.append(AutoContrastTransform())
        if np.random.random() < 0.7:
            transforms.append(
                ColorJitterTransform(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)
            )
        return transforms

    def get_transform(self, image):
        policy_idx = np.random.randint(0, min(self.num_policies, 5))
        if policy_idx == 0:
            transforms = self._policy_A()
        elif policy_idx == 1:
            transforms = self._policy_B(image)
        elif policy_idx == 2:
            transforms = self._policy_C()
        elif policy_idx == 3:
            transforms = self._policy_D()
        else:
            transforms = self._policy_E()

        if len(transforms) == 0:
            return NoOpTransform()
        if len(transforms) == 1:
            return transforms[0]
        return TransformList(transforms)
