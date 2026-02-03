# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

@dataclass
class MultiCropConfig:
    # DINOv2-ish defaults (tune to your domain)
    n_global: int = 2
    n_local: int = 8

    # Crop scale ranges as fraction of image area
    #  global_scale: Tuple[float, float] = (0.01, 0.05)
    #  local_scale: Tuple[float, float] = (0.001, 0.005)
    global_scale: Tuple[float, float] = (0.1, 0.2)
    local_scale: Tuple[float, float] = (0.01, 0.05)

    # Aspect ratio range (sqrt range common in self-supervised recipes)
    aspect_ratio: Tuple[float, float] = (3/4, 4/3)

    # Anchor heatmap settings
    blur_sigma_px: float = 5.0   # try 6–15 for sparse event images
    heatmap_power: float = 1.0    # >1 emphasizes hotspots, <1 flattens

    # Center jitter around anchor (pixels)
    center_jitter_px: float = 0.0

    # Crop acceptance (prevents empty crops)
    min_active_pixels: int = 2    # tune based on sparsity
    min_total_intensity: float = 0.0  # set >0 if using charge/intensity maps

    # Attempts to find a valid crop before fallback
    max_attempts: int = 50


def _to_activity_array(pil_img: Image.Image, binary: bool = False) -> np.ndarray:
    """
    Convert PIL image to a float activity array A in [0,1] (roughly).
    - If binary=True: A = 1 where pixel > 0 else 0
    - Else: normalize grayscale intensities to [0,1]
    """
    gray = pil_img.convert("L")
    arr = np.asarray(gray).astype(np.float32)
    if binary:
        A = (arr > 0).astype(np.float32)
    else:
        # normalize robustly to reduce sensitivity to outliers
        hi = np.percentile(arr, 99.5)
        if hi <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        A = np.clip(arr / hi, 0.0, 1.0)
    return A


def _gaussian_blur_np(A: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Gaussian blur using PIL
    """
    if sigma_px <= 0:
        return A.copy()
    pil = Image.fromarray(np.uint8(np.clip(A * 255.0, 0, 255)))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=sigma_px))
    H = np.asarray(blurred).astype(np.float32) / 255.0
    return H, blurred


def _sample_anchor_from_heatmap(H: np.ndarray, power: float = 1.0) -> Tuple[int, int]:
    """
    Sample (x,y) from heatmap probabilities proportional to H**power.
    Returns integer pixel coordinates.
    """
    Hp = np.clip(H, 0.0, None)
    if power != 1.0:
        Hp = Hp ** power

    total = float(Hp.sum())
    h, w = Hp.shape

    if total <= 1e-12:
        # Fallback: uniform
        return random.randrange(w), random.randrange(h)

    # Flatten + choice
    p = (Hp / total).ravel()
    idx = np.random.choice(p.size, p=p)
    y, x = divmod(idx, w)
    return int(x), int(y)


def _sample_crop_wh(
    img_w: int,
    img_h: int,
    scale_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
) -> Tuple[int, int]:
    """
    Sample crop (w,h) based on area scale and aspect ratio.
    """
    area = img_w * img_h
    scale = random.uniform(*scale_range)
    target_area = scale * area

    log_ar_min = math.log(aspect_range[0])
    log_ar_max = math.log(aspect_range[1])
    aspect = math.exp(random.uniform(log_ar_min, log_ar_max))

    crop_w = int(round(math.sqrt(target_area * aspect)))
    crop_h = int(round(math.sqrt(target_area / aspect)))

    # Clamp to image bounds
    crop_w = max(1, min(crop_w, img_w))
    crop_h = max(1, min(crop_h, img_h))
    return crop_w, crop_h


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _propose_crop_box(
    anchor_xy: Tuple[int, int],
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
    jitter_px: float,
) -> Tuple[int, int, int, int]:
    """
    Propose a crop box centered near anchor with gaussian jitter, clamped to image.
    Returns (left, top, right, bottom) in PIL coordinates.
    """
    ax, ay = anchor_xy
    cx = ax + random.gauss(0.0, jitter_px)
    cy = ay + random.gauss(0.0, jitter_px)

    # Convert center to top-left
    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))

    # Clamp so box fits
    left = int(_clamp(left, 0, img_w - crop_w))
    top = int(_clamp(top, 0, img_h - crop_h))

    return (left, top, left + crop_w, top + crop_h)


def _crop_is_valid(
    A: np.ndarray,
    box: Tuple[int, int, int, int],
    min_active_pixels: int,
    min_total_intensity: float,
) -> bool:
    left, top, right, bottom = box
    patch = A[top:bottom, left:right]
    if patch.size == 0:
        return False
    active = int((patch > 0).sum())
    if active < min_active_pixels:
        return False
    if float(patch.sum()) < min_total_intensity:
        return False
    return True


# -----------------------------
# Main API
# -----------------------------

def dino_multicrop_anchor_heatmap(
    pil_img: Image.Image,
    out_size_global: int = 224,
    out_size_local: int = 96,
    *,
    config: Optional[MultiCropConfig] = None,
    binary_activity: bool = False,
) -> Dict[str, List[Image.Image]]:
    """
    Implementation of anchor-biased multi-crop (#1) for sparse images.
    Input: PIL image
    Output: dict with 'global' and 'local' lists of PIL crops resized to requested sizes.

    Notes:
    - "Correct transformations" here means: anchor sampling from blurred activity heatmap,
      scale/aspect sampling, jittered center, acceptance filtering to avoid empty crops,
      and resizing to standard DINO crop sizes.
    - If you want color jitter / grayscale / blur augmentations too, apply them AFTER cropping.
    """
    if config is None:
        config = MultiCropConfig()

    img = pil_img
    img_w, img_h = img.size

    # Activity map A in [0,1]
    A = _to_activity_array(img, binary=binary_activity)

    # Heatmap H from blurred activity
    H, blurred_img = _gaussian_blur_np(A, sigma_px=config.blur_sigma_px)

    def _make_one_crop(scale_range: Tuple[float, float], out_size: int) -> Image.Image:
        for _ in range(config.max_attempts):
            anchor = _sample_anchor_from_heatmap(H, power=config.heatmap_power)
            crop_w, crop_h = _sample_crop_wh(img_w, img_h, scale_range, config.aspect_ratio)
            box = _propose_crop_box(
                anchor, crop_w, crop_h, img_w, img_h, jitter_px=config.center_jitter_px
            )
            if _crop_is_valid(A, box, config.min_active_pixels, config.min_total_intensity):
                crop = img.crop(box)
                return crop.resize((out_size, out_size), resample=Image.BICUBIC)

        # Fallback: centered crop if nothing valid
        crop_w, crop_h = _sample_crop_wh(img_w, img_h, scale_range, config.aspect_ratio)
        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        crop = img.crop((left, top, left + crop_w, top + crop_h))
        return crop.resize((out_size, out_size), resample=Image.BICUBIC)

    globals_ = [_make_one_crop(config.global_scale, out_size_global) for _ in range(config.n_global)]
    locals_ = [_make_one_crop(config.local_scale, out_size_local) for _ in range(config.n_local)]

    return {"global": globals_, "local": locals_}


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.cfg = MultiCropConfig(n_local=local_crops_number)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                #  transforms.RandomRotation(degrees=(-20, 20)),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                #  transforms.RandomRotation(degrees=(-20, 20)),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.0)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.0),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])


    def __call__(self, image):
        output = {}

        #  # global crops:
        #  im1_base = self.geometric_augmentation_global(image)
        #  global_crop_1 = self.global_transfo1(im1_base)
        #
        #  im2_base = self.geometric_augmentation_global(image)
        #  global_crop_2 = self.global_transfo2(im2_base)
        #
        #  output["global_crops"] = [global_crop_1, global_crop_2]
        #
        #  # global crops for teacher:
        #  output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        #
        #  # local crops:
        #  local_crops = [
        #      self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        #  ]
        #  output["local_crops"] = local_crops
        cvn_crops = dino_multicrop_anchor_heatmap(
            image,
            out_size_global=self.global_crops_size,
            out_size_local=self.local_crops_size,
            config=self.cfg,
            binary_activity=True,  # set True if your images are binary hit maps
            )

        output["global_crops"] = [self.normalize(crop) for crop in cvn_crops["global"]]
        output["global_crops_teacher"] = [self.normalize(crop) for crop in cvn_crops["global"]]
        output["local_crops"] = [self.normalize(crop) for crop in cvn_crops["local"]]
        output["offsets"] = ()

        return output
