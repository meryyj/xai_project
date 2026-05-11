# src/data/transforms.py

from __future__ import annotations

from typing import Dict, Any

from torchvision import transforms


def _build_common_transforms(cfg: Dict[str, Any], split: str):
    """
    Build the common transforms (resize, grayscale/RGB, ToTensor, Normalize)
    shared between train/val/test.
    """
    img_size = cfg["dataset"]["image_size"]
    channels = cfg["dataset"].get("channels", 1)

    normalize_cfg = cfg.get("preprocessing", {}).get("normalize", {})
    mean = normalize_cfg.get("mean", [0.5])
    std = normalize_cfg.get("std", [0.5])

    t = []

    # Resize to target size
    t.append(transforms.Resize((img_size, img_size)))

    # Color mode: grayscale vs RGB
    if channels == 1:
        t.append(transforms.Grayscale(num_output_channels=1))
    else:
        # do nothing special; PIL open default is RGB
        pass

    # Convert to tensor
    t.append(transforms.ToTensor())

    # Normalize
    t.append(transforms.Normalize(mean=mean, std=std))

    return t


def build_transforms(cfg: Dict[str, Any], split: str):
    """
    Build torchvision transforms for a given split ('train', 'val', 'test').

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from YAML.
    split : str
        One of 'train', 'val', 'test'.

    Returns
    -------
    transforms.Compose
        The composed torchvision transforms for this split.
    """
    split = split.lower()
    aug_cfg = cfg.get("augmentation", {})
    common = _build_common_transforms(cfg, split)

    if split == "train":
        t = []

        # Geometric augmentations
        rotation_deg = aug_cfg.get("rotation_deg", 0)
        translate = aug_cfg.get("translate", 0.0)
        if rotation_deg or translate:
            # translate can be float or tuple; torchvision expects fraction of image size
            max_translate = translate if isinstance(translate, (list, tuple)) else (translate, translate)
            t.append(
                transforms.RandomAffine(
                    degrees=rotation_deg,
                    translate=max_translate,
                )
            )

        if aug_cfg.get("horizontal_flip", True):
            t.append(transforms.RandomHorizontalFlip())

        # Photometric augmentations
        brightness = aug_cfg.get("brightness", 0.0)
        contrast = aug_cfg.get("contrast", 0.0)
        saturation = aug_cfg.get("saturation", 0.0)
        hue = aug_cfg.get("hue", 0.0)

        if any(v > 0 for v in [brightness, contrast, saturation, hue]):
            t.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

        # Then common transforms (resize, grayscale/RGB, ToTensor, Normalize)
        t.extend(common)

        # Random erasing (after ToTensor + Normalize)
        if aug_cfg.get("random_erasing", False):
            t.append(
                transforms.RandomErasing(
                    p=aug_cfg.get("random_erasing_p", 0.5),
                    scale=aug_cfg.get("random_erasing_scale", (0.02, 0.33)),
                    ratio=aug_cfg.get("random_erasing_ratio", (0.3, 3.3)),
                )
            )

        return transforms.Compose(t)

    else:
        # Validation / Test: only common transforms, no heavy augmentation
        return transforms.Compose(common)
