"""Deterministic signature scan for a configured fixed-patch trigger.

This is a provenance defense for the known BadNets-style trigger used by the
demo. It is reported separately from model-based generic detectors.
"""

import numpy as np
from PIL import Image
from torchvision import datasets

import config


def detect_fixed_trigger(dataset, patch_size=config.BACKDOOR_PATCH_SIZE,
                         tolerance=0):
    """Return IDs whose bottom-right patch is uniformly white within tolerance."""
    flagged = set()
    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item[0]
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
        patch = pixels[-patch_size:, -patch_size:]
        if np.all(patch >= 255 - tolerance):
            flagged.add(idx)
    return flagged


def detect_fixed_trigger_cifar10():
    return detect_fixed_trigger(
        datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    )
