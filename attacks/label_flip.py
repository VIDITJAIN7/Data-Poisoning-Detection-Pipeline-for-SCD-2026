"""Label-flipping attack: changes labels of source-class samples to a target class."""

import random
from typing import Callable, Set, Tuple

from PIL import Image
from torchvision import datasets

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def select_flip_indices(dataset, source_class=config.LABEL_FLIP_SOURCE,
                        poison_rate=config.POISON_RATE, seed=config.SEED):
    """Return indices of source-class samples to flip."""
    rng = random.Random(seed)
    source_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == source_class]
    n_poison = int(len(dataset) * poison_rate)
    if n_poison > len(source_indices):
        raise ValueError(
            f"Requested {n_poison} flips but source class contains only "
            f"{len(source_indices)} samples"
        )
    return set(rng.sample(source_indices, n_poison))


def make_flip_fn(target_class=config.LABEL_FLIP_TARGET):
    def flip_fn(image, label):
        return image, target_class
    return flip_fn


def prepare_label_flip_attack(poison_rate=config.POISON_RATE):
    """Return (poison_indices, poison_fn) for label-flip attack."""
    dataset = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    indices = select_flip_indices(dataset, poison_rate=poison_rate)
    fn = make_flip_fn()
    print(f"[Label-Flip] Poisoning {len(indices)} samples "
          f"({CIFAR10_CLASSES[config.LABEL_FLIP_SOURCE]} → "
          f"{CIFAR10_CLASSES[config.LABEL_FLIP_TARGET]})")
    return indices, fn
