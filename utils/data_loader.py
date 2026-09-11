"""CIFAR-10 data loading, poisoning wrappers, and DataLoader factories."""

import random
from typing import Callable, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])


def get_raw_transform():
    """ToTensor + normalise, no augmentation (deterministic)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])


class PoisonedDataset(Dataset):
    """Wraps a dataset with per-sample poisoning via ``poison_fn``."""

    def __init__(self, base_dataset, poison_indices, poison_fn, transform=None):
        self.base = base_dataset
        self.poison_indices = poison_indices
        self.poison_fn = poison_fn
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        if idx in self.poison_indices:
            image, label = self.poison_fn(image, label)
        if self.transform:
            image = self.transform(image)
        return image, label, idx


class CleanIndexDataset(Dataset):
    """Wrapper that yields (image, label, index)."""

    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx


class LabelCorrectedDataset(Dataset):
    """Wrapper that replaces labels for detector-selected sample indices."""

    def __init__(self, base_dataset, corrections, transform=None):
        self.base = base_dataset
        self.corrections = corrections
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        image, label = item[0], item[1]
        sample_id = item[2] if len(item) > 2 else idx
        label = self.corrections.get(idx, label)
        if self.transform:
            image = self.transform(image)
        return image, label, sample_id


def _make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=True,
    )


def get_clean_loaders():
    """Return (train_loader, test_loader) for clean CIFAR-10."""
    train_ds = CleanIndexDataset(
        datasets.CIFAR10(config.DATA_DIR, train=True, download=True),
        transform=get_train_transform(),
    )
    test_ds = CleanIndexDataset(
        datasets.CIFAR10(config.DATA_DIR, train=False, download=True),
        transform=get_test_transform(),
    )
    return _make_loader(train_ds, config.BATCH_SIZE, True), \
           _make_loader(test_ds, config.EVAL_BATCH_SIZE, False)


def build_poisoned_loader(poison_indices, poison_fn, shuffle=True):
    base = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    ds = PoisonedDataset(base, poison_indices, poison_fn, get_train_transform())
    return _make_loader(ds, config.BATCH_SIZE, shuffle)


def build_raw_loader(train=True, batch_size=None):
    """No-augmentation loader for detection passes."""
    base = datasets.CIFAR10(config.DATA_DIR, train=train, download=True)
    ds = CleanIndexDataset(base, transform=get_raw_transform())
    return _make_loader(ds, batch_size or config.EVAL_BATCH_SIZE, False)


def build_poisoned_raw_loader(poison_indices, poison_fn):
    """No-augmentation poisoned loader for deterministic detection."""
    base = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    ds = PoisonedDataset(base, poison_indices, poison_fn, get_raw_transform())
    return _make_loader(ds, config.EVAL_BATCH_SIZE, False)


def build_clean_subset_loader(exclude_indices, poison_indices=None, poison_fn=None):
    """Training loader that excludes IDs from the supplied training dataset.

    If an attack is supplied, the poisoned image and observed poisoned label
    are retained for every sample that was *not* selected by the detector.
    This prevents cleaning from accidentally restoring hidden ground truth.
    """
    base = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    keep = [i for i in range(len(base)) if i not in exclude_indices]
    if poison_indices is not None and poison_fn is None:
        raise ValueError("poison_fn is required when poison_indices is supplied")
    if poison_indices is not None:
        ds = PoisonedDataset(Subset(base, keep),
                             {keep.index(i) for i in poison_indices if i in keep},
                             poison_fn, get_train_transform())
    else:
        ds = CleanIndexDataset(Subset(base, keep), transform=get_train_transform())
    return _make_loader(ds, config.BATCH_SIZE, True)


def build_label_corrected_loader(corrections, poison_indices=None, poison_fn=None):
    """Training loader preserving poisoned observations and selected fixes."""
    base = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    if poison_indices is not None and poison_fn is None:
        raise ValueError("poison_fn is required when poison_indices is supplied")
    source = base if poison_indices is None else PoisonedDataset(
        base, poison_indices, poison_fn, transform=None)
    ds = LabelCorrectedDataset(source, corrections, transform=get_train_transform())
    return _make_loader(ds, config.BATCH_SIZE, True)
