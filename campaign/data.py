"""Immutable CIFAR-10 split and observed-data views for the campaign."""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import config

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class CampaignSplit:
    attack_ids: tuple
    trusted_ids: tuple
    dev_ids: tuple

    @property
    def all_ids(self):
        return self.attack_ids + self.trusted_ids + self.dev_ids


def make_stratified_split(base, seed: int) -> CampaignSplit:
    """Return deterministic 45k/2k/3k IDs, stratified over ten classes."""
    labels = np.asarray(base.targets)
    rng = np.random.default_rng(seed)
    attack, trusted, dev = [], [], []
    for cls in range(10):
        ids = np.flatnonzero(labels == cls)
        ids = rng.permutation(ids)
        trusted.extend(ids[:200].tolist())
        dev.extend(ids[200:500].tolist())
        attack.extend(ids[500:].tolist())
    return CampaignSplit(tuple(sorted(attack)), tuple(sorted(trusted)), tuple(sorted(dev)))


def raw_to_normalized(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class ImmutableObservedDataset(Dataset):
    """Observed training data; source IDs remain global CIFAR-10 IDs."""

    def __init__(self, base, ids: Iterable[int], labels: Mapping[int, int],
                 poisoned: Set[int], attack_kind: Optional[str], train: bool):
        self.base = base
        self.ids = tuple(int(i) for i in ids)
        self.labels = dict(labels)
        self.poisoned = set(poisoned)
        self.attack_kind = attack_kind
        self.train = train
        self.crop = transforms.RandomCrop(32, padding=4)
        self.flip = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.ids)

    def _transform(self, image, sample_id):
        if self.train:
            image = self.crop(image)
            image = self.flip(image)
        tensor = self.to_tensor(image)
        if sample_id in self.poisoned and self.attack_kind == "backdoor":
            tensor[:, -config.BACKDOOR_PATCH_SIZE:, -config.BACKDOOR_PATCH_SIZE:] = 1.0
        return self.normalize(tensor)

    def __getitem__(self, position):
        sample_id = self.ids[position]
        image, clean_label = self.base[sample_id]
        label = int(self.labels.get(sample_id, clean_label))
        if sample_id in self.poisoned and self.attack_kind == "backdoor":
            label = config.BACKDOOR_TARGET_LABEL
        elif sample_id in self.poisoned and self.attack_kind == "label":
            label = int(self.labels[sample_id])
        return self._transform(image, sample_id), label, sample_id


class TrustedDataset(Dataset):
    def __init__(self, base, ids):
        self.base, self.ids = base, tuple(int(i) for i in ids)
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, position):
        sample_id = self.ids[position]
        image, label = self.base[sample_id]
        return self.transform(image), int(label), sample_id


def load_train_base():
    return datasets.CIFAR10(config.DATA_DIR, train=True, download=True)


def load_test_base():
    return datasets.CIFAR10(config.DATA_DIR, train=False, download=True)


def clean_eval_dataset(base):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    class Eval(Dataset):
        def __len__(self): return len(base)
        def __getitem__(self, i):
            image, label = base[i]
            return transform(image), int(label), i
    return Eval()


def poison_labels(base, ids: Iterable[int], attack_kind: str) -> Dict[int, int]:
    labels = {int(i): int(base.targets[i]) for i in ids}
    if attack_kind == "label":
        for i in ids:
            if labels[int(i)] == config.LABEL_FLIP_SOURCE:
                labels[int(i)] = config.LABEL_FLIP_TARGET
    return labels
