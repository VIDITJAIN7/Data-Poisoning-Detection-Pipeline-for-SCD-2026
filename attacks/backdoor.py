"""BadNets-style backdoor attack: stamps a trigger patch and relabels."""

import random
from typing import Callable, Set, Tuple

import torch
from PIL import Image
from torchvision import datasets

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def apply_trigger_patch(image, patch_size=config.BACKDOOR_PATCH_SIZE):
    """Stamp the standard fixed white trigger in the bottom-right corner."""
    img = image.copy()
    pixels = img.load()
    w, h = img.size
    for dx in range(patch_size):
        for dy in range(patch_size):
            pixels[w - 1 - dx, h - 1 - dy] = (255, 255, 255)
    return img


def select_backdoor_indices(dataset, target_label=config.BACKDOOR_TARGET_LABEL,
                            poison_rate=config.POISON_RATE, seed=config.SEED):
    """Choose random non-target samples to receive the trigger."""
    rng = random.Random(seed)
    eligible = [i for i, (_, lbl) in enumerate(dataset) if lbl != target_label]
    n_poison = int(len(dataset) * poison_rate)
    if n_poison > len(eligible):
        raise ValueError(
            f"Requested {n_poison} backdoor samples but only "
            f"{len(eligible)} non-target samples are eligible"
        )
    return set(rng.sample(eligible, n_poison))


def make_backdoor_fn(target_label=config.BACKDOOR_TARGET_LABEL,
                     patch_size=config.BACKDOOR_PATCH_SIZE):
    def backdoor_fn(image, label):
        return apply_trigger_patch(image, patch_size), target_label
    return backdoor_fn


def prepare_backdoor_attack(poison_rate=config.POISON_RATE):
    """Return (poison_indices, poison_fn) for backdoor attack."""
    dataset = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    indices = select_backdoor_indices(dataset, poison_rate=poison_rate)
    fn = make_backdoor_fn()
    print(f"[Backdoor] Poisoning {len(indices)} samples with "
          f"{config.BACKDOOR_PATCH_SIZE}×{config.BACKDOOR_PATCH_SIZE} trigger "
          f"→ label '{config.BACKDOOR_TARGET_LABEL}'")
    return indices, fn


def apply_trigger_to_tensor(tensor_img, patch_size=config.BACKDOOR_PATCH_SIZE):
    """Apply trigger to normalised tensor (for test-time ASR evaluation)."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    white_norm = (1.0 - mean) / std

    patched = tensor_img.clone()
    if patched.dim() == 3:
        for c in range(3):
            patched[c, -patch_size:, -patch_size:] = white_norm[c, 0, 0]
    elif patched.dim() == 4:
        for c in range(3):
            patched[:, c, -patch_size:, -patch_size:] = white_norm[c, 0, 0]
    return patched
