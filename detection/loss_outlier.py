"""Loss-based outlier detection: flags high-loss samples as suspicious."""

from typing import Set

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


@torch.no_grad()
def compute_per_sample_loss(model, loader, device=config.DEVICE):
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses, all_indices = [], []

    for batch in loader:
        images, labels, indices = batch
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=config.USE_AMP):
            losses = criterion(model(images), labels)
        all_losses.append(losses.cpu().numpy())
        all_indices.append(indices.numpy())

    return np.concatenate(all_losses), np.concatenate(all_indices)


def detect_loss_outliers(model, loader, percentile=config.LOSS_OUTLIER_PERCENTILE):
    """Flag samples with loss above the given percentile.
    Returns (flagged_set, losses, threshold, indices).
    """
    losses, indices = compute_per_sample_loss(model, loader)
    threshold = np.percentile(losses, percentile)
    flagged = set(indices[losses > threshold].tolist())
    return flagged, losses, threshold, indices
