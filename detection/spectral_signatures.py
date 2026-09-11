"""Spectral-signature detection (Tran et al., NeurIPS 2018).

Projects per-class features onto the top singular vector; poisoned samples
cluster at one end due to the consistent backdoor signal.
"""

from typing import Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


@torch.no_grad()
def extract_features_and_labels(model, loader, device=config.DEVICE):
    """Return (features, labels, indices) as numpy arrays."""
    model.eval()
    model = model.to(device)
    feats, labs, idxs = [], [], []

    for batch in loader:
        images, labels, indices = batch
        with autocast(enabled=config.USE_AMP):
            f = model.extract_features(images.to(device))
        feats.append(f.float().cpu().numpy())
        labs.append(labels.numpy())
        idxs.append(indices.numpy())

    return np.concatenate(feats), np.concatenate(labs), np.concatenate(idxs)


def detect_spectral_signatures(model, loader, epsilon=config.SPECTRAL_EPSILON):
    """Returns (flagged_set, scores, avg_threshold, indices)."""
    features, labels, indices = extract_features_and_labels(model, loader)
    scores = np.zeros(len(features), dtype=np.float32)
    flagged = set()
    thresholds = []

    for c in range(10):
        mask = labels == c
        if mask.sum() == 0:
            continue
        feats_c = features[mask]
        centred = feats_c - feats_c.mean(axis=0, keepdims=True)

        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        proj = np.abs(centred @ Vt[0])
        scores[mask] = proj

        thr = proj.mean() + epsilon * proj.std()
        thresholds.append(thr)
        flagged.update(indices[mask][proj > thr].tolist())

    return flagged, scores, float(np.mean(thresholds)) if thresholds else 0.0, indices
