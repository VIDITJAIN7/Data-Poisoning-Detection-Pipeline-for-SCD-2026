"""Activation-clustering detection (Chen et al., 2019).

Per class: PCA to 2-D → KMeans(K=2) → flag the smaller cluster.
"""

from typing import Dict, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from detection.spectral_signatures import extract_features_and_labels


def detect_activation_clusters(model, loader, n_clusters=config.ACTIVATION_N_CLUSTERS):
    """Returns (flagged_set, debug_info_per_class)."""
    features, labels, indices = extract_features_and_labels(model, loader)
    flagged = set()
    debug_info = {}

    for c in range(10):
        mask = labels == c
        if mask.sum() < n_clusters:
            continue

        feats_c = features[mask]
        pca = PCA(n_components=min(2, feats_c.shape[1]))
        feats_2d = pca.fit_transform(feats_c)

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=config.SEED)
        cluster_labels = km.fit_predict(feats_2d)

        minority = int(np.argmin(np.bincount(cluster_labels, minlength=n_clusters)))
        flagged.update(indices[mask][cluster_labels == minority].tolist())

        debug_info[c] = {
            "features_2d": feats_2d,
            "cluster_labels": cluster_labels,
            "indices": indices[mask],
        }

    return flagged, debug_info
