"""KNN label-agreement detection.

Flags samples whose K nearest neighbours in feature space mostly
disagree on label — a strong signal for mislabelled / poisoned data.
"""

from typing import Set, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from detection.spectral_signatures import extract_features_and_labels


def detect_nn_label_agreement(model, loader, k=config.KNN_K,
                              agreement_threshold=0.5):
    """Returns (flagged_set, agreement_array, indices)."""
    features, labels, indices = extract_features_and_labels(model, loader)

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn_model.fit(features)
    _, knn_indices = nn_model.kneighbors(features)
    knn_indices = knn_indices[:, 1:]  # drop self

    neighbour_labels = labels[knn_indices]
    agreement = (neighbour_labels == labels[:, None]).mean(axis=1)
    flagged = set(indices[agreement < agreement_threshold].tolist())

    return flagged, agreement, indices
