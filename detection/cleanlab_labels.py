"""Label issue detection with Cleanlab and honest out-of-fold predictions."""

import numpy as np
from torch.utils.data import DataLoader, Subset
from cleanlab.filter import find_label_issues

import config
from models.resnet import build_resnet18
from utils.data_loader import get_train_transform, get_test_transform
from utils.train_eval import train_model, predict_indexed_probabilities


def detect_label_issues_oof(base_dataset, observed_labels, model_factory=build_resnet18,
                            folds=3, confidence=0.90):
    """Return selected corrections from models that did not see each sample.

    ``base_dataset`` must expose the observed (possibly poisoned) labels.
    No clean labels or poison indices are accepted by this function.
    """
    n = len(base_dataset)
    observed_labels = np.asarray(observed_labels, dtype=np.int64)
    if len(observed_labels) != n:
        raise ValueError("observed_labels must match base_dataset length")
    rng = np.random.default_rng(config.SEED)
    order = rng.permutation(n)
    fold_ids = np.array_split(order, folds)
    probs = np.zeros((n, 10), dtype=np.float32)

    class TransformedView:
        def __init__(self, source, transform):
            self.source, self.transform = source, transform
        def __len__(self):
            return len(self.source)
        def __getitem__(self, idx):
            item = self.source[idx]
            return self.transform(item[0]), item[1], item[2] if len(item) > 2 else idx

    for held_out in fold_ids:
        train_ids = np.setdiff1d(np.arange(n), held_out, assume_unique=False)
        train_loader = DataLoader(
            Subset(TransformedView(base_dataset, get_train_transform()), train_ids),
            batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
        )
        test_loader = DataLoader(
            Subset(TransformedView(base_dataset, get_test_transform()), held_out),
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
        )
        model, _ = train_model(model_factory(compile_model=True), train_loader,
                               silent=True)
        fold_probs = predict_indexed_probabilities(model, test_loader)
        for idx, row in fold_probs.items():
            probs[idx] = row

    issue_ids = find_label_issues(labels=observed_labels, pred_probs=probs,
                                  return_indices_ranked_by="normalized_margin")
    corrections = {}
    for idx in issue_ids:
        predicted = int(probs[idx].argmax())
        if predicted != int(observed_labels[idx]) and float(probs[idx, predicted]) >= confidence:
            corrections[int(idx)] = predicted
    return corrections, issue_ids, probs
