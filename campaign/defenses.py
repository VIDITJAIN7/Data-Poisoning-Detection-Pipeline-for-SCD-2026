"""Established, auditable defense implementations used by the campaign."""

from copy import deepcopy
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import torch
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset

import config
from campaign.data import ImmutableObservedDataset, TrustedDataset
from models.resnet import build_resnet18
from utils.train_eval import train_model, predict_indexed_probabilities


def _loader(dataset, batch, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=min(config.NUM_WORKERS, 8),
                      pin_memory=config.PIN_MEMORY, persistent_workers=False,
                      generator=generator)


def oof_cleanlab(base, attack_ids, observed_labels, trusted_ids, seed, epochs):
    """Create OOF probabilities; each sample is scored by a model that did not see it."""
    ids = np.asarray(attack_ids, dtype=np.int64)
    y = np.asarray([observed_labels[int(i)] for i in ids], dtype=np.int64)
    probs = np.zeros((len(ids), 10), dtype=np.float32)
    folds = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train_pos, held_pos) in enumerate(skf.split(ids, y)):
        held_ids = ids[held_pos].tolist()
        train_ids = ids[train_pos].tolist()
        train_ds = ImmutableObservedDataset(base, train_ids, observed_labels, set(), "label", True)
        trusted_ds = TrustedDataset(base, trusted_ids)
        model = build_resnet18(compile_model=False)
        model, history = train_model(model, _loader(ConcatDataset([train_ds, trusted_ds]), config.BATCH_SIZE, True, seed + fold), epochs=epochs)
        held_ds = ImmutableObservedDataset(base, held_ids, observed_labels, set(), "label", False)
        indexed = predict_indexed_probabilities(model, _loader(held_ds, config.EVAL_BATCH_SIZE, False, seed))
        pos_by_id = {int(sample_id): pos for pos, sample_id in enumerate(ids)}
        for sample_id, row in indexed.items():
            probs[pos_by_id[int(sample_id)]] = row
        folds.append({"fold": fold, "held_out_ids": held_ids, "history": history})
    issue_sets = {}
    for mode in ("prune_by_noise_rate", "both"):
        issue_sets[mode] = find_label_issues(
            labels=y, pred_probs=probs, filter_by=mode,
            return_indices_ranked_by="normalized_margin")
    return ids, y, probs, issue_sets, folds


def label_candidates(ids, labels, probs, issue_sets, confidences=(0.70, 0.80, 0.90)):
    candidates = {}
    for mode, issues in issue_sets.items():
        issue_set = set(int(i) for i in issues)
        for confidence in confidences:
            corrections = {}
            for pos in issue_set:
                predicted = int(probs[pos].argmax())
                if predicted != int(labels[pos]) and float(probs[pos, predicted]) >= confidence:
                    corrections[int(ids[pos])] = predicted
            candidates[f"cleanlab_{mode}_replace_{confidence:.2f}"] = {"corrections": corrections, "remove": set()}
        candidates[f"cleanlab_{mode}_remove"] = {"corrections": {}, "remove": {int(ids[pos]) for pos in issue_set}}
    return candidates


def detection_metrics(flagged, poison_ids, universe):
    tp = len(flagged & poison_ids); fp = len(flagged - poison_ids)
    fn = len(poison_ids - flagged); tn = len(universe - flagged - poison_ids)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "fpr": fpr}


def ft_sam_step(model, images, labels, optimizer, rho=0.05):
    """One sharpness-aware update for trusted-data backdoor mitigation."""
    loss_fn = torch.nn.CrossEntropyLoss(); optimizer.zero_grad(set_to_none=True)
    loss_fn(model(images), labels).backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    norm = torch.norm(torch.stack([p.grad.norm() for p in grads]))
    perturbations = []
    with torch.no_grad():
        for p in grads:
            e = p.grad * (rho / (norm + 1e-12)); p.add_(e); perturbations.append((p, e))
    optimizer.zero_grad(set_to_none=True); loss_fn(model(images), labels).backward()
    with torch.no_grad():
        for p, e in perturbations: p.sub_(e)
    optimizer.step()


def train_ft_sam(model, loader, epochs, seed, rho=0.05, lr=0.01):
    """Train a copied poisoned model on trusted data with FT-SAM."""
    torch.manual_seed(seed); model = model.to(config.DEVICE); model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 25], gamma=0.1)
    history = []
    for epoch in range(1, epochs + 1):
        total = 0; running = 0.0
        for images, labels, *_ in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            ft_sam_step(model, images, labels, optimizer, rho)
            running += float(torch.nn.functional.cross_entropy(model(images), labels).detach())
            total += 1
        schedule.step(); history.append({"epoch": epoch, "loss": running / max(total, 1)})
    return model, history


def attach_pruning_mask(model, trusted_loader, fraction: float):
    """Fine-Pruning: mask least-active penultimate channels and enforce the mask."""
    model = model.to(config.DEVICE)
    model.eval()
    activations = []
    with torch.no_grad():
        for batch in trusted_loader:
            features = model.extract_features(batch[0].to(config.DEVICE))
            activations.append(features.abs().mean(dim=0).cpu())
    activity = torch.stack(activations).mean(dim=0)
    n_prune = int(len(activity) * fraction)
    mask = torch.ones_like(activity)
    if n_prune:
        mask[torch.argsort(activity)[:n_prune]] = 0.0
    mask = mask.to(config.DEVICE)

    def hook(_module, inputs):
        return (inputs[0] * mask,)
    handle = model.linear.register_forward_pre_hook(hook)
    with torch.no_grad():
        model.linear.weight.mul_(mask.view(1, -1).to(model.linear.weight.device))
    return mask, handle


def enforce_pruning(model, mask):
    with torch.no_grad():
        model.linear.weight.mul_(mask.view(1, -1).to(model.linear.weight.device))
        if model.linear.bias is not None:
            model.linear.bias.data.clamp_(-100, 100)
