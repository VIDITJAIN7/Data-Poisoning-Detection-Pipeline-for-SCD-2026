"""Training and evaluation with AMP for Ada tensor cores."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def train_model(model, train_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE,
                device=config.DEVICE, silent=False):
    """Train with SGD + step LR + AMP. Returns (model, history)."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.USE_AMP)
    history = []

    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                    leave=False, disable=silent)
        for batch in pbar:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{running_loss/total:.4f}",
                             acc=f"{100.*correct/total:.1f}%")

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        history.append({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_acc})

        if not silent:
            print(f"  Epoch {epoch:>2d} | Loss {epoch_loss:.4f} | Acc {epoch_acc:.2f}%")

    return model, history


@torch.no_grad()
def evaluate(model, loader, device=config.DEVICE):
    """Return (loss, accuracy) on the given loader."""
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        with autocast(enabled=config.USE_AMP):
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def predict_indexed_labels(model, loader, device=config.DEVICE):
    """Return model predictions keyed by the sample indices from a loader."""
    model.eval()
    model = model.to(device)
    predictions = {}
    for batch in loader:
        images, _, indices = batch[0].to(device), batch[1], batch[2]
        with autocast(enabled=config.USE_AMP):
            predicted = model(images).argmax(dim=1).cpu().tolist()
        predictions.update({int(i): int(label) for i, label in zip(indices.tolist(), predicted)})
    return predictions


@torch.no_grad()
def predict_indexed_probabilities(model, loader, device=config.DEVICE):
    """Return out-of-sample-compatible class probabilities keyed by ID."""
    model.eval()
    model = model.to(device)
    predictions = {}
    for batch in loader:
        images, _, indices = batch[0].to(device), batch[1], batch[2]
        with autocast(enabled=config.USE_AMP):
            probabilities = model(images).softmax(dim=1).float().cpu()
        predictions.update({int(i): row.numpy() for i, row in
                           zip(indices.tolist(), probabilities)})
    return predictions


@torch.no_grad()
def evaluate_backdoor_asr(model, test_loader, trigger_fn,
                          target_label=config.BACKDOOR_TARGET_LABEL,
                          device=config.DEVICE):
    """Attack Success Rate: fraction of triggered images classified as target."""
    model.eval()
    model = model.to(device)
    success, total = 0, 0

    for batch in test_loader:
        images, labels = batch[0], batch[1]
        mask = labels != target_label
        if mask.sum() == 0:
            continue
        images = images[mask].to(device)
        triggered = trigger_fn(images)

        with autocast(enabled=config.USE_AMP):
            preds = model(triggered).argmax(dim=1)

        success += (preds == target_label).sum().item()
        total += images.size(0)

    return 100.0 * success / total if total > 0 else 0.0
