"""Metrics that distinguish clean accuracy from triggered behavior."""

import torch
from utils.train_eval import evaluate
from attacks.backdoor import apply_trigger_to_tensor
import config


@torch.no_grad()
def evaluate_asr_and_triggered_true_accuracy(model, loader):
    model.eval(); model.to(config.DEVICE)
    success = eligible = true_correct = 0
    for images, labels, *_ in loader:
        mask = labels != config.BACKDOOR_TARGET_LABEL
        if not mask.any(): continue
        images, labels = images[mask].to(config.DEVICE), labels[mask].to(config.DEVICE)
        preds = model(apply_trigger_to_tensor(images)).argmax(1)
        success += int((preds == config.BACKDOOR_TARGET_LABEL).sum())
        true_correct += int((preds == labels).sum())
        eligible += len(labels)
    if not eligible:
        return {"asr": None, "triggered_true_label_accuracy": None, "eligible": 0}
    return {"asr": 100.0 * success / eligible,
            "triggered_true_label_accuracy": 100.0 * true_correct / eligible,
            "eligible": eligible}


def evaluate_model(model, loader):
    loss, accuracy = evaluate(model, loader)
    return {"loss": loss, "accuracy": accuracy}
