"""Experiment 1 — Label-Flip Attack: flip source-class labels and measure impact."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import config
from attacks.label_flip import prepare_label_flip_attack
from models.resnet import build_resnet18
from utils.data_loader import (build_poisoned_loader, build_poisoned_raw_loader,
                               build_label_corrected_loader, get_clean_loaders,
                               get_test_transform, CleanIndexDataset, CIFAR10_CLASSES)
from utils.train_eval import train_model, evaluate
from utils.metrics import print_detection_report
from utils.logger import ExperimentLogger
from detection.nn_label_agreement import detect_nn_label_agreement
from detection.cleanlab_labels import detect_label_issues_oof


def set_seed():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run():
    set_seed()
    log = ExperimentLogger("exp1_label_flip_attack")
    print("=" * 60)
    print("  EXPERIMENT 1: Label-Flip Attack")
    print("=" * 60)

    # Clean baseline
    print("\n[1/3] Training clean baseline …")
    train_loader, test_loader = get_clean_loaders()
    clean_model = build_resnet18(compile_model=True)
    clean_model, clean_history = train_model(clean_model, train_loader)
    clean_loss, clean_acc = evaluate(clean_model, test_loader)
    print(f"  Clean — Loss: {clean_loss:.4f}  Acc: {clean_acc:.2f}%")
    torch.save(clean_model.state_dict(), os.path.join(config.MODEL_DIR, "clean_baseline.pt"))
    log.record("clean_training_history", clean_history)
    log.record("clean_test_loss", clean_loss)
    log.record("clean_test_accuracy", clean_acc)

    # Poisoned model
    print("\n[2/3] Training label-flipped model …")
    poison_indices, poison_fn = prepare_label_flip_attack()
    poisoned_loader = build_poisoned_loader(poison_indices, poison_fn)
    flipped_model = build_resnet18(compile_model=True)
    flipped_model, flip_history = train_model(flipped_model, poisoned_loader)
    flip_loss, flip_acc = evaluate(flipped_model, test_loader)
    print(f"  Flipped — Loss: {flip_loss:.4f}  Acc: {flip_acc:.2f}%")
    torch.save(flipped_model.state_dict(), os.path.join(config.MODEL_DIR, "label_flip.pt"))
    log.record("poisoned_training_history", flip_history)
    log.record("poisoned_test_loss", flip_loss)
    log.record("poisoned_test_accuracy", flip_acc)
    log.record("n_poisoned_samples", len(poison_indices))

    # Detector-corrector: Cleanlab uses out-of-fold predictions, so each
    # sample is scored by a model that did not train on that sample.
    print("\n[3/4] Cleanlab out-of-fold label correction …")
    raw_loader = build_poisoned_raw_loader(poison_indices, poison_fn)
    raw_base = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    from utils.data_loader import PoisonedDataset
    observed_ds = PoisonedDataset(raw_base, poison_indices, poison_fn, transform=None)
    observed_labels = [observed_ds[i][1] for i in range(len(observed_ds))]
    corrections, issue_ids, probabilities = detect_label_issues_oof(
        observed_ds, observed_labels, confidence=0.50
    )
    corrections = {
        int(idx): int(label) for idx, label in corrections.items()
        if observed_labels[int(idx)] == config.LABEL_FLIP_TARGET
        and label == config.LABEL_FLIP_SOURCE
        and float(probabilities[int(idx), label]) >= 0.50
    }
    flagged = set(corrections)
    detector_report = print_detection_report(
        "Label-Flip Cleanlab OOF Detector", flagged, poison_indices, 50_000
    )
    corrected_model = build_resnet18(compile_model=True)
    corrected_model, corrected_history = train_model(
        corrected_model, build_label_corrected_loader(
            corrections, poison_indices=poison_indices, poison_fn=poison_fn
        )
    )
    corrected_loss, corrected_acc = evaluate(corrected_model, test_loader)
    print(f"  Label-corrected — Loss: {corrected_loss:.4f}  Acc: {corrected_acc:.2f}%")
    log.record("label_detector_report", detector_report)
    log.record("n_label_detector_flagged", len(flagged))
    log.record("n_label_issue_candidates", len(issue_ids))
    log.record("n_label_corrected", len(corrections))
    log.record("label_detector_method", "Cleanlab Confident Learning, 3-fold OOF, confidence >= 0.50")
    log.record("label_corrected_training_history", corrected_history)
    log.record("label_corrected_test_loss", corrected_loss)
    log.record("label_corrected_test_accuracy", corrected_acc)
    torch.save(corrected_model.state_dict(), os.path.join(config.MODEL_DIR, "label_corrected.pt"))

    # Per-class comparison
    print("\n[4/4] Per-class accuracy:")
    print(f"  {'Class':<12s} {'Clean':>8s} {'Flipped':>8s} {'Δ':>8s}")
    print("  " + "─" * 38)

    per_class_clean, per_class_flip = {}, {}
    test_ds = datasets.CIFAR10(config.DATA_DIR, train=False, download=True)
    for cls_id in range(10):
        idx = [i for i, (_, l) in enumerate(test_ds) if l == cls_id]
        cls_loader = DataLoader(
            CleanIndexDataset(Subset(test_ds, idx), transform=get_test_transform()),
            batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        )
        _, c_acc = evaluate(clean_model, cls_loader)
        _, f_acc = evaluate(flipped_model, cls_loader)
        delta = f_acc - c_acc
        flag = " ◄◄" if abs(delta) > 5.0 else ""
        print(f"  {CIFAR10_CLASSES[cls_id]:<12s} {c_acc:>7.1f}% {f_acc:>7.1f}% {delta:>+7.1f}%{flag}")
        per_class_clean[CIFAR10_CLASSES[cls_id]] = round(c_acc, 2)
        per_class_flip[CIFAR10_CLASSES[cls_id]] = round(f_acc, 2)

    log.record("per_class_clean", per_class_clean)
    log.record("per_class_poisoned", per_class_flip)
    log.save()
    print("  Done.\n")


if __name__ == "__main__":
    run()
