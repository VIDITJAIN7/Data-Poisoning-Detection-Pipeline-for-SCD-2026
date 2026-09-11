"""Experiment 5 — Full End-to-End Demo.

Phase 1: Train on poisoned data (normal accuracy).
Phase 2: Trigger the backdoor (high ASR).
Phase 3: Run detection scanner.
Phase 4: Retrain on cleaned data (trigger fails).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torchvision import datasets

import config
from attacks.backdoor import prepare_backdoor_attack, apply_trigger_patch, apply_trigger_to_tensor
from attacks.label_flip import prepare_label_flip_attack
from models.resnet import build_resnet18
from utils.data_loader import (build_poisoned_loader, build_poisoned_raw_loader,
                                build_clean_subset_loader, get_clean_loaders, CIFAR10_CLASSES)
from utils.train_eval import train_model, evaluate, evaluate_backdoor_asr
from utils.metrics import print_detection_report
from utils.logger import ExperimentLogger
from utils.visualization import (plot_trigger_examples, plot_loss_distribution,
                                  plot_spectral_scores, plot_detection_comparison, plot_model_comparison)
from detection.loss_outlier import detect_loss_outliers
from detection.spectral_signatures import detect_spectral_signatures
from detection.activation_clustering import detect_activation_clusters
from detection.nn_label_agreement import detect_nn_label_agreement


def banner(text):
    print("\n" + "█" * 60)
    print(f"█  {text:<56s}█")
    print("█" * 60)


def set_seed():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run():
    set_seed()
    log = ExperimentLogger("exp5_full_demo")
    n_total = 50_000
    _, test_loader = get_clean_loaders()

    print("╔" + "═" * 58 + "╗")
    print("║   DATA POISONING & DETECTION — FULL DEMO                 ║")
    print("╚" + "═" * 58 + "╝")

    # Phase 1a: Label-Flip
    banner("PHASE 1a: Label-Flip Attack")
    flip_indices, flip_fn = prepare_label_flip_attack()
    flip_model = build_resnet18(compile_model=True)
    flip_model, fh = train_model(flip_model, build_poisoned_loader(flip_indices, flip_fn))
    _, flip_acc = evaluate(flip_model, test_loader)
    print(f"\n  Label-Flip test accuracy: {flip_acc:.2f}%")
    log.record("label_flip_training_history", fh)
    log.record("label_flip_test_accuracy", flip_acc)

    # Phase 1b: Backdoor
    banner("PHASE 1b: Backdoor Attack")
    raw_ds = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    plot_trigger_examples(
        np.stack([np.array(raw_ds[i][0]) for i in range(5)]),
        np.stack([np.array(apply_trigger_patch(raw_ds[i][0])) for i in range(5)]),
        filename="demo_trigger_examples.png",
    )
    bd_indices, bd_fn = prepare_backdoor_attack()
    bd_model = build_resnet18(compile_model=True)
    bd_model, bh = train_model(bd_model, build_poisoned_loader(bd_indices, bd_fn))
    _, bd_acc = evaluate(bd_model, test_loader)
    print(f"\n  Backdoor test accuracy: {bd_acc:.2f}% (looks normal)")
    log.record("backdoor_training_history", bh)
    log.record("backdoor_test_accuracy", bd_acc)

    # Phase 2: Trigger
    banner("PHASE 2: Triggering the Backdoor")
    asr_before = evaluate_backdoor_asr(bd_model, test_loader, apply_trigger_to_tensor)
    print(f"\n  ASR: {asr_before:.2f}% — trigger causes misclassification as "
          f"'{CIFAR10_CLASSES[config.BACKDOOR_TARGET_LABEL]}'")
    log.record("asr_before_cleaning", asr_before)

    # Phase 3: Scan
    banner("PHASE 3: Detection Scanner")
    raw_loader = build_poisoned_raw_loader(bd_indices, bd_fn)

    loss_flagged, losses, lt, li = detect_loss_outliers(bd_model, raw_loader)
    pm = np.array([int(i) in bd_indices for i in li])
    plot_loss_distribution(losses, pm, lt, "demo_loss_dist.png")
    loss_r = print_detection_report("Loss Outlier", loss_flagged, bd_indices, n_total)

    spec_flagged, scores, st, si = detect_spectral_signatures(bd_model, raw_loader)
    pm_s = np.array([int(i) in bd_indices for i in si])
    plot_spectral_scores(scores, pm_s, st, "demo_spectral.png")
    spec_r = print_detection_report("Spectral Sig.", spec_flagged, bd_indices, n_total)

    ac_flagged, _ = detect_activation_clusters(bd_model, raw_loader)
    ac_r = print_detection_report("Act. Clustering", ac_flagged, bd_indices, n_total)

    knn_flagged, _, _ = detect_nn_label_agreement(bd_model, raw_loader)
    knn_r = print_detection_report("KNN Agreement", knn_flagged, bd_indices, n_total)

    log.record_nested("detectors", "loss_outlier", loss_r)
    log.record_nested("detectors", "spectral_signatures", spec_r)
    log.record_nested("detectors", "activation_clustering", ac_r)
    log.record_nested("detectors", "knn_label_agreement", knn_r)
    plot_detection_comparison({"Loss": loss_r, "Spectral": spec_r,
                               "Clustering": ac_r, "KNN": knn_r}, "demo_detection_comparison.png")

    # Phase 4: Clean & Retrain
    banner("PHASE 4: Clean & Retrain")
    combined = spec_flagged | knn_flagged
    print(f"\n  Flagged {len(combined)} samples (Spectral ∪ KNN)")
    print_detection_report("Combined", combined, bd_indices, n_total)
    log.record("n_flagged", len(combined))

    cleaned_model = build_resnet18(compile_model=True)
    cleaned_model, ch = train_model(
        cleaned_model,
        build_clean_subset_loader(combined, poison_indices=bd_indices,
                                  poison_fn=bd_fn),
    )
    _, cleaned_acc = evaluate(cleaned_model, test_loader)
    asr_after = evaluate_backdoor_asr(cleaned_model, test_loader, apply_trigger_to_tensor)
    log.record("cleaned_training_history", ch)
    log.record("cleaned_test_accuracy", cleaned_acc)
    log.record("asr_after_cleaning", asr_after)

    # Summary
    banner("RESULTS")
    print(f"""
  ┌─────────────────────────────┬────────────┬────────────┐
  │ Metric                      │  Poisoned  │  Cleaned   │
  ├─────────────────────────────┼────────────┼────────────┤
  │ Test Accuracy (%)           │  {bd_acc:>8.2f}  │  {cleaned_acc:>8.2f}  │
  │ Backdoor ASR (%)            │  {asr_before:>8.2f}  │  {asr_after:>8.2f}  │
  │ Samples removed             │       —    │  {len(combined):>8d}  │
  └─────────────────────────────┴────────────┴────────────┘""")

    status = "✅ Backdoor neutralised" if asr_after < 5.0 else "⚠ Partially mitigated"
    print(f"\n  {status}")
    plot_model_comparison(bd_acc, bd_acc, cleaned_acc, asr_before, asr_after, "demo_model_comparison.png")
    log.save()
    print(f"\n  Plots: {config.PLOT_DIR}")
    print(f"  Logs:  {config.LOG_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    run()
