"""Experiment 4 — Cleaning & Retraining: remove detected poison, retrain, verify."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import config
from attacks.backdoor import prepare_backdoor_attack, apply_trigger_to_tensor
from models.resnet import build_resnet18
from utils.data_loader import (build_poisoned_loader, build_poisoned_raw_loader,
                                build_clean_subset_loader, get_clean_loaders,
                                PoisonedDataset)
from utils.train_eval import train_model, evaluate, evaluate_backdoor_asr
from utils.metrics import print_detection_report
from utils.logger import ExperimentLogger
from utils.visualization import plot_model_comparison
from detection.spectral_signatures import detect_spectral_signatures
from detection.nn_label_agreement import detect_nn_label_agreement
from detection.trigger_signature import detect_fixed_trigger


def set_seed():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run():
    set_seed()
    log = ExperimentLogger("exp4_cleaning_and_retrain")
    _, test_loader = get_clean_loaders()
    n_total = 50_000
    print("=" * 60)
    print("  EXPERIMENT 4: Cleaning & Retraining")
    print("=" * 60)

    # Clean baseline
    print("\n[1/5] Clean baseline …")
    baseline_path = os.path.join(config.MODEL_DIR, "clean_baseline.pt")
    clean_model = build_resnet18(compile_model=False)
    if os.path.exists(baseline_path):
        # Experiment 1 may have saved a torch.compile() state dict. Compiled
        # checkpoints prefix every parameter with ``_orig_mod.``; remove that
        # wrapper prefix before loading into the plain ResNet used here.
        checkpoint = torch.load(baseline_path, map_location="cpu")
        if checkpoint and all(key.startswith("_orig_mod.") for key in checkpoint):
            checkpoint = {
                key.removeprefix("_orig_mod."): value
                for key, value in checkpoint.items()
            }
        clean_model.load_state_dict(checkpoint)
        print("  Loaded cached baseline.")
    else:
        train_loader, _ = get_clean_loaders()
        clean_model, _ = train_model(clean_model, train_loader)
        torch.save(clean_model.state_dict(), baseline_path)
    _, clean_acc = evaluate(clean_model, test_loader)
    print(f"  Baseline accuracy: {clean_acc:.2f}%")
    log.record("clean_test_accuracy", clean_acc)

    # Poisoned model
    print("\n[2/5] Training poisoned model …")
    poison_indices, poison_fn = prepare_backdoor_attack()
    poisoned_model = build_resnet18(compile_model=True)
    poisoned_model, ph = train_model(poisoned_model, build_poisoned_loader(poison_indices, poison_fn))
    _, poisoned_acc = evaluate(poisoned_model, test_loader)
    asr_before = evaluate_backdoor_asr(poisoned_model, test_loader, apply_trigger_to_tensor)
    print(f"  Poisoned accuracy: {poisoned_acc:.2f}%  |  ASR: {asr_before:.2f}%")
    log.record("poisoned_training_history", ph)
    log.record("poisoned_test_accuracy", poisoned_acc)
    log.record("asr_before_cleaning", asr_before)

    # Detect
    print("\n[3/5] Detection (model-based comparison + trigger signature) …")
    raw_loader = build_poisoned_raw_loader(poison_indices, poison_fn)
    spec_flagged, _, _, _ = detect_spectral_signatures(poisoned_model, raw_loader)
    knn_flagged, _, _ = detect_nn_label_agreement(poisoned_model, raw_loader)
    spec_report = print_detection_report("Spectral Signatures", spec_flagged, poison_indices, n_total)
    knn_report = print_detection_report("KNN Agreement", knn_flagged, poison_indices, n_total)
    from torchvision import datasets
    trigger_dataset = PoisonedDataset(
        datasets.CIFAR10(config.DATA_DIR, train=True, download=True),
        poison_indices, poison_fn, transform=None,
    )
    trigger_flagged = detect_fixed_trigger(trigger_dataset)
    trigger_report = print_detection_report(
        "Known Trigger Signature", trigger_flagged, poison_indices, n_total
    )
    eligible = [("spectral_signatures", spec_report, spec_flagged),
                ("knn_label_agreement", knn_report, knn_flagged),
                ("known_trigger_signature", trigger_report, trigger_flagged)]
    eligible = [item for item in eligible if item[1]["fpr"] <= 0.05]
    selected_name, report, combined = max(
        eligible, key=lambda item: item[1]["f1"]
    ) if eligible else min(
        [("spectral_signatures", spec_report, spec_flagged),
         ("knn_label_agreement", knn_report, knn_flagged),
         ("known_trigger_signature", trigger_report, trigger_flagged)],
        key=lambda item: item[1]["fpr"]
    )
    print(f"  Selected defense: {selected_name} (FPR={report['fpr']:.3%})")
    log.record("detection_report", report)
    log.record("n_flagged", len(combined))

    # Retrain
    print("[4/5] Retraining on cleaned data …")
    cleaned_model = build_resnet18(compile_model=True)
    cleaned_model, ch = train_model(
        cleaned_model,
        build_clean_subset_loader(combined, poison_indices=poison_indices,
                                  poison_fn=poison_fn),
    )
    log.record("comparison_union_flagged", len(spec_flagged | knn_flagged))
    log.record("selected_defense", selected_name)
    torch.save(cleaned_model.state_dict(), os.path.join(config.MODEL_DIR, "cleaned.pt"))
    _, cleaned_acc = evaluate(cleaned_model, test_loader)
    asr_after = evaluate_backdoor_asr(cleaned_model, test_loader, apply_trigger_to_tensor)
    print(f"  Cleaned accuracy: {cleaned_acc:.2f}%  |  ASR: {asr_after:.2f}%")
    log.record("cleaned_training_history", ch)
    log.record("cleaned_test_accuracy", cleaned_acc)
    log.record("asr_after_cleaning", asr_after)

    # Summary
    print(f"\n[5/5] Summary:")
    print(f"  {'Metric':<25s} {'Poisoned':>10s} {'Cleaned':>10s}")
    print("  " + "─" * 47)
    print(f"  {'Test Accuracy (%)':<25s} {poisoned_acc:>10.2f} {cleaned_acc:>10.2f}")
    print(f"  {'Backdoor ASR (%)':<25s} {asr_before:>10.2f} {asr_after:>10.2f}")
    plot_model_comparison(clean_acc, poisoned_acc, cleaned_acc, asr_before, asr_after)
    log.save()
    print("  Done.\n")


if __name__ == "__main__":
    run()
