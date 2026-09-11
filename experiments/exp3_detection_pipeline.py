"""Experiment 3 — Detection Pipeline: run 4 detectors and compare metrics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import config
from attacks.backdoor import prepare_backdoor_attack
from models.resnet import build_resnet18
from utils.data_loader import build_poisoned_loader, build_poisoned_raw_loader
from utils.train_eval import train_model
from utils.metrics import print_detection_report
from utils.logger import ExperimentLogger
from utils.visualization import plot_loss_distribution, plot_spectral_scores, plot_activation_clusters, plot_detection_comparison
from detection.loss_outlier import detect_loss_outliers
from detection.spectral_signatures import detect_spectral_signatures
from detection.activation_clustering import detect_activation_clusters
from detection.nn_label_agreement import detect_nn_label_agreement


def set_seed():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run():
    set_seed()
    log = ExperimentLogger("exp3_detection_pipeline")
    n_total = 50_000
    print("=" * 60)
    print("  EXPERIMENT 3: Detection Pipeline")
    print("=" * 60)

    # Prepare and train
    print("\n[1/6] Preparing poisoned data & training …")
    poison_indices, poison_fn = prepare_backdoor_attack()
    model = build_resnet18(compile_model=True)
    model, history = train_model(model, build_poisoned_loader(poison_indices, poison_fn))
    log.record("training_history", history)
    log.record("n_poisoned_samples", len(poison_indices))

    raw_loader = build_poisoned_raw_loader(poison_indices, poison_fn)

    # 1. Loss-based
    print("\n[2/6] Loss-based outlier detection …")
    loss_flagged, losses, loss_thr, loss_idx = detect_loss_outliers(model, raw_loader)
    pmask = np.array([int(i) in poison_indices for i in loss_idx])
    plot_loss_distribution(losses, pmask, loss_thr)
    log.record_nested("detectors", "loss_outlier",
                      print_detection_report("Loss Outlier", loss_flagged, poison_indices, n_total))

    # 2. Spectral
    print("[3/6] Spectral-signature detection …")
    spec_flagged, scores, spec_thr, spec_idx = detect_spectral_signatures(model, raw_loader)
    pmask_s = np.array([int(i) in poison_indices for i in spec_idx])
    plot_spectral_scores(scores, pmask_s, spec_thr)
    log.record_nested("detectors", "spectral_signatures",
                      print_detection_report("Spectral Signatures", spec_flagged, poison_indices, n_total))

    # 3. Activation clustering
    print("[4/6] Activation clustering …")
    ac_flagged, ac_debug = detect_activation_clusters(model, raw_loader)
    tc = config.BACKDOOR_TARGET_LABEL
    if tc in ac_debug:
        info = ac_debug[tc]
        plot_activation_clusters(info["features_2d"], info["cluster_labels"],
                                np.array([int(i) in poison_indices for i in info["indices"]]), tc)
    log.record_nested("detectors", "activation_clustering",
                      print_detection_report("Activation Clustering", ac_flagged, poison_indices, n_total))

    # 4. KNN
    print("[5/6] KNN label-agreement …")
    knn_flagged, _, _ = detect_nn_label_agreement(model, raw_loader)
    log.record_nested("detectors", "knn_label_agreement",
                      print_detection_report("KNN Label Agreement", knn_flagged, poison_indices, n_total))

    # Summary
    print("[6/6] Comparison …")
    all_r = {
        "Loss Outlier": log.data["results"]["detectors"]["loss_outlier"],
        "Spectral Sig.": log.data["results"]["detectors"]["spectral_signatures"],
        "Act. Clustering": log.data["results"]["detectors"]["activation_clustering"],
        "KNN Agreement": log.data["results"]["detectors"]["knn_label_agreement"],
    }
    plot_detection_comparison(all_r)
    print(f"\n  {'Method':<20s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'FPR':>8s}")
    print("  " + "─" * 44)
    for name, r in all_r.items():
        print(f"  {name:<20s} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['fpr']:>8.5f}")

    log.save()
    print("  Done.\n")


if __name__ == "__main__":
    run()
