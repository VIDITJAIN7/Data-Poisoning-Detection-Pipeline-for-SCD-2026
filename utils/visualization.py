"""Visualization helpers for attacks and detection results."""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def save(fig, name):
    path = os.path.join(config.PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {path}")


def plot_trigger_examples(clean_images, poisoned_images, n=5,
                          filename="trigger_examples.png"):
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    for i in range(n):
        axes[0, i].imshow(clean_images[i]); axes[0, i].set_title("Clean", fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(poisoned_images[i]); axes[1, i].set_title("Poisoned", fontsize=9); axes[1, i].axis("off")
    fig.suptitle("Backdoor Trigger Examples", fontsize=13, y=1.02)
    fig.tight_layout(); save(fig, filename)


def plot_loss_distribution(losses, poison_mask, threshold,
                           filename="loss_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(losses[~poison_mask], bins=100, alpha=0.6, label="Clean", color="#2196F3")
    ax.hist(losses[poison_mask], bins=100, alpha=0.6, label="Poisoned", color="#F44336")
    ax.axvline(threshold, color="k", ls="--", lw=1.5, label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Loss"); ax.set_ylabel("Count"); ax.set_title("Per-Sample Loss Distribution")
    ax.legend(); fig.tight_layout(); save(fig, filename)


def plot_spectral_scores(scores, poison_mask, threshold,
                         filename="spectral_scores.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(scores))
    ax.scatter(idx[~poison_mask], scores[~poison_mask], s=1, alpha=0.3, label="Clean", color="#2196F3")
    ax.scatter(idx[poison_mask], scores[poison_mask], s=3, alpha=0.6, label="Poisoned", color="#F44336")
    ax.axhline(threshold, color="k", ls="--", lw=1.5, label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Sample index"); ax.set_ylabel("Spectral score")
    ax.set_title("Spectral Signature Outlier Scores")
    ax.legend(); fig.tight_layout(); save(fig, filename)


def plot_activation_clusters(features_2d, cluster_labels, poison_mask, class_id,
                             filename="activation_clusters.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in np.unique(cluster_labels):
        m = cluster_labels == c
        axes[0].scatter(features_2d[m, 0], features_2d[m, 1], s=4, alpha=0.5, label=f"Cluster {c}")
    axes[0].set_title(f"Cluster assignment (class {class_id})"); axes[0].legend(fontsize=8)

    axes[1].scatter(features_2d[~poison_mask, 0], features_2d[~poison_mask, 1], s=4, alpha=0.4, label="Clean", color="#2196F3")
    axes[1].scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1], s=6, alpha=0.7, label="Poisoned", color="#F44336")
    axes[1].set_title(f"Ground truth (class {class_id})"); axes[1].legend(fontsize=8)

    fig.suptitle("Activation Clustering", fontsize=13)
    fig.tight_layout(); save(fig, filename)


def plot_detection_comparison(results, filename="detection_comparison.png"):
    names = list(results.keys())
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.25
    colours = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(names)), 5))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        ax.bar(x + i * width, vals, width, label=m.capitalize(), color=colours[i])

    ax.set_xticks(x + width); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Detection Method Comparison")
    ax.legend(); fig.tight_layout(); save(fig, filename)


def plot_model_comparison(clean_acc, poisoned_acc, cleaned_acc,
                          asr_before, asr_after, filename="model_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = axes[0].bar(["Clean\nBaseline", "Poisoned\nModel", "Cleaned\nModel"],
                        [clean_acc, poisoned_acc, cleaned_acc],
                        color=["#4CAF50", "#F44336", "#2196F3"])
    axes[0].set_ylim(0, 105); axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Test Accuracy")
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height()+1,
                     f"{b.get_height():.1f}%", ha="center", fontsize=9)

    bars2 = axes[1].bar(["Poisoned\nModel", "Cleaned\nModel"],
                        [asr_before, asr_after], color=["#F44336", "#4CAF50"])
    axes[1].set_ylim(0, 105); axes[1].set_ylabel("Attack Success Rate (%)")
    axes[1].set_title("Backdoor ASR")
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height()+1,
                     f"{b.get_height():.1f}%", ha="center", fontsize=9)

    fig.suptitle("Model Quality: Before vs After Cleaning", fontsize=13)
    fig.tight_layout(); save(fig, filename)
