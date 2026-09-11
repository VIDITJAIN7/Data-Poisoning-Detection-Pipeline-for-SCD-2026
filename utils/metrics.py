"""Detection and model-quality metrics."""

from typing import Dict, Set


def detection_metrics(flagged: Set[int], poison_gt: Set[int],
                      total_samples: int) -> Dict[str, float]:
    """Compute TP, FP, FN, precision, recall, F1, and FPR."""
    tp = len(flagged & poison_gt)
    fp = len(flagged - poison_gt)
    fn = len(poison_gt - flagged)
    tn = total_samples - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def print_detection_report(name, flagged, poison_gt, total_samples):
    """Pretty-print detection metrics and return them."""
    m = detection_metrics(flagged, poison_gt, total_samples)
    print(f"\n{'═'*50}")
    print(f"  Detection Report: {name}")
    print(f"{'═'*50}")
    print(f"  True positives  : {m['true_pos']}")
    print(f"  False positives : {m['false_pos']}")
    print(f"  False negatives : {m['false_neg']}")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  Recall          : {m['recall']:.4f}")
    print(f"  F1 Score        : {m['f1']:.4f}")
    print(f"  FPR             : {m['fpr']:.6f}")
    print(f"{'═'*50}\n")
    return m
