"""Experiment 0 — Clean Baseline: train ResNet-18 on unmodified CIFAR-10."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import config
from models.resnet import build_resnet18
from utils.data_loader import get_clean_loaders, get_test_transform, CleanIndexDataset, CIFAR10_CLASSES
from utils.train_eval import train_model, evaluate
from utils.logger import ExperimentLogger


def set_seed():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run():
    set_seed()
    # This experiment intentionally uses no attack samples.  Override the
    # global attack default in the log so its metadata reports the true rate.
    log = ExperimentLogger("exp0_clean_baseline", poison_rate=0.0)
    print("=" * 60)
    print("  EXPERIMENT 0: Clean Baseline")
    print("=" * 60)

    print("\n[1/2] Training on clean CIFAR-10 …")
    train_loader, test_loader = get_clean_loaders()
    model = build_resnet18(compile_model=True)
    model, history = train_model(model, train_loader)
    log.record("training_history", history)

    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "clean_baseline.pt"))

    print("\n[2/2] Evaluating …")
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"  Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.2f}%")
    log.record("test_loss", test_loss)
    log.record("test_accuracy", test_acc)

    per_class = {}
    test_ds = datasets.CIFAR10(config.DATA_DIR, train=False, download=True)
    for cls_id in range(10):
        idx = [i for i, (_, l) in enumerate(test_ds) if l == cls_id]
        cls_loader = DataLoader(
            CleanIndexDataset(Subset(test_ds, idx), transform=get_test_transform()),
            batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        )
        _, acc = evaluate(model, cls_loader)
        per_class[CIFAR10_CLASSES[cls_id]] = round(acc, 2)

    log.record("per_class_accuracy", per_class)
    print(f"\n  {'Class':<12s} {'Accuracy':>8s}")
    print("  " + "─" * 22)
    for name, acc in per_class.items():
        print(f"  {name:<12s} {acc:>7.1f}%")

    log.save()
    print("  Done.\n")


if __name__ == "__main__":
    run()
