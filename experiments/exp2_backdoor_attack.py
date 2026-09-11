"""Experiment 2 — Backdoor Attack: inject trigger patch and measure ASR."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torchvision import datasets

import config
from attacks.backdoor import prepare_backdoor_attack, apply_trigger_patch, apply_trigger_to_tensor
from models.resnet import build_resnet18
from utils.data_loader import build_poisoned_loader, get_clean_loaders, CIFAR10_CLASSES
from utils.train_eval import train_model, evaluate, evaluate_backdoor_asr
from utils.visualization import plot_trigger_examples
from utils.logger import ExperimentLogger


def set_seed():
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run():
    set_seed()
    log = ExperimentLogger("exp2_backdoor_attack")
    print("=" * 60)
    print("  EXPERIMENT 2: Backdoor Attack")
    print("=" * 60)

    # Visualise trigger
    print("\n[1/4] Generating trigger examples …")
    raw_ds = datasets.CIFAR10(config.DATA_DIR, train=True, download=True)
    clean_imgs = np.stack([np.array(raw_ds[i][0]) for i in range(5)])
    poisoned_imgs = np.stack([np.array(apply_trigger_patch(raw_ds[i][0])) for i in range(5)])
    plot_trigger_examples(clean_imgs, poisoned_imgs, filename="backdoor_trigger_examples.png")

    # Train
    print("\n[2/4] Training on poisoned data …")
    poison_indices, poison_fn = prepare_backdoor_attack()
    log.record("n_poisoned_samples", len(poison_indices))
    log.record("poison_rate", config.POISON_RATE)

    model = build_resnet18(compile_model=True)
    model, history = train_model(model, build_poisoned_loader(poison_indices, poison_fn))
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "backdoor.pt"))
    log.record("training_history", history)

    # Evaluate
    print("\n[3/4] Standard test accuracy …")
    _, test_loader = get_clean_loaders()
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"  Test Loss: {test_loss:.4f}  Acc: {test_acc:.2f}% (looks normal)")
    log.record("test_loss", test_loss)
    log.record("test_accuracy", test_acc)

    print("\n[4/4] Attack Success Rate …")
    asr = evaluate_backdoor_asr(model, test_loader, trigger_fn=apply_trigger_to_tensor)
    print(f"  ASR: {asr:.2f}% → triggered images classified as "
          f"'{CIFAR10_CLASSES[config.BACKDOOR_TARGET_LABEL]}'")
    log.record("attack_success_rate", asr)
    log.save()
    print("  Done.\n")


if __name__ == "__main__":
    run()
