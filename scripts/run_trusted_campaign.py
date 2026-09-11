#!/usr/bin/env python3
"""Run the trusted-data campaign with immutable manifests.

Smoke mode uses two epochs and a single seed. Full mode keeps the configured
30-epoch budget; development and final selections are recorded separately.
"""
import argparse, hashlib, json, os, platform, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from campaign.data import (load_train_base, load_test_base, make_stratified_split,
    clean_eval_dataset, ImmutableObservedDataset, TrustedDataset, poison_labels)
from campaign.defenses import (_loader, oof_cleanlab, label_candidates, attach_pruning_mask,
    enforce_pruning, train_ft_sam)
from campaign.metrics import evaluate_model, evaluate_asr_and_triggered_true_accuracy
from models.resnet import build_resnet18
from utils.train_eval import train_model
from attacks.label_flip import select_flip_indices
from attacks.backdoor import select_backdoor_indices


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha_ids(ids):
    return hashlib.sha256(np.asarray(sorted(ids), dtype=np.int64).tobytes()).hexdigest()


def train_eval(base, train_ids, labels, poisoned, kind, trusted_ids, seed, epochs, out, name, remove=None, corrections=None, initial_state=None):
    # Reset initialization and loader RNGs for every paired condition.  A
    # stable digest avoids Python's process-randomized hash().
    # Identical initialization makes paired differences attributable to the
    # data or defense rather than a new random model.
    seed_everything(seed)
    remove = set(remove or ())
    corrections = dict(corrections or {})
    effective = [i for i in train_ids if i not in remove]
    effective_labels = dict(labels); effective_labels.update(corrections)
    ds = ImmutableObservedDataset(base, effective, effective_labels, set(poisoned) - remove, kind, True)
    trusted = TrustedDataset(base, trusted_ids)
    model = build_resnet18(compile_model=False)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    model, history = train_model(model, _loader(ConcatDataset([ds, trusted]), config.BATCH_SIZE, True, seed), epochs=epochs)
    torch.save(model.state_dict(), out / f"{name}.pt")
    return model, history


def run(args):
    seed_everything(args.seed)
    # Include attack in the directory key: label and backdoor campaigns at
    # the same seed/rate must never overwrite one another.
    out = Path(args.output) / (f"{args.attack}-seed-{args.seed}-rate-{args.rate:g}".replace(".", ""))
    out.mkdir(parents=True, exist_ok=True)
    base = load_train_base(); test = clean_eval_dataset(load_test_base())
    split = make_stratified_split(base, args.split_seed)
    labels = {i: int(base.targets[i]) for i in split.attack_ids}
    n_poison = int(len(split.attack_ids) * args.rate)
    if args.attack == "label":
        eligible = [i for i in split.attack_ids if labels[i] == config.LABEL_FLIP_SOURCE]
        poison = set(np.random.default_rng(args.seed).choice(eligible, n_poison, replace=False).tolist())
    else:
        eligible = [i for i in split.attack_ids if labels[i] != config.BACKDOOR_TARGET_LABEL]
        poison = set(np.random.default_rng(args.seed).choice(eligible, n_poison, replace=False).tolist())
    observed = dict(labels)
    if args.attack == "label":
        observed.update({i: config.LABEL_FLIP_TARGET for i in poison})
    config_data = {"seed": args.seed, "split_seed": args.split_seed, "rate": args.rate,
                   "attack": args.attack, "epochs": args.epochs, "poison_count": len(poison),
                   "attack_pool": len(split.attack_ids), "trusted": len(split.trusted_ids),
                   "dev": len(split.dev_ids), "split_hash": sha_ids(split.all_ids),
                   "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "torch": torch.__version__, "python": platform.python_version()}
    (out / "manifest.json").write_text(json.dumps({"status":"running", "config":config_data}, indent=2))
    results = {"config": config_data, "poison_ids_hash": sha_ids(poison)}
    base_model, base_hist = train_eval(base, split.attack_ids, labels, set(), None, split.trusted_ids, args.seed, args.epochs, out, "baseline")
    poison_model, poison_hist = train_eval(base, split.attack_ids, observed, poison, args.attack, split.trusted_ids, args.seed, args.epochs, out, "poisoned")
    results["baseline"] = evaluate_model(base_model, DataLoader(test, batch_size=config.EVAL_BATCH_SIZE, shuffle=False))
    results["poisoned"] = evaluate_model(poison_model, DataLoader(test, batch_size=config.EVAL_BATCH_SIZE, shuffle=False))
    test_loader = DataLoader(test, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
    if args.attack == "backdoor":
        results["poisoned"].update(evaluate_asr_and_triggered_true_accuracy(poison_model, test_loader))
    if args.attack == "label":
        ids, ys, probs, issue_sets, folds = oof_cleanlab(base, split.attack_ids, observed, split.trusted_ids, args.seed, args.epochs)
        candidates = label_candidates(ids, ys, probs, issue_sets)
        candidate_scores = {}
        candidate_test = {}
        dev_ds = ImmutableObservedDataset(base, split.dev_ids, {i:int(base.targets[i]) for i in split.dev_ids}, set(), None, False)
        dev_loader = DataLoader(dev_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
        for name, spec in candidates.items():
            model, hist = train_eval(base, split.attack_ids, observed, poison, "label", split.trusted_ids, args.seed, args.epochs, out, name, spec["remove"], spec["corrections"])
            candidate_scores[name] = evaluate_model(model, dev_loader)
            candidate_test[name] = evaluate_model(model, test_loader)
        selected = max(candidate_scores, key=lambda n: (candidate_scores[n]["accuracy"], -len(candidates[n]["corrections"])))
        results.update({"oof_issue_count": {k: len(v) for k, v in issue_sets.items()}, "folds": folds, "candidate_scores": candidate_scores,
                        "selected_test": candidate_test[selected], "selected_defense": selected,
                        "selection_source": "development split"})
    else:
        dev_ds = ImmutableObservedDataset(base, split.dev_ids,
            {i: int(base.targets[i]) for i in split.dev_ids}, set(), None, False)
        trusted_loader = _loader(TrustedDataset(base, split.trusted_ids), config.EVAL_BATCH_SIZE, False, args.seed)
        candidate_scores = {}; candidate_test = {}
        dev_loader = DataLoader(dev_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
        for fraction in (0.05, 0.10, 0.20):
            model = build_resnet18(compile_model=False); model.load_state_dict(poison_model.state_dict())
            mask, handle = attach_pruning_mask(model, trusted_loader, fraction)
            model, hist = train_model(model, _loader(TrustedDataset(base, split.trusted_ids), config.BATCH_SIZE, True, args.seed), epochs=args.epochs, lr=0.01)
            enforce_pruning(model, mask)
            torch.save(model.state_dict(), out / f"fine_prune_{fraction:.2f}.pt")
            candidate_scores[f"fine_prune_{fraction:.2f}"] = evaluate_model(model, dev_loader) | evaluate_asr_and_triggered_true_accuracy(model, dev_loader)
            candidate_test[f"fine_prune_{fraction:.2f}"] = evaluate_model(model, test_loader) | evaluate_asr_and_triggered_true_accuracy(model, test_loader)
            handle.remove()
        ft, _ = train_eval(base, split.trusted_ids, {i:int(base.targets[i]) for i in split.trusted_ids}, set(), None, (), args.seed, args.epochs, out, "trusted_finetune", initial_state=poison_model.state_dict())
        candidate_scores["trusted_finetune"] = evaluate_model(ft, dev_loader) | evaluate_asr_and_triggered_true_accuracy(ft, dev_loader)
        candidate_test["trusted_finetune"] = evaluate_model(ft, test_loader) | evaluate_asr_and_triggered_true_accuracy(ft, test_loader)
        for rho in (0.05, 0.10):
            model = build_resnet18(compile_model=False); model.load_state_dict(poison_model.state_dict())
            model, _ = train_ft_sam(model, _loader(TrustedDataset(base, split.trusted_ids), config.BATCH_SIZE, True, args.seed), args.epochs, args.seed, rho=rho, lr=0.01)
            name = f"ft_sam_{rho:.2f}"; torch.save(model.state_dict(), out / f"{name}.pt")
            candidate_scores[name] = evaluate_model(model, dev_loader) | evaluate_asr_and_triggered_true_accuracy(model, dev_loader)
            candidate_test[name] = evaluate_model(model, test_loader) | evaluate_asr_and_triggered_true_accuracy(model, test_loader)
        baseline_dev = evaluate_model(base_model, dev_loader)["accuracy"]
        eligible = [n for n, m in candidate_scores.items() if abs(m["accuracy"] - baseline_dev) <= 1.0]
        selected = min(eligible or list(candidate_scores), key=lambda n: (candidate_scores[n].get("asr") if candidate_scores[n].get("asr") is not None else 1e9, abs(candidate_scores[n]["accuracy"] - baseline_dev)))
        results.update({"candidate_scores": candidate_scores, "selected_test": candidate_test[selected],
                        "selected_defense": selected, "selection_source": "development split"})
    results["status"] = "complete"
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "manifest.json").write_text(json.dumps({"status":"complete", "config":config_data}, indent=2))
    print(json.dumps({"output": str(out), "selected": results.get("selected_defense"), "poison_count": len(poison)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=("label", "backdoor"), required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--rate", type=float, default=.05); p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--output", default="outputs/trusted_campaign")
    run(p.parse_args())
