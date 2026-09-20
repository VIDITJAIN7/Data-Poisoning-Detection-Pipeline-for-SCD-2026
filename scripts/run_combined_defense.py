#!/usr/bin/env python3
"""Ensemble the selected label-flip and backdoor checkpoints.

Usage (after a layered campaign has written checkpoints):

    python scripts/run_combined_defense.py \\
        --label-ckpt outputs/layered_campaign/<label-run>/<selected>.pt \\
        --backdoor-ckpt outputs/layered_campaign/<backdoor-run>/<selected>.pt

Without checkpoints this prints the stacked cleaning spec that joins
Solution A (Cleanlab) with Solution B (fine-prune / FT-SAM).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from campaign.combined import ensemble_logits, load_selected_pair, stack_cleaning_spec
from campaign.data import load_test_base, clean_eval_dataset
from campaign.metrics import evaluate_asr_and_triggered_true_accuracy
from utils.train_eval import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-ckpt")
    p.add_argument("--backdoor-ckpt")
    p.add_argument("--weight-a", type=float, default=0.5,
                   help="Logit weight for the label-recovery model (Solution A)")
    p.add_argument("--label-spec", help="JSON with Cleanlab corrections/remove")
    p.add_argument("--backdoor-flags", help="JSON list of backdoor-flagged IDs")
    args = p.parse_args()

    if args.label_spec or args.backdoor_flags:
        label_spec = json.loads(Path(args.label_spec).read_text()) if args.label_spec else {"corrections": {}, "remove": []}
        flags = json.loads(Path(args.backdoor_flags).read_text()) if args.backdoor_flags else []
        stacked = stack_cleaning_spec(label_spec, flags)
        print(json.dumps({
            "n_corrections": len(stacked["corrections"]),
            "n_removed": len(stacked["remove"]),
            "stacked": {k: (sorted(v) if isinstance(v, set) else v) for k, v in stacked.items()},
        }, indent=2))

    if not (args.label_ckpt and args.backdoor_ckpt):
        print("Stacked spec ready. Pass --label-ckpt and --backdoor-ckpt to evaluate the ensemble.")
        return

    label_model, backdoor_model = load_selected_pair(args.label_ckpt, args.backdoor_ckpt)
    test = clean_eval_dataset(load_test_base())
    loader = DataLoader(test, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)

    class Ensemble(torch.nn.Module):
        def forward(self, images):
            return ensemble_logits(label_model, backdoor_model, images, args.weight_a)

    wrapped = Ensemble().to(config.DEVICE)
    loss, acc = evaluate(wrapped, loader)
    asr = evaluate_asr_and_triggered_true_accuracy(wrapped, loader)
    print(json.dumps({"ensemble_accuracy": acc, "ensemble_loss": loss, **asr,
                      "weight_a": args.weight_a}, indent=2))


if __name__ == "__main__":
    main()
