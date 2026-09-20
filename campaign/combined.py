"""Stack and ensemble the two selected campaign solutions.

The layered campaign treated the threat models independently:

* Solution A (label-flip recovery): Cleanlab out-of-fold issue detection,
  then replace or remove flagged labels.
* Solution B (backdoor mitigation): Fine-Pruning and/or trusted FT-SAM on
  a poisoned checkpoint.

This module joins them so a single pipeline can clean mixed poisoning and,
at inference, average logits from the two selected checkpoints.
"""

from copy import deepcopy
from typing import Dict, Iterable, Optional, Set

import torch
import torch.nn as nn

from models.resnet import build_resnet18
import config


def stack_cleaning_spec(label_spec: Dict, backdoor_flagged: Iterable[int]) -> Dict:
    """Union Cleanlab removals with backdoor-flagged IDs; keep remaining corrections.

    Label corrections are dropped for any ID that is also removed as a
    suspected backdoor sample so the two policies cannot fight each other.
    """
    corrections = {int(i): int(y) for i, y in dict(label_spec.get("corrections") or {}).items()}
    remove: Set[int] = {int(i) for i in (label_spec.get("remove") or ())}
    remove |= {int(i) for i in backdoor_flagged}
    corrections = {i: y for i, y in corrections.items() if i not in remove}
    return {"corrections": corrections, "remove": remove}


@torch.no_grad()
def ensemble_logits(model_a: nn.Module, model_b: nn.Module, images: torch.Tensor,
                    weight_a: float = 0.5) -> torch.Tensor:
    """Soft-ensemble Solution A and Solution B by averaging logits."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    was_a_training, was_b_training = model_a.training, model_b.training
    model_a.eval()
    model_b.eval()
    logits = weight_a * model_a(images) + (1.0 - weight_a) * model_b(images)
    model_a.train(was_a_training)
    model_b.train(was_b_training)
    return logits


@torch.no_grad()
def ensemble_predict(model_a: nn.Module, model_b: nn.Module, images: torch.Tensor,
                     weight_a: float = 0.5) -> torch.Tensor:
    return ensemble_logits(model_a, model_b, images, weight_a).argmax(dim=1)


def load_selected_pair(label_ckpt: str, backdoor_ckpt: str, device=None):
    """Load the two selected campaign checkpoints for inference-time ensembling."""
    device = device or config.DEVICE
    label_model = build_resnet18(compile_model=False)
    backdoor_model = build_resnet18(compile_model=False)
    label_state = torch.load(label_ckpt, map_location="cpu")
    backdoor_state = torch.load(backdoor_ckpt, map_location="cpu")
    label_model.load_state_dict(_unwrap_compiled(label_state))
    backdoor_model.load_state_dict(_unwrap_compiled(backdoor_state))
    return label_model.to(device).eval(), backdoor_model.to(device).eval()


def apply_stacked_mitigation(poisoned_model: nn.Module, trusted_ids, base,
                             prune_fraction: float = 0.10, rho: float = 0.05,
                             epochs: Optional[int] = None, seed: int = 42):
    """Run Solution B (fine-prune + FT-SAM) after Solution A has cleaned labels.

    Intended for a checkpoint already trained on Cleanlab-corrected data.
    """
    from campaign.defenses import attach_pruning_mask, enforce_pruning, train_ft_sam, _loader
    from campaign.data import TrustedDataset

    epochs = config.EPOCHS if epochs is None else epochs
    model = build_resnet18(compile_model=False)
    model.load_state_dict(deepcopy(poisoned_model.state_dict()))
    trusted_eval = _loader(TrustedDataset(base, trusted_ids), config.EVAL_BATCH_SIZE, False, seed)
    trusted_train = _loader(TrustedDataset(base, trusted_ids), config.BATCH_SIZE, True, seed)
    mask, handle = attach_pruning_mask(model, trusted_eval, prune_fraction)
    model, history = train_ft_sam(model, trusted_train, epochs, seed, rho=rho, lr=0.01)
    enforce_pruning(model, mask)
    return model, {"mask_zeros": int((mask == 0).sum().item()), "history": history, "handle": handle}


def _unwrap_compiled(state):
    if state and all(key.startswith("_orig_mod.") for key in state):
        return {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    return state
