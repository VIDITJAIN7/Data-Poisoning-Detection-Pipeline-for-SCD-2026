import torch
import torch.nn as nn

from campaign.combined import ensemble_logits, ensemble_predict, stack_cleaning_spec


class Tiny(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias)

    def forward(self, x):
        return self.linear(x)


def test_stack_drops_corrections_that_conflict_with_backdoor_flags():
    spec = {"corrections": {1: 9, 2: 9, 3: 0}, "remove": {4}}
    stacked = stack_cleaning_spec(spec, backdoor_flagged={2, 5})
    assert stacked["remove"] == {2, 4, 5}
    assert stacked["corrections"] == {1: 9, 3: 0}


def test_ensemble_averages_logits():
    a = Tiny(bias=2.0)
    b = Tiny(bias=0.0)
    images = torch.zeros(2, 4)
    logits = ensemble_logits(a, b, images, weight_a=0.5)
    assert torch.allclose(logits, torch.ones(2, 3))
    preds = ensemble_predict(a, b, images)
    assert preds.shape == (2,)


def test_ensemble_weight_a_one_matches_model_a():
    a = Tiny(bias=4.0)
    b = Tiny(bias=-4.0)
    images = torch.zeros(1, 4)
    logits = ensemble_logits(a, b, images, weight_a=1.0)
    assert torch.allclose(logits, a(images))
