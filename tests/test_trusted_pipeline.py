import numpy as np
import torch
from PIL import Image

from campaign.data import make_stratified_split, ImmutableObservedDataset


class Fake:
    targets = [c for c in range(10) for _ in range(5000)]
    def __getitem__(self, i):
        return Image.new("RGB", (32, 32)), self.targets[i]

def test_split_is_stratified_and_disjoint():
    s = make_stratified_split(Fake(), 42)
    assert (len(s.attack_ids), len(s.trusted_ids), len(s.dev_ids)) == (45000, 2000, 3000)
    assert len(set(s.all_ids)) == 50000

def test_ids_and_missed_poison_survive():
    base = Fake(); labels = {i: base.targets[i] for i in range(10)}
    ds = ImmutableObservedDataset(base, [2, 5, 9], labels, {5}, "label", False)
    assert [ds[i][2] for i in range(3)] == [2, 5, 9]
    assert ds[1][1] == base.targets[5]

def test_patch_is_normalized_after_augmentation():
    base = Fake(); labels = {0: 1}
    ds = ImmutableObservedDataset(base, [0], labels, {0}, "backdoor", False)
    image, label, sample_id = ds[0]
    assert label == 0 and sample_id == 0
    assert torch.all(image[:, -3:, -3:] > 1.5)
