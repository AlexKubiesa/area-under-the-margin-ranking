import numpy as np
import torch
from torch.utils.data import Dataset

import aum_ranking


class DummyDataset(Dataset):
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_assign_threshold_samples():
    flags_1, flags_2 = aum_ranking.assign_threshold_samples(
        num_samples=100, num_classes=5
    )
    for flags in [flags_1, flags_2]:
        assert isinstance(flags, np.ndarray)
        assert flags.shape == (100,)
        assert np.count_nonzero(flags == 1) == 16  # 100 // (5 + 1)
        assert np.count_nonzero(flags == 0) == 84  # 100 - 16


def test_ThresholdSamplesDataset():
    data = [("fluffy cat", 0), ("happy dog", 1), ("angry dog", 1), ("scared cat", 0)]
    classes = ["cat", "dog"]
    dataset = DummyDataset(data, classes)
    flags = [1, 0, 1, 0]
    threshold_dataset = aum_ranking.ThresholdSamplesDataset(dataset, flags)
    assert threshold_dataset.classes == ["cat", "dog", "fake_label"]
    assert threshold_dataset[0] == ("fluffy cat", 2, 0)  # (x, y, index)
    assert threshold_dataset[1] == ("happy dog", 1, 1)
    assert threshold_dataset[2] == ("angry dog", 2, 2)
    assert threshold_dataset[3] == ("scared cat", 0, 3)


def test_AUM():
    aum = aum_ranking.AUM(num_samples=3)
    # Epoch 1
    aum.update(
        # margin updates: +1.16, -0.92,  0.00
        logits=torch.tensor([[0.63, -0.53], [-0.28, 0.64]]),
        y=torch.tensor([0, 0]),
        indexes=torch.tensor([0, 1]),
    )
    aum.update(
        # margin updates:  0.00,  0.00, +1.04
        logits=torch.tensor([[-0.10, 0.94]]),
        y=torch.tensor([1]),
        indexes=torch.tensor([2]),
    )
    # Epoch 2
    aum.update(
        # margin updates: +0.71,  0.00, +1.63
        logits=torch.tensor([[0.97, 0.26], [-0.64, 0.99]]),
        y=torch.tensor([0, 1]),
        indexes=torch.tensor([0, 2]),
    )
    aum.update(
        # margin updates:  0.00, -0.32,  0.00
        logits=torch.tensor([[-0.84, -0.52]]),
        y=torch.tensor([0]),
        indexes=torch.tensor([1]),
    )
    aum_values = aum.compute(epochs=2).numpy()
    expected = np.array([0.935, -0.620, 1.335])
    assert np.linalg.norm(aum_values - expected) < 0.0001


def test_compute_aum_threshold():
    aum_values = np.linspace(-5, 5, num=11, dtype=np.float32)
    flags = np.array([1] * 5 + [0] * 6)
    print(aum_values, flags)
    threshold = aum_ranking.compute_aum_threshold(aum_values, flags)
    expected = -1.04
    assert abs(threshold - expected) < 0.0001


def test_flag_mislabeled_samples():
    aum_values = np.linspace(-5, 5, num=11, dtype=np.float32)
    threshold_sample_flags = np.array([0, 1] * 5 + [0], dtype=np.uint8)
    threshold = 0.0  # Doesn't matter if it's the real threshold
    mislabeled_sample_flags = aum_ranking.flag_mislabeled_samples(
        aum_values, threshold_sample_flags, threshold
    )
    expected = np.array([1, 0] * 3 + [0] * 5, dtype=np.uint8)
    assert np.all(mislabeled_sample_flags == expected)


def test_combine_mislabeled_sample_flags():
    flags_1 = np.array([0, 0, 1, 1])
    flags_2 = np.array([0, 1, 0, 1])
    flags = aum_ranking.combine_mislabeled_sample_flags(flags_1, flags_2)
    expected = np.array([0, 1, 1, 1])
    assert np.all(flags == expected)
