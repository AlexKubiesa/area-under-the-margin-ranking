import numpy as np
import torch
from torch.utils.data import Dataset


def assign_threshold_samples(num_samples, num_classes):
    threshold_sample_flags = np.zeros((num_samples,), dtype=np.uint8)
    num_threshold_samples = num_samples // (num_classes + 1)
    threshold_sample_flags[:num_threshold_samples] = 1
    threshold_sample_flags[num_threshold_samples : 2 * num_threshold_samples] = 2
    np.random.shuffle(threshold_sample_flags)
    threshold_sample_flags_1 = (threshold_sample_flags == 1).astype(np.uint8)
    threshold_sample_flags_2 = (threshold_sample_flags == 2).astype(np.uint8)
    return threshold_sample_flags_1, threshold_sample_flags_2


class ThresholdSamplesDataset(Dataset):
    """
    A Dataset wrapper used to identify mislabeled data.

    Samples are returned as (x, y, index), and a subset of samples are returned with a new, fake label
    instead of their original label.
    """

    def __init__(self, dataset, threshold_sample_flags):
        if not hasattr(dataset, "classes"):
            raise ValueError("dataset must have 'classes' attribute.")

        self.dataset = dataset
        self.threshold_sample_flags = threshold_sample_flags
        self.classes = dataset.classes + ["fake_label"]

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.threshold_sample_flags[index]:
            y = len(self.dataset.classes)
        return x, y, index

    def __len__(self):
        return len(self.dataset)


class AUM:
    def __init__(self, num_samples, device=None):
        self.num_samples = num_samples
        self.device = device
        self.reset()

    @torch.inference_mode()
    def update(self, logits, y, indexes):
        """
        Updates states with the ground truth labels and predictions.

        Args:
            pred (Tensor): Tensor of label predictions logits of shape (batch_size,
                num_classes).
            y (Tensor): Tensor of ground truth labels of shape (batch_size,).
            indexes (Tensor): Tensor of sample indexes within the dataset of shape
                (batch_size,).
        """
        # Get the logits for the ground truth labels
        batch_size = y.shape[0]
        assigned_logits = logits[torch.arange(batch_size), y]

        # Get the next highest logits
        masked_logits = torch.scatter(
            logits, dim=1, index=y[..., None], value=-torch.inf
        )
        largest_other_logits, _ = torch.max(masked_logits, dim=1)

        # Calculate the margins
        margins = assigned_logits - largest_other_logits

        # Accumulate the margin totals
        indexes = indexes.to(self.margin_totals.device)
        margins = margins.to(self.margin_totals.device)
        self.margin_totals[indexes] += margins

    @torch.inference_mode()
    def compute(self, epochs):
        """
        Returns the AUM values.

        Args:
            epochs (int): The number of training epochs that have occurred.
        """
        return self.margin_totals / epochs

    @torch.inference_mode()
    def reset(self):
        """
        Resets the state.
        """
        self.margin_totals = torch.zeros((self.num_samples,), device=self.device)


def compute_aum_threshold(aum_values, threshold_sample_flags, percentile=99):
    threshold_sample_aum_values = aum_values[threshold_sample_flags == 1]
    return np.percentile(threshold_sample_aum_values, percentile)


def flag_mislabeled_samples(aum_values, threshold_sample_flags, aum_threshold):
    mislabeled_sample_flags = (threshold_sample_flags == 0) & (
        aum_values <= aum_threshold
    )
    mislabeled_sample_flags = mislabeled_sample_flags.astype(
        threshold_sample_flags.dtype
    )
    return mislabeled_sample_flags


def combine_mislabeled_sample_flags(
    mislabeled_sample_flags_1, mislabeled_sample_flags_2
):
    mislabeled_sample_flags = (
        mislabeled_sample_flags_1 | mislabeled_sample_flags_2
    ).astype(mislabeled_sample_flags_1.dtype)
    return mislabeled_sample_flags
