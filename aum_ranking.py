
import numpy as np
import torch
from torch.utils.data import Dataset


def assign_threshold_samples(num_examples, num_classes):
    threshold_sample_flags = np.zeros((num_examples,), dtype=np.uint8)
    num_threshold_samples = num_examples // (num_classes + 1)
    threshold_sample_flags[:num_threshold_samples] = 1
    np.random.shuffle(threshold_sample_flags)
    return threshold_sample_flags


class ThresholdSamplesDataset(Dataset):
    """A Dataset wrapper that adds threshold samples."""
    def __init__(self, dataset, threshold_sample_flags):
        if not hasattr(dataset, "classes"):
            raise ValueError("dataset must have 'classes' attribute.")
        
        self.dataset = dataset
        self.threshold_sample_flags = threshold_sample_flags
        self.classes = self.dataset.classes + ["fake_label"]

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.threshold_sample_flags[index]:
            return x, len(self.dataset.classes)
        return x, y

    def __len__(self):
        return len(self.dataset)


class AUM:
    def __init__(self, num_examples, device=None):
        self.num_examples = num_examples
        self.device = device
        self.reset()

    @torch.inference_mode()
    def update(self, logits, y, start):
        """
        Updates states with the ground truth labels and predictions.

        Args:
            pred (Tensor): Tensor of label predictions logits of shape (batch_size,
                num_classes).
            y (Tensor): Tensor of ground truth labels with shape (batch_size,).
            start (int): Index of the first example within the dataset.
        """
        logits = logits.to(self.device)
        y = y.to(self.device)

        # Get the logits for the ground truth labels
        batch_size = y.shape[0]
        assigned_logits = logits[torch.arange(batch_size), y]

        # Get the next highest logits
        masked_logits = torch.scatter(logits, dim=1, index=y[..., None], value=-torch.inf)
        largest_other_logits, _ = torch.max(masked_logits, dim=1)

        # Calculate the margins
        margins = assigned_logits - largest_other_logits

        # Accumulate the margin totals
        stop = start + batch_size
        self.margin_totals[start:stop] += margins

        return self
    

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
        self.margin_totals = torch.zeros((self.num_examples,), device=self.device)


def compute_aum_threshold(aum_values, threshold_sample_flags, percentile=0.99):
    threshold_sample_aum_values = aum_values[threshold_sample_flags == 1]
    return np.percentile(threshold_sample_aum_values, percentile)


def flag_mislabeled_examples(aum_values, threshold_sample_flags, aum_threshold):
    mislabeled_example_flags = (threshold_sample_flags == 0) & (aum_values <= aum_threshold)
    mislabeled_example_flags = mislabeled_example_flags.astype(threshold_sample_flags.dtype)
    return mislabeled_example_flags
