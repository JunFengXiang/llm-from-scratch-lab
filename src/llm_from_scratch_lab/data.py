"""Small deterministic datasets used to learn the training workflow."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_toy_classification(
    num_samples: int = 1_000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create two overlapping Gaussian classes without downloading any data."""
    if num_samples < 4:
        raise ValueError("num_samples must be at least 4")

    generator = torch.Generator().manual_seed(seed)
    num_class_0 = num_samples // 2
    num_class_1 = num_samples - num_class_0

    class_0 = torch.randn(num_class_0, 2, generator=generator) * 0.65
    class_0 += torch.tensor([-1.0, -1.0])
    class_1 = torch.randn(num_class_1, 2, generator=generator) * 0.65
    class_1 += torch.tensor([1.0, 1.0])

    features = torch.cat((class_0, class_1), dim=0)
    labels = torch.cat(
        (
            torch.zeros(num_class_0, dtype=torch.long),
            torch.ones(num_class_1, dtype=torch.long),
        )
    )
    permutation = torch.randperm(num_samples, generator=generator)
    return features[permutation], labels[permutation]


def create_dataloaders(
    num_samples: int = 1_000,
    batch_size: int = 64,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible train and validation loaders."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    features, labels = make_toy_classification(num_samples=num_samples, seed=seed)
    split_index = int(num_samples * train_fraction)
    train_dataset = TensorDataset(features[:split_index], labels[:split_index])
    val_dataset = TensorDataset(features[split_index:], labels[split_index:])

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def isolate_one_batch(train_loader: DataLoader) -> DataLoader:
    """Return a loader that repeats one batch, useful for the first debugging gate."""
    features, labels = next(iter(train_loader))
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=len(labels),
        shuffle=False,
    )
