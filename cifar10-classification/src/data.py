"""CIFAR-10 data pipeline: datasets, transforms and DataLoaders."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import Config
from src.utils import get_device, make_generator, seed_worker

CLASS_NAMES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)  # fmt: skip

# Per-channel statistics of the 45,000-image training split (seed 42), computed with
# `compute_channel_stats`. Cross-check against the commonly quoted full-train-set values
# mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616).
CIFAR10_MEAN: tuple[float, float, float] = (0.4913, 0.4822, 0.4466)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2436, 0.2617)

MAX_AUTO_WORKERS = 4
SAMPLE_GRID_SIZE = 16


def resolve_num_workers(requested: int | None) -> int:
    """Return the configured worker count, or ``min(4, cpu_count)`` when it is ``None``."""
    if requested is not None:
        return requested
    return min(MAX_AUTO_WORKERS, os.cpu_count() or 1)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build the (train, eval) transform pipelines.

    Training uses random crop with padding and horizontal flips; evaluation only
    converts to a tensor and normalises.
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def split_indices(num_samples: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle ``range(num_samples)`` with a seeded generator and split off ``val_size`` indices.

    Returns:
        ``(train_indices, val_indices)``.
    """
    permutation = torch.randperm(num_samples, generator=make_generator(seed)).tolist()
    return permutation[val_size:], permutation[:val_size]


def get_datasets(config: Config) -> tuple[Subset, Subset, datasets.CIFAR10]:
    """Create the train, validation and test datasets.

    Train and validation are two views of the official training set that share one
    index split but use separate dataset instances, so validation images never
    receive training augmentation. The official test set is left untouched.
    """
    train_transform, eval_transform = build_transforms()
    augmented = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=train_transform)
    plain = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=eval_transform)

    train_indices, val_indices = split_indices(len(augmented), config.val_split, config.seed)
    return Subset(augmented, train_indices), Subset(plain, val_indices), test_set


def get_dataloaders(
    config: Config, limit: int | None = None
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    """Build the train, validation and test DataLoaders.

    Args:
        config: Experiment configuration.
        limit: If given, keep only the first ``limit`` samples of each split (smoke tests).

    Returns:
        ``(train_loader, val_loader, test_loader, class_names)``. Only the training
        loader shuffles (with a seeded generator) and drops its last incomplete batch.
    """
    train_set, val_set, test_set = get_datasets(config)
    if limit is not None:
        train_set, val_set, test_set = (
            Subset(split, range(min(limit, len(split)))) for split in (train_set, val_set, test_set)
        )
    num_workers = resolve_num_workers(config.num_workers)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": get_device().type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=make_generator(config.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, CLASS_NAMES


def denormalize(images: Tensor) -> Tensor:
    """Undo normalisation for display; accepts ``(C, H, W)`` or ``(N, C, H, W)`` tensors."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def compute_channel_stats(config: Config) -> tuple[Tensor, Tensor]:
    """Compute per-channel mean and std of the raw (un-augmented) training split."""
    raw = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=transforms.ToTensor())
    train_indices, _ = split_indices(len(raw), config.val_split, config.seed)
    images = torch.stack([raw[i][0] for i in train_indices])
    return images.mean(dim=(0, 2, 3)), images.std(dim=(0, 2, 3))


def save_sample_grid(config: Config, path: str | Path | None = None) -> Path:
    """Save a 4x4 grid of augmented training images with their labels.

    Args:
        config: Experiment configuration.
        path: Output file; defaults to ``<plot_dir>/sample_batch.png``.

    Returns:
        The path the figure was written to.
    """
    output_path = Path(path) if path is not None else Path(config.plot_dir) / "sample_batch.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_set, _, _ = get_datasets(config)

    side = int(SAMPLE_GRID_SIZE**0.5)
    fig = Figure(figsize=(6, 6.5))
    axes = fig.subplots(side, side)
    for index, axis in enumerate(axes.flat):
        image, label = train_set[index]
        axis.imshow(denormalize(image).permute(1, 2, 0).numpy())
        axis.set_title(CLASS_NAMES[label], fontsize=9)
        axis.axis("off")
    fig.suptitle("Augmented training samples", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return output_path
