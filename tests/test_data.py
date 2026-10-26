"""Tests for the data pipeline, using a small in-memory stand-in for CIFAR-10 (no download)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from src import data
from src.config import Config
from src.data import CIFAR10_MEAN, CIFAR10_STD, build_transforms, denormalize, get_dataloaders, split_indices

TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 200, 40, 20


class FakeCIFAR10(Dataset):
    """Random 32x32 RGB images with the same constructor signature as torchvision's CIFAR10."""

    def __init__(self, root: str, train: bool = True, download: bool = False, transform=None) -> None:
        rng = np.random.default_rng(0 if train else 1)
        size = TRAIN_SIZE if train else TEST_SIZE
        self.images = rng.integers(0, 256, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = rng.integers(0, 10, size=size).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return (self.transform(image) if self.transform else image), self.targets[index]


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Config:
    monkeypatch.setattr(data.datasets, "CIFAR10", FakeCIFAR10)
    return Config(data_dir=str(tmp_path), batch_size=8, num_workers=0, val_split=VAL_SIZE)


def test_split_sizes_and_no_overlap() -> None:
    train_idx, val_idx = split_indices(TRAIN_SIZE, VAL_SIZE, seed=42)
    assert (len(train_idx), len(val_idx)) == (TRAIN_SIZE - VAL_SIZE, VAL_SIZE)
    assert set(train_idx).isdisjoint(val_idx)
    assert sorted(train_idx + val_idx) == list(range(TRAIN_SIZE))


def test_split_is_reproducible_and_depends_on_seed() -> None:
    assert split_indices(TRAIN_SIZE, VAL_SIZE, seed=1) == split_indices(TRAIN_SIZE, VAL_SIZE, seed=1)
    assert split_indices(TRAIN_SIZE, VAL_SIZE, seed=1) != split_indices(TRAIN_SIZE, VAL_SIZE, seed=2)


def test_datasets_use_disjoint_indices_and_expected_sizes(config: Config) -> None:
    train_set, val_set, test_set = data.get_datasets(config)
    assert (len(train_set), len(val_set), len(test_set)) == (TRAIN_SIZE - VAL_SIZE, VAL_SIZE, TEST_SIZE)
    assert set(train_set.indices).isdisjoint(val_set.indices)


def test_batches_have_expected_shapes_and_label_range(config: Config) -> None:
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    images, labels = next(iter(train_loader))
    assert images.shape == (8, 3, 32, 32) and images.dtype == torch.float32
    assert labels.shape == (8,) and 0 <= labels.min() and labels.max() <= 9
    assert len(class_names) == 10
    assert len(train_loader) == (TRAIN_SIZE - VAL_SIZE) // 8  # incomplete last batch dropped
    assert sum(len(batch[1]) for batch in val_loader) == VAL_SIZE
    assert sum(len(batch[1]) for batch in test_loader) == TEST_SIZE


def test_channel_stats_are_computed_from_the_training_split_only(config: Config) -> None:
    mean, std = data.compute_channel_stats(config)
    train_indices, _ = split_indices(TRAIN_SIZE, VAL_SIZE, config.seed)
    pixels = FakeCIFAR10(config.data_dir).images[train_indices] / 255.0
    assert np.allclose(mean.numpy(), pixels.mean(axis=(0, 1, 2)), atol=1e-4)
    assert np.allclose(std.numpy(), pixels.std(axis=(0, 1, 2), ddof=1), atol=1e-4)


def test_eval_transform_applies_per_channel_normalisation() -> None:
    _, eval_transform = build_transforms()
    pixel = (120, 60, 200)
    tensor = eval_transform(Image.new("RGB", (32, 32), pixel))
    for channel in range(3):
        expected = (pixel[channel] / 255 - CIFAR10_MEAN[channel]) / CIFAR10_STD[channel]
        assert torch.allclose(tensor[channel], torch.full((32, 32), expected), atol=1e-5)


def test_denormalize_inverts_the_eval_transform() -> None:
    _, eval_transform = build_transforms()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    restored = denormalize(eval_transform(image))
    assert torch.allclose(restored, torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255, atol=1e-5)


def test_validation_images_are_not_augmented_but_training_images_are(config: Config) -> None:
    train_set, val_set, _ = data.get_datasets(config)
    assert torch.equal(val_set[0][0], val_set[0][0])
    assert any(not torch.equal(train_set[0][0], train_set[0][0]) for _ in range(10))


def test_only_the_train_loader_shuffles(config: Config) -> None:
    train_loader, val_loader, test_loader, _ = get_dataloaders(config)
    first_epoch = [batch[1].tolist() for batch in train_loader]
    second_epoch = [batch[1].tolist() for batch in train_loader]
    assert first_epoch != second_epoch
    assert [b[1].tolist() for b in val_loader] == [b[1].tolist() for b in val_loader]
    assert [b[1].tolist() for b in test_loader] == [b[1].tolist() for b in test_loader]
