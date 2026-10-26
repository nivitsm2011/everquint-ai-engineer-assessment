"""Tests for the model definitions."""

import pytest
import torch
from torch import nn

from src.models import ConvBlock, ResidualBlock, build_model, count_parameters

MODEL_NAMES = ["baseline_cnn", "resnet_cnn"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape_is_batch_by_ten_logits(name: str) -> None:
    model = build_model(name).eval()
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_parameter_count_is_positive(name: str) -> None:
    assert count_parameters(build_model(name)) > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_has_batchnorm_dropout_and_linear_head(name: str) -> None:
    model = build_model(name)
    modules = list(model.modules())
    assert any(isinstance(m, nn.BatchNorm2d) for m in modules)
    assert any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in modules)
    assert isinstance(modules[-1], nn.Linear)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_configured_dropout_rate_is_used_in_the_head(name: str) -> None:
    model = build_model(name, dropout=0.42)
    assert any(isinstance(m, nn.Dropout) and m.p == 0.42 for m in model.modules())


def test_conv_blocks_have_no_conv_bias() -> None:
    block = ConvBlock(3, 8)
    assert all(m.bias is None for m in block.modules() if isinstance(m, nn.Conv2d))


def test_residual_block_without_downsampling_keeps_shape_and_uses_identity() -> None:
    block = ResidualBlock(16, 16).eval()
    x = torch.randn(2, 16, 8, 8)
    assert block(x).shape == x.shape
    assert isinstance(block.shortcut, nn.Identity)


def test_residual_block_with_downsampling_halves_resolution_and_projects() -> None:
    block = ResidualBlock(16, 32, stride=2).eval()
    assert block(torch.randn(2, 16, 8, 8)).shape == (2, 32, 4, 4)
    assert isinstance(block.shortcut, nn.Sequential)


def test_residual_block_adds_the_skip_connection() -> None:
    block = ResidualBlock(8, 8).eval()
    for conv in (m for m in block.modules() if isinstance(m, nn.Conv2d)):
        nn.init.zeros_(conv.weight)
    x = torch.rand(2, 8, 4, 4)  # non-negative, so the final ReLU leaves it unchanged
    assert torch.allclose(block(x), x)


def test_resnet_is_comparable_in_size_to_baseline() -> None:
    baseline, resnet = (count_parameters(build_model(name)) for name in MODEL_NAMES)
    assert abs(resnet - baseline) / baseline < 0.2


def test_build_model_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vgg")
