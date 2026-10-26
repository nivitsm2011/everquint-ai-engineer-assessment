"""Model definitions: baseline CNN and its residual variant."""

from __future__ import annotations

from typing import Callable, Sequence

from torch import Tensor, nn

NUM_CLASSES = 10
BASELINE_CHANNELS = (32, 64, 128)
BLOCK_DROPOUT = 0.1


class ConvBlock(nn.Module):
    """Two ``Conv3x3 -> BatchNorm -> ReLU`` stages, then max-pooling and spatial dropout.

    Halves the spatial resolution and maps ``in_channels`` to ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = BLOCK_DROPOUT) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class BaselineCNN(nn.Module):
    """Three ``ConvBlock`` stages followed by global average pooling and a linear classifier.

    Outputs raw logits of shape ``(N, num_classes)``; apply softmax outside the model.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability applied before the final linear layer.
        channels: Output channels of each convolutional block.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        channels: Sequence[int] = BASELINE_CHANNELS,
    ) -> None:
        super().__init__()
        in_channels = [3, *channels[:-1]]
        self.features = nn.Sequential(
            *(ConvBlock(c_in, c_out) for c_in, c_out in zip(in_channels, channels))
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Basic ResNet block: ``Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN``, plus a skip connection.

    The skip is the identity when the shape is unchanged, otherwise a 1x1 convolution
    with batch norm that matches the stride and channel count. A final ReLU follows the
    addition.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.residual(x) + self.shortcut(x))


class ResNetCNN(nn.Module):
    """Small ResNet: conv stem, three residual stages, global average pooling, linear head.

    Each stage is one :class:`ResidualBlock` (the first keeps the resolution, the others
    halve it with stride 2) followed by spatial dropout. The widths match
    :class:`BaselineCNN` so the two models are comparable in size and compute.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability applied before the final linear layer.
        channels: Output channels of each residual stage.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        channels: Sequence[int] = BASELINE_CHANNELS,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stage_inputs = [channels[0], *channels[:-1]]
        stages: list[nn.Module] = []
        for index, (c_in, c_out) in enumerate(zip(stage_inputs, channels)):
            stages += [ResidualBlock(c_in, c_out, stride=1 if index == 0 else 2), nn.Dropout2d(BLOCK_DROPOUT)]
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.stages(self.stem(x)))


def _init_weights(module: nn.Module) -> None:
    """Kaiming-initialise convolution weights (ReLU gain, fan-out)."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "baseline_cnn": BaselineCNN,
    "resnet_cnn": ResNetCNN,
}


def build_model(name: str, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    """Instantiate a model by its config name.

    Raises:
        ValueError: If ``name`` is not a registered model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
