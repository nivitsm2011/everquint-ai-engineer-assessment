"""Export a trained model to ONNX and check it against PyTorch.

Example:
    python export_onnx.py                      # model named in the config (baseline_cnn)
    python export_onnx.py --model resnet_cnn   # export the residual network instead
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn

from src.config import Config, build_arg_parser, config_from_args
from src.data import get_datasets
from src.engine import load_checkpoint
from src.models import build_model
from src.utils import set_seed, setup_logger

ONNX_OPSET = 18
INPUT_NAME = "image"
OUTPUT_NAME = "logits"
VERIFY_IMAGES = 100


def export_model(model: nn.Module, path: str | Path, opset: int = ONNX_OPSET) -> Path:
    """Export ``model`` to ONNX with a dynamic batch axis and return the file path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        (torch.randn(2, 3, 32, 32),),
        str(output_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        opset_version=opset,
        external_data=False,
    )
    return output_path


def compare_with_pytorch(model: nn.Module, onnx_path: str | Path, images: Tensor) -> tuple[float, float]:
    """Run ``images`` through PyTorch and onnxruntime.

    Returns:
        ``(max_abs_logit_difference, prediction_agreement)`` where agreement is a fraction.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model.cpu().eval()(images).numpy()
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: images.numpy()})[0]
    max_difference = float(np.abs(expected - actual).max())
    agreement = float((expected.argmax(axis=1) == actual.argmax(axis=1)).mean())
    return max_difference, agreement


def parse_args(argv: Sequence[str] | None = None) -> tuple[Config, Path | None]:
    """Parse the command line into ``(config, checkpoint_path)``."""
    parser = build_arg_parser("Export a trained CIFAR-10 model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to export (default: best.pt).")
    args = parser.parse_args(argv)
    return config_from_args(args), args.checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    """Export the best checkpoint and verify it on the first test images."""
    config, checkpoint_path = parse_args(argv)
    set_seed(config.seed)
    logger = setup_logger("export_onnx")

    model = build_model(config.model_name, dropout=config.dropout)
    checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / config.model_name / "best.pt"
    state = load_checkpoint(checkpoint_path, model)
    logger.info("Loaded %s (epoch %d, val acc %.2f%%)", checkpoint_path, state.epoch, state.best_val_acc * 100)

    onnx_path = export_model(model, config.onnx_path)
    logger.info("Exported %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)

    _, _, test_set = get_datasets(config)
    images = torch.stack([test_set[index][0] for index in range(VERIFY_IMAGES)])
    max_difference, agreement = compare_with_pytorch(model, onnx_path, images)
    logger.info(
        "onnxruntime vs PyTorch on %d test images: max |logit difference| %.2e, prediction agreement %.1f%%",
        VERIFY_IMAGES,
        max_difference,
        agreement * 100,
    )


if __name__ == "__main__":
    main()
