"""Evaluation entry point: test-set metrics, plots and misclassified samples.

Example:
    python evaluate.py --model baseline_cnn
    python evaluate.py --model baseline_cnn --checkpoint outputs/checkpoints/baseline_cnn/best.pt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor, nn
from torch.utils.data import Dataset

from src.config import Config, build_arg_parser, config_from_args
from src.data import get_dataloaders
from src.engine import evaluate, load_checkpoint
from src.models import build_model
from src.utils import get_device, make_generator, set_seed, setup_logger
from src.visualize import plot_confusion_matrix, plot_misclassified, plot_training_curves

NUM_MISCLASSIFIED_SHOWN = 25


def parse_args(argv: Sequence[str] | None = None) -> tuple[Config, Path | None]:
    """Parse the command line into ``(config, checkpoint_path)``."""
    parser = build_arg_parser("Evaluate a trained CIFAR-10 classifier on the test set")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate (default: best.pt).")
    args = parser.parse_args(argv)
    return config_from_args(args), args.checkpoint


def compute_metrics(predictions: Tensor, targets: Tensor, class_names: Sequence[str]) -> dict[str, Any]:
    """Compute the confusion matrix, per-class accuracy and the sklearn classification report."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(targets.numpy(), predictions.numpy(), labels=labels)
    return {
        "confusion_matrix": matrix,
        "per_class_accuracy": dict(zip(class_names, (matrix.diagonal() / matrix.sum(axis=1)).tolist())),
        "classification_report": classification_report(
            targets.numpy(), predictions.numpy(), labels=labels, target_names=list(class_names),
            output_dict=True, zero_division=0,
        ),
    }


def select_misclassified(
    model: nn.Module,
    dataset: Dataset,
    predictions: Tensor,
    targets: Tensor,
    device: torch.device,
    count: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pick ``count`` random misclassified samples and compute the model's confidence in each.

    ``predictions`` and ``targets`` must follow the order of ``dataset`` (an unshuffled loader).

    Returns:
        ``(images, true_labels, predicted_labels, confidences)``.
    """
    wrong = (predictions != targets).nonzero().flatten()
    order = torch.randperm(len(wrong), generator=make_generator(seed))[:count]
    chosen = wrong[order].sort().values

    images = torch.stack([dataset[index][0] for index in chosen.tolist()])
    model.eval()
    with torch.inference_mode():
        probabilities = model(images.to(device)).softmax(dim=1).cpu()
    predicted = predictions[chosen]
    confidences = probabilities[torch.arange(len(chosen)), predicted]
    return images, targets[chosen], predicted, confidences


def log_per_class_table(logger: Any, metrics: dict[str, Any]) -> None:
    """Log per-class accuracy, precision, recall and F1."""
    report = metrics["classification_report"]
    logger.info("%-12s %9s %9s %9s %9s", "class", "accuracy", "precision", "recall", "f1")
    for name, accuracy in metrics["per_class_accuracy"].items():
        row = report[name]
        logger.info("%-12s %8.2f%% %9.3f %9.3f %9.3f", name, accuracy * 100, row["precision"], row["recall"], row["f1-score"])


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate a checkpoint on the untouched test set and write metrics and plots."""
    config, checkpoint_path = parse_args(argv)
    set_seed(config.seed)
    device = get_device()
    name = config.model_name
    logger = setup_logger(f"{name}_eval", Path(config.log_dir) / f"{name}_eval.log")

    _, _, test_loader, class_names = get_dataloaders(config)
    model = build_model(name, dropout=config.dropout).to(device)
    checkpoint_path = checkpoint_path or Path(config.checkpoint_dir) / name / "best.pt"
    state = load_checkpoint(checkpoint_path, model, device=device)
    logger.info("Loaded %s (epoch %d, val acc %.2f%%)", checkpoint_path, state.epoch, state.best_val_acc * 100)

    result = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, return_predictions=True)
    logger.info("TEST loss %.4f | TEST accuracy %.2f%% (%d images)", result.loss, result.accuracy * 100, len(result.targets))

    metrics = compute_metrics(result.predictions, result.targets, class_names)
    log_per_class_table(logger, metrics)
    logger.info("\n%s", classification_report(
        result.targets.numpy(), result.predictions.numpy(), target_names=list(class_names), digits=3))

    plot_dir, results_dir = Path(config.plot_dir), Path(config.results_dir)
    history_path = results_dir / f"{name}_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        plot_training_curves(history, plot_dir / f"{name}_curves.png", f"{name}: training curves")
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names, plot_dir / f"{name}_confusion_matrix.png",
        f"{name}: test confusion matrix (accuracy {result.accuracy:.2%})",
    )
    plot_misclassified(
        *select_misclassified(
            model, test_loader.dataset, result.predictions, result.targets, device,
            NUM_MISCLASSIFIED_SHOWN, config.seed,
        ),
        class_names, plot_dir / f"{name}_misclassified.png", f"{name}: sample misclassified test images",
    )

    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    summary = {
        "model": name,
        "checkpoint": str(checkpoint_path),
        "best_epoch": state.epoch,
        "best_val_acc": state.best_val_acc,
        "test_loss": result.loss,
        "test_accuracy": result.accuracy,
        **metrics,
    }
    metrics_path = results_dir / f"{name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s and plots to %s", metrics_path, plot_dir)


if __name__ == "__main__":
    main()
