"""Compare the baseline CNN with the residual CNN using their saved results.

Run after training and evaluating both models:
    python compare.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.config import Config, build_arg_parser, config_from_args
from src.engine import History
from src.models import build_model, count_parameters
from src.visualize import plot_comparison_curves

BASELINE = "baseline_cnn"
IMPROVED = "resnet_cnn"
DISPLAY_NAMES = {BASELINE: "Baseline CNN", IMPROVED: "ResNet CNN"}
TEST_SET_SIZE = 10_000
COMPARISON_EPOCH = 5  # early-training checkpoint used to compare convergence speed


@dataclass
class RunSummary:
    """Everything the comparison needs about one trained and evaluated model."""

    name: str
    parameters: int
    epochs: int
    train_minutes: float
    best_epoch: int
    best_val_acc: float
    test_acc: float
    test_loss: float
    per_class_accuracy: dict[str, float]
    history: History

    @property
    def label(self) -> str:
        return DISPLAY_NAMES[self.name]


def load_run(config: Config, name: str) -> RunSummary:
    """Load the history and test metrics written by ``train.py`` and ``evaluate.py``."""
    results_dir = Path(config.results_dir)
    history = json.loads((results_dir / f"{name}_history.json").read_text(encoding="utf-8"))["history"]
    metrics = json.loads((results_dir / f"{name}_metrics.json").read_text(encoding="utf-8"))
    return RunSummary(
        name=name,
        parameters=count_parameters(build_model(name, dropout=config.dropout)),
        epochs=len(history["train_loss"]),
        train_minutes=sum(history["epoch_time"]) / 60,
        best_epoch=metrics["best_epoch"],
        best_val_acc=metrics["best_val_acc"],
        test_acc=metrics["test_accuracy"],
        test_loss=metrics["test_loss"],
        per_class_accuracy=metrics["per_class_accuracy"],
        history=history,
    )


def _row(label: str, base: str, improved: str, delta: str) -> str:
    return f"| {label} | {base} | {improved} | {delta} |"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _train_val_gap_pp(run: RunSummary) -> float:
    """Final-epoch train accuracy minus validation accuracy, in percentage points."""
    return (run.history["train_acc"][-1] - run.history["val_acc"][-1]) * 100


def build_summary_table(base: RunSummary, improved: RunSummary) -> str:
    """Markdown table of the headline numbers for both models."""
    rows = [
        f"| Metric | {base.label} | {improved.label} | Difference |",
        "|---|---|---|---|",
        _row("Parameters", f"{base.parameters:,}", f"{improved.parameters:,}",
             f"{improved.parameters - base.parameters:+,}"),
        _row("Epochs trained", str(base.epochs), str(improved.epochs), "-"),
        _row("Training time (min)", f"{base.train_minutes:.1f}", f"{improved.train_minutes:.1f}",
             f"{improved.train_minutes - base.train_minutes:+.1f}"),
        _row("Best validation accuracy", f"{_percent(base.best_val_acc)} (epoch {base.best_epoch})",
             f"{_percent(improved.best_val_acc)} (epoch {improved.best_epoch})",
             f"{(improved.best_val_acc - base.best_val_acc) * 100:+.2f} pp"),
        _row("Final train accuracy", _percent(base.history["train_acc"][-1]), _percent(improved.history["train_acc"][-1]),
             f"{(improved.history['train_acc'][-1] - base.history['train_acc'][-1]) * 100:+.2f} pp"),
        _row("**Test accuracy**", f"**{_percent(base.test_acc)}**", f"**{_percent(improved.test_acc)}**",
             f"**{(improved.test_acc - base.test_acc) * 100:+.2f} pp**"),
        _row("Test loss", f"{base.test_loss:.4f}", f"{improved.test_loss:.4f}",
             f"{improved.test_loss - base.test_loss:+.4f}"),
    ]
    return "\n".join(rows)


def build_per_class_table(base: RunSummary, improved: RunSummary) -> str:
    """Markdown table of per-class test accuracy for both models."""
    rows = [f"| Class | {base.label} | {improved.label} | Difference |", "|---|---|---|---|"]
    for name, base_acc in base.per_class_accuracy.items():
        improved_acc = improved.per_class_accuracy[name]
        rows.append(_row(name, f"{base_acc * 100:.1f}%", f"{improved_acc * 100:.1f}%",
                         f"{(improved_acc - base_acc) * 100:+.1f} pp"))
    return "\n".join(rows)


def build_conclusion(base: RunSummary, improved: RunSummary) -> str:
    """Write a conclusion that only states what the measured numbers show."""
    diff_pp = (improved.test_acc - base.test_acc) * 100
    std_error_pp = math.sqrt(base.test_acc * (1 - base.test_acc) / TEST_SET_SIZE) * 100
    margin_pp = 2 * std_error_pp
    gains = {c: (improved.per_class_accuracy[c] - a) * 100 for c, a in base.per_class_accuracy.items()}
    best_class, worst_class = max(gains, key=gains.get), min(gains, key=gains.get)
    epoch_index = COMPARISON_EPOCH - 1
    early_base = base.history["val_acc"][epoch_index] * 100
    early_improved = improved.history["val_acc"][epoch_index] * 100

    if diff_pp > margin_pp:
        verdict = (f"The residual network **improved** test accuracy from {base.test_acc:.2%} to {improved.test_acc:.2%} "
                   f"({diff_pp:+.2f} pp) and changed test loss from {base.test_loss:.4f} to {improved.test_loss:.4f}.")
    elif diff_pp < -margin_pp:
        verdict = (f"The residual network did **not** help: test accuracy fell from {base.test_acc:.2%} to "
                   f"{improved.test_acc:.2%} ({diff_pp:+.2f} pp) and test loss went from {base.test_loss:.4f} "
                   f"to {improved.test_loss:.4f}.")
    else:
        verdict = (f"The two models are **statistically indistinguishable** on this test set: {base.test_acc:.2%} vs "
                   f"{improved.test_acc:.2%} ({diff_pp:+.2f} pp), with test loss {base.test_loss:.4f} vs {improved.test_loss:.4f}.")

    evidence = (
        f"- Per class, the most favourable change was **{best_class}** ({gains[best_class]:+.1f} pp) and the least "
        f"favourable was **{worst_class}** ({gains[worst_class]:+.1f} pp).\n"
        f"- After epoch {COMPARISON_EPOCH}, validation accuracy was {early_base:.2f}% for the baseline and "
        f"{early_improved:.2f}% for the ResNet, which shows how quickly each one trained under the same schedule.\n"
        f"- The train-minus-validation accuracy gap at the last epoch was {_train_val_gap_pp(base):+.2f} pp (baseline) and "
        f"{_train_val_gap_pp(improved):+.2f} pp (ResNet); a negative gap means neither model is overfitting.\n"
        f"- Cost: {improved.parameters - base.parameters:+,} parameters "
        f"({(improved.parameters / base.parameters - 1) * 100:+.1f}%) and "
        f"{improved.train_minutes / base.train_minutes:.2f}x the training time."
    )

    if diff_pp > margin_pp:
        interpretation = (
            "The skip connections give gradients a direct path to the early layers and let each block learn a "
            "correction to its input, which is consistent with the faster early progress above."
        )
    elif diff_pp < -margin_pp:
        interpretation = (
            "At this size (about seven convolution layers) the skip connections did not offset the other "
            "architectural changes, so this run gives no evidence that residual connections help here."
        )
    else:
        interpretation = (
            "At this size (about seven convolution layers, with BatchNorm already easing optimisation) there is "
            "little for skip connections to fix, so a large gain was not expected and was not observed."
        )
    caveats = (
        f"Caveats: each model was trained once (seed 42), and the binomial standard error of a test accuracy near "
        f"{base.test_acc:.0%} is about {std_error_pp:.2f} pp, so differences under roughly {margin_pp:.1f} pp are "
        f"within noise. The ResNet also downsamples with stride-2 convolutions instead of max-pooling, so any "
        f"difference cannot be attributed to the skip connections alone."
    )
    return f"{verdict}\n\n{evidence}\n\n{interpretation}\n\n{caveats}"


def write_report(config: Config, base: RunSummary, improved: RunSummary, path: Path) -> None:
    """Write the full comparison as a Markdown file."""
    text = "\n\n".join(
        [
            "# Baseline vs residual CNN",
            f"Both models used identical settings: seed {config.seed}, {base.epochs} epochs, batch size "
            f"{config.batch_size}, {config.optimizer.upper()} (lr {config.learning_rate}, momentum {config.momentum}, "
            f"weight decay {config.weight_decay}), {config.scheduler} schedule, dropout {config.dropout}, gradient "
            f"clipping {'off' if not config.grad_clip else config.grad_clip}, early-stopping patience "
            f"{config.early_stopping_patience}, and the same augmentation and train/validation split. Training time "
            f"is the summed epoch time (train + validation) on a shared CPU, so it is approximate.",
            "## Summary\n\n" + build_summary_table(base, improved),
            "## Per-class test accuracy\n\n" + build_per_class_table(base, improved),
            "![Validation curves](../plots/comparison_curves.png)",
            "## Conclusion\n\n" + build_conclusion(base, improved),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Build the comparison report and plot from the saved results."""
    config = config_from_args(build_arg_parser("Compare baseline and residual CNNs").parse_args(argv))
    base, improved = load_run(config, BASELINE), load_run(config, IMPROVED)
    plot_comparison_curves(
        {run.label: run.history for run in (base, improved)},
        Path(config.plot_dir) / "comparison_curves.png",
        "Baseline vs ResNet: validation curves",
    )
    report_path = Path(config.results_dir) / "comparison.md"
    write_report(config, base, improved, report_path)
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
