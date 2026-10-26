"""Plotting: training curves, confusion matrix and misclassified samples."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from torch import Tensor

from src.data import denormalize
from src.engine import History

DPI = 150


def _save(fig: Figure, path: str | Path) -> Path:
    """Write ``fig`` to ``path``, creating parent folders, and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    return output_path


def plot_training_curves(history: History, path: str | Path, title: str) -> Path:
    """Plot train vs validation loss, train vs validation accuracy and the LR schedule."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig = Figure(figsize=(15, 4.5))
    loss_ax, acc_ax, lr_ax = fig.subplots(1, 3)

    loss_ax.plot(epochs, history["train_loss"], marker="o", label="Train")
    loss_ax.plot(epochs, history["val_loss"], marker="o", label="Validation")
    loss_ax.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")

    train_acc = np.array(history["train_acc"]) * 100
    val_acc = np.array(history["val_acc"]) * 100
    acc_ax.plot(epochs, train_acc, marker="o", label="Train")
    acc_ax.plot(epochs, val_acc, marker="o", label="Validation")
    best = int(val_acc.argmax())
    acc_ax.scatter([epochs[best]], [val_acc[best]], s=140, facecolors="none", edgecolors="red", zorder=3,
                   label=f"Best val: {val_acc[best]:.2f}% (epoch {epochs[best]})")
    acc_ax.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")

    lr_ax.plot(epochs, history["lr"], marker="o", color="tab:green", label="Learning rate")
    lr_ax.set(title="Learning-rate schedule (value at start of each epoch)", xlabel="Epoch", ylabel="Learning rate")

    for axis in (loss_ax, acc_ax, lr_ax):
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)


def plot_comparison_curves(histories: Mapping[str, History], path: str | Path, title: str) -> Path:
    """Overlay validation accuracy and validation loss for several runs."""
    fig = Figure(figsize=(11, 4.5))
    acc_ax, loss_ax = fig.subplots(1, 2)
    for label, history in histories.items():
        epochs = np.arange(1, len(history["val_acc"]) + 1)
        acc_ax.plot(epochs, np.array(history["val_acc"]) * 100, marker="o", label=label)
        loss_ax.plot(epochs, history["val_loss"], marker="o", label=label)
    acc_ax.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    loss_ax.set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    for axis in (acc_ax, loss_ax):
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], path: str | Path, title: str) -> Path:
    """Plot a confusion matrix: colour is the row-normalised rate, text is the sample count."""
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    fig = Figure(figsize=(9, 8))
    axis = fig.subplots()
    image = axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axis, label="Fraction of true class")

    ticks = np.arange(len(class_names))
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", labelrotation=45)
    for row in ticks:
        for col in ticks:
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8,
                      color="white" if normalised[row, col] > 0.5 else "black")
    return _save(fig, path)


def plot_misclassified(
    images: Tensor,
    targets: Tensor,
    predictions: Tensor,
    confidences: Tensor,
    class_names: Sequence[str],
    path: str | Path,
    title: str,
    columns: int = 5,
) -> Path:
    """Plot normalised ``images`` with their true label, predicted label and confidence."""
    rows = int(np.ceil(len(images) / columns))
    fig = Figure(figsize=(2.4 * columns, 2.5 * rows + 0.6))
    axes = fig.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, image, target, prediction, confidence in zip(
        axes.flat, denormalize(images), targets.tolist(), predictions.tolist(), confidences.tolist()
    ):
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(
            f"True: {class_names[target]} / Pred: {class_names[prediction]} ({confidence:.0%})",
            fontsize=7,
            color="firebrick",
        )
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)
