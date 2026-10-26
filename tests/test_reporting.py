"""Tests for the evaluation metrics and the baseline-vs-improved comparison report."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from compare import BASELINE, IMPROVED, RunSummary, build_conclusion, build_per_class_table, build_summary_table
from evaluate import compute_metrics, select_misclassified
from src.data import CLASS_NAMES


def test_compute_metrics_matches_hand_counted_results() -> None:
    targets = torch.arange(10).repeat(2)  # two samples of every class
    predictions = targets.clone()
    predictions[0] = 1  # airplane predicted as automobile
    predictions[10] = 1  # the other airplane predicted as automobile too

    metrics = compute_metrics(predictions, targets, CLASS_NAMES)

    matrix = metrics["confusion_matrix"]
    assert matrix.sum() == 20 and matrix[0, 1] == 2 and matrix[0, 0] == 0
    assert metrics["per_class_accuracy"]["airplane"] == 0.0
    assert metrics["per_class_accuracy"]["ship"] == 1.0
    assert metrics["classification_report"]["automobile"]["precision"] == pytest.approx(2 / 4)


def test_select_misclassified_returns_only_wrong_predictions_with_confidences() -> None:
    torch.manual_seed(0)
    images, labels = torch.randn(40, 3, 4, 4), torch.randint(0, 10, (40,))
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 10)).eval()
    with torch.inference_mode():
        predictions = model(images).argmax(dim=1)

    shown, true, predicted, confidence = select_misclassified(
        model, TensorDataset(images, labels), predictions, labels, torch.device("cpu"), count=5, seed=0
    )

    assert len(shown) == len(true) == len(predicted) == len(confidence) == 5
    assert (true != predicted).all()
    assert ((confidence > 0) & (confidence <= 1)).all()


def _run(name: str, test_acc: float) -> RunSummary:
    per_class = {c: test_acc for c in CLASS_NAMES}
    history = {"train_acc": [test_acc] * 10, "val_acc": [test_acc] * 10}
    return RunSummary(
        name=name, parameters=1000, epochs=10, train_minutes=5.0, best_epoch=10, best_val_acc=test_acc,
        test_acc=test_acc, test_loss=0.5, per_class_accuracy=per_class, history=history,
    )


@pytest.mark.parametrize(
    "improved_acc,expected",
    [(0.85, "improved"), (0.75, "did **not** help"), (0.801, "statistically indistinguishable")],
)
def test_conclusion_wording_follows_the_measured_difference(improved_acc: float, expected: str) -> None:
    conclusion = build_conclusion(_run(BASELINE, 0.80), _run(IMPROVED, improved_acc))
    assert expected in conclusion
    assert "single" in conclusion or "trained once" in conclusion


def test_tables_contain_the_measured_numbers() -> None:
    base, improved = _run(BASELINE, 0.8170), _run(IMPROVED, 0.7532)
    summary = build_summary_table(base, improved)
    assert "81.70%" in summary and "75.32%" in summary and "-6.38 pp" in summary
    per_class = build_per_class_table(base, improved)
    assert all(name in per_class for name in CLASS_NAMES)
