import numpy as np
import pytest
import matplotlib.pyplot as plt

from code.confusion_matrices.generate_random_classification_results import (
    GenerateRandomClassificationResults,
)
from code.confusion_matrices.confusion_matrix_generator import ConfusionMatrixGenerator
from code.confusion_matrices.metrics import Metrics


@pytest.mark.parametrize(
    "labels",
    [
        ["Cancer", "Not Cancer"],
        ["a", "b", "c", "d"],
    ],
)
def test_confusion_matrix_generate(labels):
    results_generator = GenerateRandomClassificationResults(20, labels)
    predictions, actuals = results_generator.generate()

    confusion_matrix_generator = ConfusionMatrixGenerator()
    cm = confusion_matrix_generator.generate(predictions, actuals, labels)

    matrix_array = cm.matrix_array
    assert matrix_array.shape == (
        len(labels),
        len(labels),
    ), f"shape should be {len(labels)},{len(labels)} got {matrix_array.shape}"
    assert matrix_array.sum() == len(predictions)
    assert matrix_array[:, 0].sum() == (np.array(predictions) == labels[0]).sum()
    assert matrix_array[1, :].sum() == (np.array(actuals) == labels[1]).sum()


def test_confusion_matrix_plot():
    labels = ["a", "b", "c", "d"]
    results_generator = GenerateRandomClassificationResults(500, labels)
    predictions, actuals = results_generator.generate(probability=0.8)

    cm_generator = ConfusionMatrixGenerator()
    cm = cm_generator.generate(predictions, actuals, labels)

    cm_generator.plot(cm)
    # plt.show()
    plt.close()


def test_confusion_matrix_metrics_perfect_score():
    labels = ["a", "b"]
    predictions = ["a", "a", "b", "b"]
    actuals = ["a", "a", "b", "b"]
    confusion_matrix = ConfusionMatrixGenerator()
    matrix_array = confusion_matrix.generate(predictions, actuals, labels)

    metrics = Metrics(matrix_array)
    metrics.calculate()

    assert all(metric == 1 for metric in metrics.recall.values())
    assert all(metric == 1 for metric in metrics.precision.values())
    assert all(metric == 1 for metric in metrics.f1_score.values())
    assert all(metric == 1 for metric in metrics.iou.values())


@pytest.mark.parametrize("predictor_probability", [0.1, 0.5, 0.9])
def test_confusion_matrix_metrics_(predictor_probability):
    # predictor_probability = 0.7
    labels = ["apple", "bannana", "carrot", "durian"]
    results_generator = GenerateRandomClassificationResults(500, labels)
    predictions, actuals = results_generator.generate(probability=predictor_probability)
    confusion_matrix = ConfusionMatrixGenerator()
    matrix_array = confusion_matrix.generate(predictions, actuals, labels)

    metrics = Metrics(matrix_array)
    metrics.calculate()

    calc_f1 = lambda precision, recall: 2 * precision * recall / (precision + recall)
    assert np.allclose(
        list(metrics.f1_score.values()),
        [calc_f1(p, r) for p, r in zip(metrics.precision.values(), metrics.recall.values())],
    )
    assert np.allclose(
        list(metrics.iou.values()), [f1_ / (2 - f1_) for f1_ in metrics.f1_score.values()]
    )
    assert all(metric >= (predictor_probability - 0.05) for metric in metrics.recall.values())
    assert all(metric >= (predictor_probability - 0.05) for metric in metrics.precision.values())
    assert all(metric >= (predictor_probability - 0.05) for metric in metrics.f1_score.values())
