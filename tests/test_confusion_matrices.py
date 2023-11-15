import numpy as np
import pytest

from code.confusion_matrices.random_classification_results import (
    RandomClassificationResults,
)
from code.confusion_matrices.confusion_matrix import ConfusionMatrix



@pytest.mark.parametrize(
    "labels", [
        ["Cancer", "Not Cancer"],
        ["a", "b", "c", "d"],
    ]
)
def test_confusion_matrix_generate(labels):
    results_generator = RandomClassificationResults(20, labels)
    predictions, actuals = results_generator.generate()

    confusion_matrix = ConfusionMatrix()
    matrix_array = confusion_matrix.generate(predictions, actuals, labels)

    assert matrix_array.shape == (
        len(labels),
        len(labels),
    ), f"shape should be {len(labels)},{len(labels)} got {matrix_array.shape}"
    assert matrix_array.sum() == len(predictions)
    assert matrix_array[:,0].sum() == (np.array(predictions) == labels[0]).sum()
    assert matrix_array[1,:].sum() == (np.array(actuals) == labels[1]).sum()