from typing import Tuple
from typing import List
from typing import Dict
from collections import Counter

import numpy as np

from .random_classification_results import Label


class ConfusionMatrix():
    def __init__(self) -> None:
        self.results_per_actual:Counter[Tuple[Label, Label]] = Counter()

    def generate(self, predictions: List[Label], actuals: List[Label], labels: List[Label]) -> np.ndarray:
        """
        Generates a confusion matrix based on the predictions and actual labels.
        """
        for actual, pred in zip(actuals, predictions):
            self.results_per_actual[actual, pred] += 1

        matrix = []
        for label_cls in labels:
            matrix_row = [self.results_per_actual[label_cls, pred_cls] for pred_cls in labels]
            matrix.append(matrix_row)

        return np.array(matrix)

    def plot(self):
        pass
