from typing import Tuple
from typing import List
from typing import Dict
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


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

    def plot(self, confusion_matrix:np.ndarray, labels:List[Label]):
        figure, axes = plt.subplots()

        cmap = "Oranges"
        ax_image = axes.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
        min_colour = ax_image.cmap(0.)
        max_colour = ax_image.cmap(1.)
        mid_intensity = (confusion_matrix.max() + confusion_matrix.min()) / 2.
        for column in range(len(labels)):
            for row in range(len(labels)):
                cm_value = confusion_matrix[row, column]
                text_colour = max_colour if cm_value < mid_intensity else min_colour
                axes.text(column, row, cm_value, ha="center", va="center", color=text_colour)

        figure.colorbar(ax_image, ax=axes)
        axes.set(
            xticks=list(range(len(labels))),
            yticks=list(range(len(labels))),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="Actuals",
            xlabel="Predictions"
        )