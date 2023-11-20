from typing import Dict
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from .generate_random_classification_results import Label
from .confusion_matrix import ConfusionMatrix


class ConfusionMatrixGenerator:
    def __init__(self) -> None:
        self.results_per_actual: Counter[tuple[Label, Label]] = Counter()

    def generate(
        self, predictions: list[Label], actuals: list[Label], labels: list[Label]
    ) -> ConfusionMatrix:
        for actual, pred in zip(actuals, predictions):
            self.results_per_actual[actual, pred] += 1

        matrix = []
        for label_cls in labels:
            matrix_row = [self.results_per_actual[label_cls, pred_cls] for pred_cls in labels]
            matrix.append(matrix_row)

        confusion_matrix = ConfusionMatrix(np.array(matrix), labels=tuple(labels))
        return confusion_matrix

    def plot(
        self, confusion_matrix: ConfusionMatrix, cmap: str = "Oranges"
    ) -> [plt.Figure, plt.Axes]:
        figure, axes = plt.subplots()

        ax_image = axes.imshow(confusion_matrix.matrix_array, interpolation="nearest", cmap=cmap)
        min_colour = ax_image.cmap(0.0)
        max_colour = ax_image.cmap(1.0)
        mid_intensity = (
            confusion_matrix.matrix_array.max() + confusion_matrix.matrix_array.min()
        ) / 2.0
        for column in range(len(confusion_matrix.labels)):
            for row in range(len(confusion_matrix.labels)):
                cm_value = confusion_matrix.matrix_array[row, column]
                text_colour = max_colour if cm_value < mid_intensity else min_colour
                axes.text(column, row, cm_value, ha="center", va="center", color=text_colour)

        figure.colorbar(ax_image, ax=axes)
        axes.set(
            xticks=list(range(len(confusion_matrix.labels))),
            yticks=list(range(len(confusion_matrix.labels))),
            xticklabels=confusion_matrix.labels,
            yticklabels=confusion_matrix.labels,
            ylabel="Actuals",
            xlabel="Predictions",
        )
        return figure, axes
