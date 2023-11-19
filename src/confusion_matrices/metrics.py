from typing import List
from typing import Dict

from .confusion_matrix import ConfusionMatrix


class Metrics:
    def __init__(self, confusion_matrix: ConfusionMatrix) -> None:
        self.__confusion_matrix = confusion_matrix
        if len(self.__confusion_matrix.labels) != self.__confusion_matrix.matrix_array.shape[0]:
            raise ValueError(
                f"The number of class names, {len(self.__confusion_matrix.labels)}, must match "
                f"the number of classes in the confusion matrix ({self.__confusion_matrix.matrix_array.shape})"
            )

        self.__recall: Dict[str, float] = {}
        self.__precision: Dict[str, float] = {}
        self.__f1_score: Dict[str, float] = {}
        self.__iou: Dict[str, float] = {}

    def calculate(self):
        for class_index, class_name in enumerate(self.__confusion_matrix.labels):
            true_positives = self.__confusion_matrix.matrix_array[class_index, class_index]
            total_predictions_per_class = self.__confusion_matrix.matrix_array[:, class_index].sum()
            total_actuals_per_class = self.__confusion_matrix.matrix_array[class_index, :].sum()
            false_positives = total_predictions_per_class - true_positives
            false_negatives = total_actuals_per_class - true_positives

            self.__recall[class_name] = true_positives / total_actuals_per_class
            self.__precision[class_name] = true_positives / total_predictions_per_class
            self.__f1_score[class_name] = (
                2 * true_positives / (2 * true_positives + false_negatives + false_positives)
            )
            self.__iou[class_name] = true_positives / (
                true_positives + false_positives + false_negatives
            )

    @property
    def recall(self) -> Dict[str, float]:
        return self.__recall

    @property
    def precision(self) -> Dict[str, float]:
        return self.__precision

    @property
    def f1_score(self) -> Dict[str, float]:
        return self.__f1_score

    @property
    def iou(self) -> Dict[str, float]:
        return self.__iou
