import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).parent.parent
SRC_FOLDER = str(ROOT_DIR / "src")
if SRC_FOLDER not in sys.path and Path(SRC_FOLDER).is_dir():
    sys.path.append(SRC_FOLDER)
# pylint: disable=wrong-import-position
from src.confusion_matrices.generate_random_classification_results import (
    GenerateRandomClassificationResults,
)
from src.confusion_matrices.confusion_matrix_generator import ConfusionMatrixGenerator
from src.confusion_matrices.metrics import Metrics


def create_confusion_matrix_plot(
    labels: list[str], predictor_probability: float = 0.8, number_labels: int = 500
):
    results_generator = GenerateRandomClassificationResults(
        number_labels, labels, random_seed=42.42
    )
    predictions, actuals = results_generator.generate(probability=predictor_probability)

    cm_generator = ConfusionMatrixGenerator()
    cm = cm_generator.generate(predictions, actuals)

    fig, axes = cm_generator.plot(cm)

    return fig, axes


def text_binary_classification_plot():
    labels = ["Cancer", "Not Cancer"]
    cm_labels = [
        ["True Positives (TP)", "False Negatives (FN)"],
        ["False Positives (FP)", "True Negatives (TN)"],
    ]
    fig, axes = create_confusion_matrix_plot(labels, predictor_probability=0.55, number_labels=200)
    text_colours_at_position = {t_obj.get_position(): t_obj.get_color() for t_obj in axes.texts}
    [text.remove() for text in axes.texts]  # pylint: disable=expression-not-assigned
    for column in range(len(labels)):
        for row in range(len(labels)):
            cm_value = cm_labels[row][column]
            text_colour = text_colours_at_position[(column, row)]
            axes.text(
                column,
                row,
                cm_value,
                size="medium",
                ha="center",
                va="center",
                color=text_colour,
            )

    axes.set_title("Binary Classification")
    plt.savefig(ROOT_DIR / "assets/confusion_matrices" / "text_binary_classification_plot.png")


def binary_classification_plot():
    labels = ["Cancer", "Not Cancer"]
    fig, axes = create_confusion_matrix_plot(labels, number_labels=300)
    axes.set_title("Binary Classification")
    plt.savefig(ROOT_DIR / "assets/confusion_matrices" / "binary_classification_plot.png")


def multiclass_classification_plot():
    labels = ["Cat", "Dog", "Mouse", "Bird"]
    predictor_probability = 0.65
    number_labels = 600
    results_generator = GenerateRandomClassificationResults(
        number_labels, labels, random_seed=42.42
    )
    predictions, actuals = results_generator.generate(probability=predictor_probability)

    cm_generator = ConfusionMatrixGenerator()
    cm = cm_generator.generate(predictions, actuals)

    _, axes = cm_generator.plot(cm)

    axes.set_title("Multi-Class Classification")
    plt.savefig(ROOT_DIR / "assets/confusion_matrices" / "multiclass_classification_plot.png")

    metrics = Metrics(cm)
    metrics.calculate()
    print(metrics)


if __name__ == "__main__":
    text_binary_classification_plot()
    binary_classification_plot()
    multiclass_classification_plot()
