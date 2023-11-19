import sys
import matplotlib.pyplot as plt
from pathlib import Path

SRC_FOLDER = str(Path(__file__).parent / 'src')
if SRC_FOLDER not in sys.path:
    sys.path.append(SRC_FOLDER)
# pylint: disable=wrong-import-position
from src.confusion_matrices.generate_random_classification_results import GenerateRandomClassificationResults
from src.confusion_matrices.confusion_matrix_generator import ConfusionMatrixGenerator 


def binary_classifaction_plot():
    labels = ["Benign", "Malignant"]
    results_generator = GenerateRandomClassificationResults(500, labels)
    predictions, actuals = results_generator.generate(probability=0.8)

    cm_generator = ConfusionMatrixGenerator()
    cm = cm_generator.generate(predictions, actuals, labels)

    fig, axes = cm_generator.plot(cm)
    
    plt.savefig("binary_classification_plot.png")


if __name__ == "__main__":
    binary_classifaction_plot()