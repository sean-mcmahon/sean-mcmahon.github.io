---
layout: post
title:  "Confusion Matrices"
categories: metrics measurements
---

This blog will go in depth about confusion matrices, what they are, why we use them and some code examples on how to generate them. 

Confusion matrices are a common way for visualise your classifier's performance, whether that is a Transformer, CNN or SVM. Perhaps most importantly, they show how classifier fails in an easy to read manor.

| ![Confusion Matrix](/assets/confusion_matrices/text_binary_classification_plot.png) 
 *The rows of a Confusion Matrix show the number of actual or ground truth labels, the columns show the predicted results, and in combination give the following; the True Positive (TP) rate, the False Negative (FN) rate, the False Positive (FP) rate and the True Negative (TN) rate.* | 

### Example of confusion matrix, binary classification


| ![Binary Classification Confusion Matrix](/assets/confusion_matrices/binary_classification_plot.png) |

Confusion matrix creation from predictions and actuals.
{% highlight python %}
from collections import Counter

import numpy as np

Label = str

def generate_confusion_matrix(
    predictions: list[Label], actuals: list[Label], labels: list[Label]
) -> np.ndarray:
    results_per_actual: Counter[tuple[Label, Label]] = Counter()
    for actual, pred in zip(actuals, predictions):
        results_per_actual[actual, pred] += 1
    confusion_matrix = []
    for label_cls in labels:
        matrix_row = [results_per_actual[label_cls, pred_cls] for pred_cls in labels]
        confusion_matrix.append(matrix_row)

    return np.array(confusion_matrix)

{% endhighlight python %}

Generate a confusion matrix plot
{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

Label = str

def plot(
    confusion_matrix: np.ndarray, labels: list[Label], cmap: str = "Oranges"
) -> [plt.Figure, plt.Axes]:
    figure, axes = plt.subplots()

    ax_image = axes.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    min_colour = ax_image.cmap(0.0)
    max_colour = ax_image.cmap(1.0)
    mid_intensity = (
        confusion_matrix.max() + confusion_matrix.min()
    ) / 2.0
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
        xlabel="Predictions",
    )
    return figure, axes
{% endhighlight %}

### Multi-class confusion matrix

This code scales for multi-class classification, the more common use case.

| ![Multiclass Classification Confusion Matrix](/assets/confusion_matrices/multiclass_classification_plot.png) |


### Metrics calculated from Confusion Matrices

Even if you don't want to plot a confusion matrix, the information contained within them are often used to calculate common classification results, such as; Precision, Recall and F1-Score


#### Recall
Recall the ratio of true positives, or correct, predictions, against the total number of actuals, or ground truths, for that class. It gives an idea of how many of the total number of classes you classifier correctly detected in the dataset. Defined as:

$$ recall = \frac{ TP }{ TP + FN } $$

[Further explanation on recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)

#### Precision
Precision shows the ratio of true positive, or correct, predictions, against the total number of model predictions for that class. By showing the ratio of correct predictions against all predictions for that class, precision gives an indication on how *precise* a classifier is at predicting a given class.

$$ precision = \frac{ TP }{ TP + FP } $$

[Further explanation on Precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision)

#### F1-Score
F1-score is the harmonic mean between precision and recall, I think of it as a metric summarising both precision and recall, this is a good metric to look at first when evaluating a model. 

Two commonly used equations for the f1-score are:

$$ F_1 = 2 * \frac{ precision * recall }{ precision + recall } $$

and: 

$$ F_1 = \frac{ 2 * TP }{ 2 * TP + FP + FN } $$

This is identical to a lot of the definitions of the Dice coefficient, commonly used in segmentation and in the medical field, defined with the same equation [here](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and [here](https://torchmetrics.readthedocs.io/en/stable/classification/dice.html).

[Further explanation on F1-score](https://en.wikipedia.org/wiki/F-score)

#### Side note: Intersection Over Union (IOU) or Jaccard score, and the Dice score.

The IOU metric frequently come up for me as someone who has trained many semantic segmentation or pixel-wise classification models over the years. However, few explain this metric in terms of a confusion matrix, from that information, the Jaccard or IOU is defined as:

$$ IOU = \frac{ TP }{ TP + FP + FN } $$ 

and also can be defined from the Dice coefficient or F1 score.

$$ IOU = \frac{ Dice }{ 2 - Dice } $$ 

Further reading, these are particularly useful if you aren't already familiar with IOU:
 - [Py Image Search](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
 - [Learn Data Sci](https://www.learndatasci.com/glossary/jaccard-similarity/#:~:text=Jaccard%20Similarity%20is%20a%20common,the%20similarity%20between%20two%20sets.)
 - [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)



#### Metrics calculation code:

Below is a code snippet for calculating the aforementioned metrics, I opted for code that's easier to read rather than succinctness. 

{% highlight python %}
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
{% endhighlight python %}

### Further reading 

 - [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
 - [Sci-Kit Learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)