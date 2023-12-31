---
layout: post
title:  "Confusion Matrices"
categories: metrics measurements
---

This blog will briefly cover confusion matrices, what they are, and how we generate them with code examples. 

* TOC
{:toc}

## Introduction
Confusion matrices are a vital way to visualise your machine learning model's performance for classification tasks, regardless of which type of model you use, and whether that task is detecting pedestrians in a self-driving car, or predicting your customers' next purchase on your website. Most notably, a confusion matrix efficiently highlights how a classifier fails, not just how it succeeds, facilitating a simple method to make informed decisions for model improvement.

Below is an example binary classification confusion matrix, which displays the performance of a cancer detecting classification model. 

![Binary Classification Confusion Matrix]({{ site.baseurl }}/assets/confusion_matrices/binary_classification_plot.png) 


## Layout of a Confusion Matrix

![Confusion Matrix]({{ site.baseurl }}/assets/confusion_matrices/text_binary_classification_plot.png)

The rows of a Confusion Matrix show the number of actual or ground truth labels, the columns show the predicted results, and in combination give the following four attributes.
 - The True Positive (TP) rate, shown on the top left, is the number of predictions correctly identified to be the class of interest, in this case "Cancer".
 - The False Negative (FN) rate, seen to the right of the True Positive rate, shows the number of incorrect model predictions assigned to another label/class when they are actually the class of interest, in this case "Cancer". 
 - The False Positive (FP) rate shows the number of predictions incorrectly identified as the class/label of interest.
 - The True Negative (TN) rate, is the number of predictions correctly identified to be the negative class, in this case "Not Cancer".


### Python Code to Create and Plot

The following codes creates a confusion matrix given a list of model predictions and actuals (ground truths labels).

The two code snippets below are slightly modified excerpts from the class `ConfusionMatrixGenerator` from this blog's github repository [found here](https://github.com/sean-mcmahon/sean-mcmahon.github.io/blob/main/src/confusion_matrices/confusion_matrix_generator.py).

#### Example Code for Creating a Confusion Matrix
{% highlight python %}
from collections import Counter

import numpy as np

Label = str

def generate_confusion_matrix(
    predictions: list[Label], actuals: list[Label]
) -> np.ndarray, list[label]:
    results_per_actual: Counter[tuple[Label, Label]] = Counter()
    for actual, pred in zip(actuals, predictions):
        results_per_actual[actual, pred] += 1
    confusion_matrix = []
    labels = sorted(set(actuals))
    for label_cls in labels:
        matrix_row = [results_per_actual[label_cls, pred_cls] for pred_cls in labels]
        confusion_matrix.append(matrix_row)

    return np.array(confusion_matrix), labels

{% endhighlight python %}


The following code generates the confusion matrix plots seen in this blog, given a confusion matrix and a list of label names to display. 

#### Example Code for Visualising a Confusion Matrix
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

## Multi-Class Confusion Matrix

This code scales for multi-class classification, the more common use case can be seen below.

![Multiclass Classification Confusion Matrix]({{ site.baseurl }}/assets/confusion_matrices/multiclass_classification_plot.png)


# Performance Metrics Calculated from Confusion Matrices

Even if you don't want to plot a confusion matrix, the information contained within them are often used to calculate common classification results, such as; Precision, Recall and F1-Score.


## Recall
Recall the ratio of true positives, or correct predictions, against the total number of actuals, for that class. It gives an idea on how many of the total number of classes you classifier correctly detected in the dataset, defined as:

$$ recall = \frac{ TP }{ TP + FN } $$

[Further explanation on Recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)

## Precision
Precision shows the ratio of true positive, or correct predictions, against the total number of model predictions for that class. By showing the ratio of correct predictions against all predictions for that class, precision gives an indication on how *precise* a classifier is at predicting a given class.

$$ precision = \frac{ TP }{ TP + FP } $$

[Further explanation on Precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision)

## F1-Score
F1-score is the harmonic mean between precision and recall, I think of it as a metric summarising both precision and recall, this is a good metric to look at first when evaluating a model. 

Two commonly used equations for the f1-score are:

$$ F_1 = 2 * \frac{ precision * recall }{ precision + recall } $$

and: 

$$ F_1 = \frac{ 2 * TP }{ 2 * TP + FP + FN } $$

*Note* F1-Score is identical to a lot of the definitions of the Dice coefficient, commonly used in segmentation and in the medical field, defined with the same equation [here](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and [here](https://torchmetrics.readthedocs.io/en/stable/classification/dice.html). So therefore;

$$ F_1 \equiv Dice $$

I've found this equivalence between the two to be the easiest way to gain an intuition as to what the Dice coefficient is telling me about a model's performance.

[Further explanation on F1-score](https://en.wikipedia.org/wiki/F-score)\
[Further explanation on the Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)

## Intersection Over Union (IOU) / Jaccard score.

The IOU metric frequently come up for me as someone who has trained many models for semantic segmentation over the years. However, few explain this metric in terms of a confusion matrix, from that information, the Jaccard or IOU is defined as:

$$ IOU = \frac{ TP }{ TP + FP + FN } $$ 

and also can be defined from the Dice coefficient or F1 score.

$$ IOU = \frac{ Dice }{ 2 - Dice } $$ 

Further reading, which os particularly useful if you aren't already familiar with IOU:
 - [Py Image Search](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
 - [Learn Data Sci](https://www.learndatasci.com/glossary/jaccard-similarity/#:~:text=Jaccard%20Similarity%20is%20a%20common,the%20similarity%20between%20two%20sets.)
 - [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)



## Calculated Metrics for the Multi-Class Classification Example

From the above mutli-class classification confusion matrix we can compute the four metrics discussed above:

{% highlight text %}

Labels: ('Bird', 'Cat', 'Dog', 'Mouse')
Confusion matrix:
[[121  14  15  12]
 [ 13 100  11  19]
 [  9  13 111  10]
 [ 17  18  11 106]]
Gives the following results.
Bird:
recall = 0.75, precision = 0.76, f1_score = 0.75, iou = 0.60
Cat:
recall = 0.70, precision = 0.69, f1_score = 0.69, iou = 0.53
Dog:
recall = 0.78, precision = 0.75, f1_score = 0.76, iou = 0.62
Mouse:
recall = 0.70, precision = 0.72, f1_score = 0.71, iou = 0.55

{% endhighlight text %}

Below is a code snippet for calculating the aforementioned metrics, I opted for code that's easier to read rather than succinctness. The code  is from the class `Metrics` which can be found in this blog post's GitHub repository [via this link](https://github.com/sean-mcmahon/sean-mcmahon.github.io/blob/main/src/confusion_matrices/metrics.py).


{% highlight python %}
# An excerpt from the Metrics class, used to generate the printout above.
def calculate(self, labels:list[str], confusion_matrix: np.ndarray):
    for class_index, class_name in enumerate(labels):
        true_positives = confusion_matrix[class_index, class_index]
        total_predictions_per_class = confusion_matrix[:, class_index].sum()
        total_actuals_per_class = confusion_matrix[class_index, :].sum()
        false_positives = total_predictions_per_class - true_positives
        false_negatives = total_actuals_per_class - true_positives

        # All are defined as Dict[str, float] outside of this code snippet
        self.__recall[class_name] = true_positives / total_actuals_per_class
        self.__precision[class_name] = true_positives / total_predictions_per_class
        self.__f1_score[class_name] = (
            2 * true_positives / (2 * true_positives + false_negatives + false_positives)
        )
        self.__iou[class_name] = true_positives / (
            true_positives + false_positives + false_negatives
        )
{% endhighlight python %}

# Summary

This blog has briefly touched on confusion matrices, a very useful evaluation tool for classification problems in Machine Learning. In addition to providing examples for binary and multi-class classification, some code was provided to give further insights on how to generate and plot these matrices. 

Finally, we covered the essential performance metrics used in most classification problems, such as precision, recall, F1-score, and IOU, and again further clarity was added with code examples. The simplicity of calculating these metrics showcases how confusion matrices serve as the cornerstone for assessing the performance of machine learning models across many applications.

# Further reading on Confusion Matrices

 - [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
 - [Sci-Kit Learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)