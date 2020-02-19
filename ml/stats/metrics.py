"""
Stats functions used in various places in our ml library. The functions in this file
should be multiclass by default.

These functions should take a predictions/labels pair as their input.

This includes two implementations of each of these functions: our in-house version, and
one using sklearn's functions.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics


def predictive_value_for_class(predictions, reality, class_id):
    """
    Predictive value is |True positives| / |Positive predictions|.

    Equivalent definitions: "precision", P(positive example | predictive positive), the
    accuracy of the classifier on examples of this class. Equivalent to PPV/NPV if this
    is binary classification.
    """
    assert len(predictions) == len(reality)

    if len(reality) == 0:
        return np.nan

    return sklearn_predictive_value_for_class(predictions, reality, class_id)


def positive_rate_for_class(predictions, reality, class_id):
    """
    Positive rate is |True Positives| / |Positives in dataset|

    Equivalent definitions: "sensitivity", P(predicted positive | positive example),
    portion of the positives in the dataset that were classified positive or "captured"
    by the classifier. Equivalent to TPR/TNR if class_id=1/0 and this is binary
    classification.
    """
    assert len(predictions) == len(reality)

    if len(reality) == 0:
        return np.nan

    return native_positive_rate_for_class(predictions, reality, class_id)


def acc(predictions, reality):
    """
    Accuracy: what portion of the dataset was correctly classified.
    """
    assert len(predictions) == len(reality)

    if len(reality) == 0:
        return np.nan

    return sklearn_acc(predictions, reality)


def class_lr(predictions, reality, class_id):
    """
    Likelihood ratio for a given class. How do the odds of the given outcome change
    after the classifier makes that prediction.
    """
    cpr = positive_rate_for_class(predictions, reality, class_id)
    cpv = predictive_value_for_class(predictions, reality, class_id)

    return cpr / (1 - cpv)


# Implementations of these functions using sklearn.


def sklearn_predictive_value_for_class(predictions, reality, class_id):
    predictive_value = sklearn.metrics.precision_score(
        reality,
        predictions,
        labels=[class_id],
        average=None,
    )[0]

    return predictive_value


def sklearn_positive_rate_for_class(predictions, reality, class_id):
    """
    In the confusion matrix, the columns represent the different predictions and the
    rows represent the different realities. Thus, with (row, col), (0, 1) is a real 0
    but predicted 1, (0, 2) is a real 0 but predicted 2. e.g. for TPR:
        tpr = (0, 0) / (sum over row 1)

    In this implementation, we use the assumption that class_ids match the array
    indices of that class_id in the confusion matrix. Seems reasonable, but this breaks
    down if there are missing classes in the dataset, e.g if 0 is missing then (0, 0)
    will be True 1's.

    We solve this by inferring the existence of all classes up to class_id, then do a
    simple set union with that list and present classes will make our assumption valid.

    Honestly, this might be more work than it is worth to use sklearn in this case.
    We'll see.
    """
    inferred_classes = list(set(reality) | set(range(class_id + 1))) # max maybe?

    confusion_matrix = sklearn.metrics.confusion_matrix(
        reality,
        predictions,
        labels=inferred_classes,
    )

    true_positives = confusion_matrix[class_id][class_id]
    data_positives = confusion_matrix[class_id].sum()

    if data_positives == 0:
        return np.nan
    else:
        return true_positives / float(data_positives)


def sklearn_acc(predictions, reality):
    accuracy = sklearn.metrics.precision_score(reality, predictions, average='micro')

    return accuracy


# In-house implementations of these functions. Currently trying to deprecate them.

def native_predictive_value_for_class(predictions, reality, class_id):
    """
    Preditive Value for a given class. Equivalent to PPV/NPV if class_id=1/0 and this is
    binary classification.
    """
    true_positives = np.equal(predictions, class_id) & np.equal(reality, class_id)

    false_positives = (
        np.equal(predictions, class_id) &
        (np.equal(reality, class_id) == False)
    )

    num_true_positives = true_positives.sum()
    num_false_positives = float(false_positives.sum())

    if num_true_positives + num_false_positives == 0:
        return np.nan
    else:
        return num_true_positives / (num_true_positives + num_false_positives)


def native_positive_rate_for_class(predictions, reality, class_id):
    """
    Positive Rate for a given class. Equivalent to TPR/TNR if class_id=1/0 and this is
    binary classification.
    """
    data_positives = np.equal(reality, class_id)
    true_positives = np.equal(predictions, class_id) & data_positives

    num_true_positives = true_positives.sum()
    num_data_positives = float(data_positives.sum())

    if num_data_positives == 0:
        return np.nan
    else:
        return num_true_positives / num_data_positives


def native_acc(predictions, reality):
    correct = np.equal(predictions, reality)

    if len(predictions) == 0:
        return np.nan
    else:
        return correct.sum() / float(len(predictions))

