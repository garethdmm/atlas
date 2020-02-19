"""
Metrics for binary classification. The plan is to deprecate these in favour of functions
that support multi-class, and are implemented using other libraries (to prevent any
more) silly bugs.
"""

import numpy as np


def acc(predictions, reality):
    assert len(predictions) == len(reality)

    correct = np.equal(predictions, reality)
    #correct = np.equal(predictions.values, reality.values)

    if len(predictions) == 0:
        return np.nan
    else:
        return correct.sum() / float(len(predictions))


def ppv(predictions, reality):
    """
    Positive predictive value (positive precision) of a classification. Takes in numpy
    arrays (and possibly pandas series but that's not the intention). Intuitively this
    represents how much we can trust any given positive classification.

    PPV is |True Positives| / |Predicted Positive|.
    """
    assert len(predictions) == len(reality)

    true_positives = np.equal(predictions, 1) & np.equal(reality, 1)
    false_positives = np.equal(predictions, 1) & np.equal(reality, 0)

    num_true_positives = true_positives.sum()
    num_false_positives = float(false_positives.sum())

    if num_true_positives + num_false_positives == 0:
        return np.nan
    else:
        return num_true_positives / (num_true_positives + num_false_positives)


def npv(predictions, reality):
    """
    Negative predictive value (negative precision) of a classification. Takes in numpy
    arrays (and possibly pandas series but that's not the intention). Intuitively this
    represents how much we can trust any given negative classification.

    NPV is |True Negatives| / |Predicted Negative|.
    """
    assert len(predictions) == len(reality)

    true_negatives = np.equal(predictions, 0) & np.equal(reality, 0)
    false_negatives = np.equal(predictions, 0) & np.equal(reality, 1)

    num_true_negatives = true_negatives.sum()
    num_false_negatives = float(false_negatives.sum())

    if num_true_negatives + num_false_negatives == 0:
        return np.nan
    else:
        return num_true_negatives / (num_true_negatives + num_false_negatives)


def tpr(predictions, reality):
    """
    True positive rate (positive recall) of a classification. Takes in numpy arrays
    (and possible pandas series but that's not the intention). Intuitively this
    represents how much of the positive cases the prediction captured.

    TPR is |True Positives| / |All Positives in the dataset|
    """
    assert len(predictions) == len(reality)

    true_positives = np.equal(predictions, 1) & np.equal(reality, 1)

    num_true_positives = true_positives.sum()
    num_data_positives = float(reality.sum())

    if num_data_positives == 0:
        return np.nan
    else:
        return num_true_positives / num_data_positives


def tnr(predictions, reality):
    """
    True negative rate (negative recall) of a classification. Takes in numpy arrays
    (and possible pandas series but that's not the intention). Intuitively this
    represents how much of the negative cases the predictions captured.

    TPR is |True Negatives| / |All Negatives in the dataset|
    """
    assert len(predictions) == len(reality)

    true_negatives = np.equal(predictions, 0) & np.equal(reality, 0)

    num_true_negatives = true_negatives.sum()
    num_data_negatives = float(np.equal(reality, 0).sum())

    if num_data_negatives == 0:
        return np.nan
    else:
        return num_true_negatives / num_data_negatives


def lrp(preds, rels):
    """
    Positive likelihood ratio. Tells us how to change our beliefs about whether the
    patient has the disease post-test.

    Because of np.float64 behaviour, this will return inf it it is 1 / 0 and Nan if it
    is 0 / 0. I think that's fine.
    """
    return tpr(preds, rels) / (1.0 - ppv(preds, rels))


def lrn(preds, rels):
    """
    Negative likelihood ratio, the pair of positive likelihood ratio.
    """
    return tnr(preds, rels) / (1.0 - npv(preds, rels))

