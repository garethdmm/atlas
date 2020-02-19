"""
Stats functions to create plottable curves.
"""

from collections import defaultdict
import numpy as np
import pandas as pd

import ml.stats.binary_metrics
import ml.stats.thresholding as thresholding


def acc_over_time(preds, rels, bucket_size=24*60):
    """
    Get the accuracy of a binary classifier in buckets of a given size over the whole
    dataset. The idea here is to recognize if a classifier's results are stable over
    time or have strong regimes.

    This kind of doesn't feel like it belongs here, since all the other curves have
    something to do with the probability threshold, but it feels like it doesn't
    belong in stats.metrics either, since those are all single-number metrics.
    """

    accs = [
        ml.stats.binary_metrics.acc(
            rels[i*bucket_size: i*bucket_size + bucket_size],
            preds[i*bucket_size: i*bucket_size + bucket_size],
        )
        for i in range(0, len(rels) / bucket_size)
    ]

    return pd.Series(accs)


def precisions_and_frequency_at_z(probs, reality):
    """
    Get the curves (acc|ppv|npv|num_preds)[i] for i an increasing probablility
    threshold.
    """
    accs = []
    ppvs = []
    npvs = []
    num_preds = []

    idx = np.arange(0.5, 0.9, 0.01)

    series_at_confidences = thresholding.threshold_predictions_at_many_confidences(
        probs,
        reality,
        idx,
    )

    for i in idx:
        accs.append(ml.stats.binary_metrics.acc(
            series_at_confidences[i][0],
            series_at_confidences[i][1]),
        )

        ppvs.append(ml.stats.binary_metrics.ppv(
            series_at_confidences[i][0],
            series_at_confidences[i][1]),
        )

        npvs.append(ml.stats.binary_metrics.npv(
            series_at_confidences[i][0],
            series_at_confidences[i][1]),
        )

        num_preds.append(len(series_at_confidences[i][0]))

    accs = pd.Series(accs, index=idx, name='ACC')
    ppvs = pd.Series(ppvs, index=idx, name='PPV')
    npvs = pd.Series(npvs, index=idx, name='NPV')
    num_preds = pd.Series(num_preds, index=idx, name='Freq')
    num_preds = num_preds / float(len(reality))

    pvs_and_freq_df = pd.concat([accs, ppvs, npvs, num_preds], axis=1)

    return pvs_and_freq_df


def likelihoods_at_z(probs, reality, thresholds=None):
    """
    Likelihood curves over thresholds. Compatible with binary or multiclassification.
    Returns a dataframe.
    """

    lrs_and_freq_df = likelihoods_and_frequencies_at_z(probs, reality, thresholds)

    lr_column_names = lrs_and_freq_df.columns.tolist()[:-1]

    return lrs_and_freq_df[lr_column_names]


def likelihoods_at_freq(probs, reality, thresholds=None):
    """
    Likelihood curves over thresholds. Compatible with binary or multiclassification.
    Returns a dataframe.
    """

    lrs_and_freq_df = likelihoods_and_frequencies_at_z(probs, reality, thresholds)

    return lrs_and_freq_df.set_index('Freq')


def likelihoods_and_frequencies_at_z(probs, reality, thresholds=None):
    """
    Likelihood curves over frequency of predictions. Compatible with binary or
    multiclassificaiton. Returns a dataframe. There's a bunch of shared code here with
    lr_at_z, possibly they could be refactored.
    """

    num_classes = probs.shape[1]
    class_ids = range(num_classes)
    lrs = defaultdict(lambda: [])
    num_preds = []

    if thresholds is None:
        thresholds = thresholding.default_thresholds_for_classes(num_classes)

    series_at_confidences = thresholding.threshold_predictions_at_many_confidences(
        probs,
        reality,
        thresholds,
    )

    for i in thresholds:
        for class_id in class_ids:
            lr_at_this_confidence = ml.stats.metrics.class_lr(
                np.array(series_at_confidences[i][0]),
                np.array(series_at_confidences[i][1]),
                class_id,
            )

            lrs[class_id].append(lr_at_this_confidence)

        num_preds.append(len(series_at_confidences[i][0]) / float(len(reality)))

    lrs_and_freq = {
        'LR%s' % class_id: lrs[class_id] for class_id in class_ids
    }

    lrs_and_freq.update({'Freq': num_preds})

    lrs_and_freq_df = pd.DataFrame(lrs_and_freq, index=thresholds)

    return lrs_and_freq_df

