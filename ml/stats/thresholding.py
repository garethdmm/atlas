"""
These functions take a probabilities list and convert it into predictions lists at
different probability thresholds. Can be quite computationally intensive so it is worth
optimizing every now and then.
"""

from collections import defaultdict
import numpy as np
import pandas as pd


def default_thresholds_for_classes(num_classes):
    if num_classes == 2:
        return np.arange(0.5, 0.9, 0.01)
    else:
        min_z = 1.0 / num_classes
        max_z = 2 * min_z

        return np.arange(min_z, max_z, 0.01)


def threshold_predictions_at_confidence(probabilities, reality, confidence):
    """
    Convert a list of probabilities and labels into a list of predictions and their
    corresponding labels only where the probability is greater than a certain threshold.
    """

    has_confidence = np.array([p.any() for p in probabilities > confidence])

    assert len(has_confidence) == len(probabilities) == len(reality)

    reality_where_confident = reality[has_confidence]

    confident_predictions = np.array(
        [np.argmax(p) for p in probabilities[has_confidence]]
    )

    return confident_predictions, reality_where_confident


def threshold_predictions_at_many_confidences(probabilities, reality, confidences):
    """
    Get many different predictions|realities-at-confidence lists at once, saving some
    performance in the process by batching it.

    This method supports multiclass and is very fast. We find the best p for each
    example once, then add it on to each confidence's array until we find a z it is
    less than, and then we break. 2-3x speedup over the old multiclass function, 33%
    speedup over the old binary function.
    """

    series_at_confidences = defaultdict(lambda: ([], []))

    for i, p in enumerate(probabilities):
        best_p_class = np.argmax(p)
        best_p = p[best_p_class]

        for confidence in confidences:
            if best_p > confidence:
                series_at_confidences[confidence][0].append(best_p_class)
                series_at_confidences[confidence][1].append(reality[i])
            else:
                # break out if we didn't find a satisfactory p here.
                break

    return series_at_confidences


# Old functions. Still useful for reference.

def _old_binary_threshold_predictions_at_many_confidences(probabilities, reality, confidences):
    """
    Get many different predictions|realities-at-confidence lists at once for a binary
    classifier. Currently not actually faster than the version that supports multiclass.
    """

    series_at_confidences = defaultdict(lambda: ([], []))
    confidences = sorted(confidences)

    for i, p in enumerate(probabilities):
        for c in confidences:
            if p[0] > c:
                series_at_confidences[c][0].append(0)
                series_at_confidences[c][1].append(reality[i])
            elif abs(1 - p[0]) > c:
                series_at_confidences[c][0].append(1)
                series_at_confidences[c][1].append(reality[i])
            else:
                break # Use this short circuiting to improve performance.

    return series_at_confidences


def _old_multiclass_threshold_predictions_at_many_confidences(probabilities, reality, confidences):
    """
    Get many different predictions|realities-at-confidence lists at once for a multi
    classifier. Just a for-loop over threshold_predictions_at_confidence. Useful as
    ground-truth and for benchmarking.
    """

    series_at_confidences = defaultdict(lambda: ([], []))

    for c in confidences:
        z_preds, z_rels = threshold_predictions_at_confidence(
            probabilities,
            reality,
            c,
        )

        series_at_confidences[c] = (z_preds, z_rels)

    return series_at_confidences

