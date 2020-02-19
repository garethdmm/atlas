"""
Libary for doing a variety of type or format conversions. Most have to do with
conversion into/out of classification series.
"""

import numpy as np
import pandas as pd


BOUNDARY_ASSIGN_LEFT = 'assign_left'
BOUNDARY_ASSIGN_RIGHT = 'assign_right'
BOUNDARY_ASSIGN_RANDOM = 'assign_random'
BOUNDARY_DROP = 'drop'


def decimal_labels_to_categorical_by_boundaries(labels, boundaries, handle_boundary=None):
    """
    Convert a series into categorical along the given boundaries with ascending integer
    category labels starting with 0. It does this by applying the categorize function to
    every example in 'labels'. Returns a pandas series of dtype categorical.

    A couple details here. The "boundaries" input is a list of n non-infinite real
    numbers. We sort these boundaries after the come in, since categorize() is ill-
    defined without that guarantee.
    """

    boundaries = sorted(boundaries)
    categories = range(len(boundaries) + 1)
    boundaries = [-np.inf] + boundaries + [np.inf]

    class_and_boundaries = {i: (boundaries[i], boundaries[i + 1]) for i in categories}

    vals = labels.apply(lambda x: categorize(x, class_and_boundaries, handle_boundary))

    output_series = vals.astype('category', categories=categories)

    return output_series


def categorize(value, class_and_boundaries, handle_boundary=None):
    """
    class_and_boundaries is a dict of class names (integers) mapped to the left/right
    boundaries of that class. This function determines which of those classes a given
    datapoint belongs to.

    We have to be pretty smart about what we do on the boundaries here, especially since
    our most common boundary is 0, which is, surprisingly, a frequent occurrence in our
    1m series'.
    """

    if np.isnan(value):
        return np.nan

    for category_id, boundaries in class_and_boundaries.items():
        left_boundary = boundaries[0]
        right_boundary = boundaries[1]

        if value == left_boundary:
            if handle_boundary is None or handle_boundary == BOUNDARY_ASSIGN_LEFT:
                return category_id - 1
            elif handle_boundary == BOUNDARY_ASSIGN_RIGHT:
                return category_id
            elif handle_boundary == BOUNDARY_ASSIGN_RANDOM:
                use_left = (np.random.randint(0, 2) == 0)

                if use_left:
                    return category_id - 1
                else:
                    return category_id
            elif handle_boundary == BOUNDARY_DROP:
                return np.nan

        if value == right_boundary:
            if handle_boundary is None or handle_boundary == BOUNDARY_ASSIGN_LEFT:
                return category_id
            elif handle_boundary == BOUNDARY_ASSIGN_RIGHT:
                return category_id + 1
            elif handle_boundary == BOUNDARY_ASSIGN_RANDOM:
                use_left = (np.random.randint(0, 2) == 0)

                if use_left:
                    return category_id
                else:
                    return category_id + 1
            elif handle_boundary == BOUNDARY_DROP:
                return np.nan

        if value > left_boundary and value < right_boundary:
            return category_id

    raise ValueError('%s is not in any category!' % str(value))


def decimal_labels_to_categorical_by_percentile(labels, percentiles, handle_boundary=None):
    """
    Convert a series into categorical with percentile boundaries.
    """
    boundaries = [np.percentile(labels.dropna(), p * 100) for p in percentiles]

    return decimal_labels_to_categorical_by_boundaries(
        labels,
        boundaries,
        handle_boundary,
    )


def decimal_labels_to_binary_categorical(labels, handle_boundary=None):
    """
    In: real-valued labels in a numpy array or pandas series of real numbers.
    Out: a numpy array or pandas series of the form [0, 1, 1, 0, ...], whichever the
    input was.

    NaN behaviour: leave the nan's in the series.
    """
    return decimal_labels_to_categorical_by_boundaries(labels, [0], handle_boundary)


def categorical_labels_to_one_hot(labels, n_classes=2):
    """
    In: a numpy array or pandas series of the form [0, 1, 1, 0, ...]
    Out: a numpy array of the form [[1,0], [0,1], [0,1], [1,0], ...]
    (Above translation is accurate).

    Nan behaviour: leave it in the series
    """
    return np.eye(n_classes)[labels]


def one_hot_labels_to_categorical(labels):
    """
    In: a numpy array of the form [[1,0], [0,1], [0,1], [1,0], ...]
    Out: a numpy array or pandas series of the form [0, 1, 1, 0, ...]
    (Above translation is accurate).
    """
    return np.asarray(pd.DataFrame(labels).T.idxmax())


def categorical_to_decimal(labels):
    return np.asarray(labels)


# Legacy, kept here for testing.

def _old_decimal_labels_to_binary_categorical(labels):
    """
    In: real-valued labels in a numpy array or pandas series of real numbers.
    Out: a numpy array or pandas series of the form [0, 1, 1, 0, ...], whichever the
    input was.

    NaN behaviour: leave the nan's in the series.

    You can replicate the behaviour of this by passing in BOUNDARY_ASSIGN_LEFT to
    decimal_labels_to_binary_categorical.
    """
    return (labels > 0) * 1

