"""
Library for working with imbalanced datasets, in particlar rebalancing them. This is
mostly an interface to the imblearn library right now and a few unimplemented functions.
"""

import imblearn.under_sampling
import imblearn.over_sampling
import pandas as pd


def undersample(feature_data, label_data):
    """
    Balance the number of examples of each class in a classification dataset by removing
    random examples in the majority class until the classes are the same size.

    This function does work with non binary classification.

    TODO:
    - implement ratio to balance to (and more complex logic for multi classification.)
    """

    sampler = imblearn.under_sampling.RandomUnderSampler(
        return_indices=True,
        replacement=False,  # We can play around with this but False is intuitive.
    )

    features_balanced, labels_balanced, indices = sampler.fit_sample(
        feature_data,
        label_data,
    )

    resampled_index = feature_data.index[indices]

    features_balanced = pd.DataFrame(features_balanced, index=resampled_index)
    labels_balanced = pd.Series(labels_balanced, index=resampled_index)

    # These come out of the imblearn library in the order they were selected, not index
    # order, so we re-sort them by their index here.
    features_balanced = features_balanced.sort_index()
    labels_balanced = labels_balanced.sort_index()

    return features_balanced, labels_balanced


def oversample(feature_data, label_data):
    """
    Balance the number of examples of each class in a classification dataset by removing
    random examples in the majority class until the classes are the same size.

    This function does work with non binary classification.

    This has the effect of losing our pristine index. If we want to preserve that
    assumption, we'll have to write our own oversampler, which wouldn't be too hard, but
    that won't fix the problem for synthetic sampling methods like ADASYN/SMOTE.

    TODO:
    - implement ratio to balance to (and more complex logic for multi classification.)
    """

    sampler = imblearn.over_sampling.RandomOverSampler()

    features_balanced, labels_balanced = sampler.fit_sample(
        feature_data,
        label_data,
    )

    features_balanced = pd.DataFrame(features_balanced)
    labels_balanced = pd.Series(labels_balanced)

    return features_balanced, labels_balanced


def smote(feature_data, label_data):
    """
    Rebalance using the SMOTE algorithm to generate synthetic training examples for
    minority class.
    """

    sampler = imblearn.over_sampling.SMOTE()

    features_balanced, labels_balanced = sampler.fit_sample(
        feature_data,
        label_data,
    )

    features_balanced = pd.DataFrame(features_balanced)
    labels_balanced = pd.Series(labels_balanced)

    return features_balanced, labels_balanced


def adasyn(feature_data, label_data):
    """
    Rebalance using the ADASYN algorithm to generate synthetic training examples for
    minority class. Imblearn 0.3.0dev0 supports multiclass.

    There is another library that appears to support mutliclass here:
    https://github.com/stavskal/ADASYN
    """

    sampler = imblearn.over_sampling.ADASYN()

    features_balanced, labels_balanced = sampler.fit_sample(
        feature_data,
        label_data,
    )

    features_balanced = pd.DataFrame(features_balanced)
    labels_balanced = pd.Series(labels_balanced)

    return features_balanced, labels_balanced


def up_and_down_sample(feature_data, label_data):
    """
    Over- and under-sample at the same time, to reduce the negative effects of each.

    One option would be to meet in the middle at x | len(maj) / x == x * len(min), or
    we can leave it parameterized.
    """
    raise NotImplementedError

