"""
A small library for doing logical operations on pandas series in the presence of
np.nan's. It turns out that the default behaviour for these operations implemented in
pandas/numpy does not match my intuition, so I've reimplemented these operations here
to match my intuition, and therefore, universal truth.

The crux is just that any operation involving an np.nan returns an np.nan. Numpy
default behaviour is for np.nan > int == False, which is dumb and I hate them.

This uses classes for namespacing, but whatever.
"""

import pandas as pd
import numpy as np


class ops(object):
    @staticmethod
    def greater_than(a, b):
        return np.nan if np.isnan(a) else a > b

    @staticmethod
    def greater_than_or_equal(a, b):
        return np.nan if np.isnan(a) else a >= b

    @staticmethod
    def less_than(a, b):
        return np.nan if np.isnan(a) else a < b

    @staticmethod
    def less_than_or_equal(a, b):
        return np.nan if np.isnan(a) else a <= b

    @staticmethod
    def logical_and(a, b):
        return np.nan if (np.isnan(a) or np.isnan(b)) else a and b


class series_ops(object):
    @staticmethod
    def greater_than(series, val):
        return series.apply(lambda x: ops.greater_than(x, val))

    @staticmethod
    def less_than(series, val):
        return series.apply(lambda x: ops.less_than(x, val))

    @staticmethod
    def greater_than_or_equal(series, val):
        return series.apply(lambda x: ops.greater_than_or_equal(x, val))

    @staticmethod
    def less_than_or_equal(series, val):
        return series.apply(lambda x: ops.less_than_or_equal(x, val))

    @staticmethod
    def logical_and(series_a, series_b):
        return pd.concat([series_a, series_b], axis=1).T\
            .apply(lambda x: ops.logical_and(x[0], x[1]))

