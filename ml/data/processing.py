"""
Commonly used functions for massaging/formatting data before we send it into a model go
in here.
"""

import pandas as pd


def quick_past_returns_series(series, lookback, tick=1):
    """
    Given a (log, not simple) returns series (close, log or normal), generates a new series that
    is the return over the past [lookback] hours. The resulting series is of size
    |series| - lookback.

    e.g. if your last three hourly returns are 1, -5, 3, your current three hour
    past return is -1.

    Returns a pandas series.

    TODO: add asserts on the characteristics of these series.
    """

    new_series = pd.rolling_sum(series, window=lookback)[lookback-1:][::tick]

    return new_series


def past_returns_series(series, lookback, tick=1):
    """
    Given an hourly returns (close, log or normal) series, generates a new series that
    is the return over the past [lookback] hours. The resulting series is of size
    |series| - lookback.

    e.g. if your last three hourly returns are 1, -5, 3, your current three hour
    past return is -1.

    Returns a pandas series.
    """

    start = lookback - 1
    past_returns = []
    ts = []

    """
    If this is a close-close series, the return at hour i is the past 1hr return.
    The past 2 hour return at hour i is rets[i] + rets[i-1]. The past 3 hour return
    is rets[i] + rets[i - 1] + rets[i - 2].
    """
    for i in range(start, len(series), tick):
        t = series.index[i].to_datetime()

        past_return = 0

        for j in range(0, lookback):
            past_return += series[i-j]

        past_returns.append(past_return)
        ts.append(t)

    new_series = pd.Series(past_returns, index=ts)

    return new_series


def future_returns_series(series, lookforward, start=0, tick=1):
    """
    Given an close-close hourly log return series generates the [lookforward]-hour
    future return series.
    """
    future_returns = []
    ts = []

    for i in range(start, len(series) - lookforward, tick):
        t = series.index[i].to_datetime()

        future_return = 0

        for j in range(0, lookforward):
            # We add 1 because these are close prices, the first future return is
            # at i + 1
            future_return += series[i+1+j]

        future_returns.append(future_return)
        ts.append(t)

    new_series = pd.Series(future_returns, index=ts)

    return new_series


def future_return_signs_series(series, lookforward, start=0, tick=1):
    """
    Constructs a series such that ret[i] = True iff the lookforward-period future
    return at time i is positive and False otherwise.

    Do you have to worry about overlapping observations in this model?

    It's probably safest not to use them.

    e.g. Mnist has no analog of overlapping observations, one picture doesn't include
    parts of the previous picture.
    """

    future_returns = future_returns_series(series, lookforward, start, tick)

    future_return_signs = (future_returns > 0) * 1

    return future_return_signs


def rescale_feature(feature):
    mean = feature.mean()
    std = feature.std()

    series = (feature - mean) / std

    return series, mean, std


