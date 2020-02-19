"""
A set of commonly shared functions between models go in here.
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import tensorflow as tf

import ml.data.feature as feature
import ml.data.conversion as conversion


def regression_backtest(db, model, feature_data, price_feature):
    # Get the price data.
    idx = feature_data.index
    start = idx[0]
    end = idx[-1]

    price_data = price_feature.get_data(db, start, end)

    # Get the predictions.
    #   - we don't have a public get_predictions function right now so hack around it.
    feature_data = model.vectorize(feature_data)
    predictions = model.get_predictions_on_dataset(feature_data)
    predictions = conversion.decimal_labels_to_binary_categorical(predictions)

    # Get the p&l from the prices and predictions
    preds, prices, diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        price_data,
        idx,
    )

    return preds, prices, diffs, realized


def backtest(db, model, feature_data, price_feature, confidence=0.5):
    # Get the price data.
    idx = feature_data.index
    start = idx[0]
    end = idx[-1]

    price_data = price_feature.get_data(db, start, end)

    # Get the predictions.
    #   - we don't have a public get_predictions function right now so hack around it.
    feature_data = model.vectorize(feature_data)
    probabilities = model.get_probabilities_on_dataset(feature_data)

    predictions, new_idx = threshold_predictions_at_confidence_with_idx(
        probabilities,
        confidence,
        idx,
    )
        
    # Get the p&l from the prices and predictions
    preds, prices, diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        price_data,
        new_idx,
    )

    return preds, prices, diffs, realized


def get_pl_from_predictions_and_prices(predictions, prices, index):
    """
    I think you also need to reindex the positions array here, and then output
    what the loss of points was
    """
    price_diffs = convert_prices_to_price_diffs(prices)
    price_diffs = price_diffs.reindex(index)

    positions = convert_predictions_array_to_positions(predictions)

    realized = predictions * price_diffs

    return predictions, prices, price_diffs, realized


def threshold_predictions_at_confidence_with_idx(probabilities, confidence, idx):
    """This idx stuff is kinda gross"""

    has_confidence = np.array([p.any() for p in probabilities > confidence])

    assert len(has_confidence) == len(probabilities)

    new_idx = idx[has_confidence]

    confident_predictions = np.array(
        [np.argmax(p) for p in probabilities[has_confidence]]
    )

    return confident_predictions, new_idx


def convert_prices_to_price_diffs(prices):
    """This could easily be part of the Price() feature"""
    return (prices - prices.shift(1)).shift(-1)


def convert_predictions_array_to_positions(predictions):
    positions = np.array([p if p == 1 else -1 for p in predictions])

    return positions


def test_preds_prices():
    # Should return -1 + 1 -1 == -1

    index = pd.date_range('2015-1-1', '2015-1-1 3:00', freq='H')
    predictions = pd.Series([0, 1, 0, 1], index=index)
    prices = pd.Series([1, 2, 3, 4], index=index)

    preds, prices, price_diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        prices,
        index,
    )

    assert realized.sum() == -1

    # Should return 1 + 1 + 1 == 3

    index = pd.date_range('2015-1-1', '2015-1-1 3:00', freq='H')
    predictions = pd.Series([1, 1, 1, 1], index=index)
    prices = pd.Series([1, 2, 3, 4], index=index)

    preds, prices, price_diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        prices,
        index,
    )

    assert realized.sum() == 3

    # Should return -1 -1 -1 == -3

    index = pd.date_range('2015-1-1', '2015-1-1 3:00', freq='H')
    predictions = pd.Series([-1, -1, -1, -1], index=index)
    prices = pd.Series([1, 2, 3, 4], index=index)

    preds, prices, price_diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        prices,
        index,
    )

    assert realized.sum() == -3

    # Should return - 1 - 11 + 5

    index = pd.date_range('2015-1-1', '2015-1-1 3:00', freq='H')
    predictions = pd.Series([-1, 1, 1, -1], index=index)
    prices = pd.Series([200, 201, 190, 195], index=index)

    preds, prices, price_diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        prices,
        index,
    )

    assert realized.sum() == -7

    index = pd.date_range('2015-1-1', '2015-1-1 3:00', freq='H')
    predictions = pd.Series([-1, 1, 1, -1], index=index)
    prices = pd.Series([200, 201, 190, 195], index=index)

    preds, prices, price_diffs, realized = get_pl_from_predictions_and_prices(
        predictions,
        prices,
        index,
    )

    assert realized.sum() == -7

