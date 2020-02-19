"""
Overnight training to be done for Thursday 4 May 2017. Classifier results for inner
strength series, both USD and BTC denominations for all exchanges, with data rebalanced
to have an equal number of positive and negative examples by undersampling.
"""

import ml.data.preconfigured_feature_label_sets as featuresets
from ml.models.tfl_linear_classifier import TFLLinearClassifier as linc
from ml.models.tfl_dnn_classifier import TFLDNNClassifier as dnnc
from ml.infra.work_spec import WorkSpec
from ml.infra.work_unit import WorkUnit
from ml.infra.model_spec import ModelSpec


# Model fn's.
def linear(td):
    return linc(td.shape[1], n_classes=2)


def dnn_small(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1]])


def dnn_large(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1]*4])


def dnn_dropout(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1]*4], dropout=0.5)


def dnn_layers(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1], td.shape[1]])


# Model Specs.
model_specs = [
    ModelSpec('linear', linear, 100),
    ModelSpec('dnn_small', dnn_small, 30000),
    ModelSpec('dnn_large', dnn_large, 30000),
    ModelSpec('dnn_dropout', dnn_dropout, 75000),
    ModelSpec('dnn_layers', dnn_layers, 75000),
]


# Work Units.
bitstamp_price_diff_rebalanced = WorkUnit(
    'bitstamp_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

bitfinex_price_diff_rebalanced = WorkUnit(
    'bitfinex_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

coinbase_price_diff_rebalanced = WorkUnit(
    name='coinbase_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

itbit_price_diff_rebalanced = WorkUnit(
    'itbit_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

okcoin_price_diff_rebalanced = WorkUnit(
    'okcoin_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

# For future runs of gemini/itbit, add in the different train/test windows here.
gemini_price_diff_rebalanced = WorkUnit(
    'gemini_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

# Upon reflection, simple returns/price diff shouldn't matter for classification, but
# Including this here for now for symmetry.
bitstamp_simple_returns_rebalanced = WorkUnit(
    'bitstamp_simple_returns_rebalanced',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitstamp,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)

bitfinex_simple_returns_rebalanced = WorkUnit(
    'bitfinex_simple_returns_rebalanced',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs
)


pipeline_work_units = {
    0: [bitstamp_price_diff_rebalanced],
    1: [bitfinex_price_diff_rebalanced],
    2: [coinbase_price_diff_rebalanced],
    3: [itbit_price_diff_rebalanced],
    4: [okcoin_price_diff_rebalanced],
    5: [gemini_price_diff_rebalanced],
    6: [bitstamp_simple_returns_rebalanced],
    7: [bitfinex_simple_returns_rebalanced],
}

Spec = WorkSpec('may_4_2017', pipeline_work_units)

