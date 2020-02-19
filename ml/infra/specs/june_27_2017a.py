"""
Overnight training for June 26th. Let's see if targeting 5th/95th percentiles is
different than deciles.
"""

import ml.data.preconfigured_feature_label_sets as featuresets
from ml.defaults import *
from ml.models.tfl_linear_classifier import TFLLinearClassifier as linc
from ml.models.tfl_dnn_classifier import TFLDNNClassifier as dnnc
from ml.infra.work_spec import WorkSpec
from ml.infra.work_unit import WorkUnit
from ml.infra.model_spec import ModelSpec


# Model fn's.
def linear(td):
    return linc(td.shape[1], n_classes=3)

def dnn1(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[1])

def dnn2(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[2])

def dnn3(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[3])

def dnn6(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[6])


# Model Specs.
model_specs = [
    ModelSpec('linear', linear, 1000),
    ModelSpec('dnn1', dnn1, 20000),
    ModelSpec('dnn2', dnn2, 20000),
    ModelSpec('dnn3', dnn3, 20000),
    ModelSpec('dnn6', dnn6, 20000),
]


# Work Units.
bitstamp_tails_oversample = WorkUnit(
    'bitstamp_tails_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.5, 0.95],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Bitfinex.
bitfinex_tails_oversample = WorkUnit(
    'bitfinex_tails_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.5, 0.95],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Coinbase.
coinbase_tails_oversample = WorkUnit(
    'coinbase_tails_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.5, 0.95],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Itbit.
itbit_tails_oversample = WorkUnit(
    'itbit_tails_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.5, 0.95],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# OkCoin.
okcoin_tails_oversample = WorkUnit(
    'okcoin_tails_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.5, 0.95],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Gemini.
gemini_tails_oversample = WorkUnit(
    'gemini_tails_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.5, 0.95],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)


# 6 WUs and 8 pipelines.
pipeline_work_units = {
    0: [
        bitstamp_tails_oversample,
    ],
    1: [
        bitfinex_tails_oversample,
    ],
    2: [
        coinbase_tails_oversample,
    ],
    3: [
        itbit_tails_oversample,
    ],
    4: [
        okcoin_tails_oversample,
    ],
    5: [
        gemini_tails_oversample,
    ],
    6: [
    ],
    7: [
    ],
}

Spec = WorkSpec('june_27_2017a', pipeline_work_units)

