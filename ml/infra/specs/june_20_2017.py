"""
Overnight training for June 20th. Lets see if using simple returns improves our
multiclassification performance on different percentiles. Idea being that the average
dollar size of moves over time might change, but percentage moves probably will not,
so simple returns might be a better metric to split these up into different classes by.

- All exchanges.
- Small network sizes.
- 3 class classification
  - with borders at 0.33, 0.67, oversampled
  - with borders at 0.10, 0.90, oversampled

12 models.
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
bitstamp_thirds_oversample = WorkUnit(
    'bitstamp_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

bitstamp_deciles_oversample = WorkUnit(
    'bitstamp_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Bitfinex.

bitfinex_thirds_oversample = WorkUnit(
    'bitfinex_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

bitfinex_deciles_oversample = WorkUnit(
    'bitfinex_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Coinbase.

coinbase_thirds_oversample = WorkUnit(
    'coinbase_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

coinbase_deciles_oversample = WorkUnit(
    'coinbase_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Itbit.

itbit_thirds_oversample = WorkUnit(
    'itbit_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

itbit_deciles_oversample = WorkUnit(
    'itbit_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# OkCoin.

okcoin_thirds_oversample = WorkUnit(
    'okcoin_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_deciles_oversample = WorkUnit(
    'okcoin_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)


# Gemini.

gemini_thirds_oversample = WorkUnit(
    'gemini_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_deciles_oversample = WorkUnit(
    'gemini_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_simple_returns_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

# 12 WUs and 8 pipelines.
pipeline_work_units = {
    0: [
        bitstamp_thirds_oversample,
    ],
    1: [
        bitstamp_deciles_oversample,
    ],
    2: [
        bitfinex_thirds_oversample,
    ],
    3: [
        bitfinex_deciles_oversample,
    ],
    4: [
        coinbase_thirds_oversample,
        coinbase_deciles_oversample,
    ],
    5: [
        itbit_thirds_oversample,
        itbit_deciles_oversample,
    ],
    6: [
        okcoin_thirds_oversample,
        okcoin_deciles_oversample,
    ],
    7: [
        gemini_thirds_oversample,
        gemini_deciles_oversample,
    ],
}

Spec = WorkSpec('june_20_2017', pipeline_work_units)

