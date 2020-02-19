"""
Examining whether adding arb revenue features into the featureset improves on our
results from june 20th. Deciles only to start.

Follow-ons from this:
- try 5/95-iles instead of deciles
- try ensemble of a train of both nets
- try larger nets (we have one of these, did it show better results)?
  - or try dropout
- add arb volume usd as well into the featureset
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
bitstamp_deciles_oversample = WorkUnit(
    'bitstamp_deciles_oversample',
    featuresets.inner_1d_and_arb_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Bitfinex.
bitfinex_deciles_oversample = WorkUnit(
    'bitfinex_deciles_oversample',
    featuresets.inner_1d_and_arb_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Coinbase.
coinbase_deciles_oversample = WorkUnit(
    'coinbase_deciles_oversample',
    featuresets.inner_1d_and_arb_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Itbit.
itbit_deciles_oversample = WorkUnit(
    'itbit_deciles_oversample',
    featuresets.inner_1d_and_arb_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# OkCoin.
okcoin_deciles_oversample = WorkUnit(
    'okcoin_deciles_oversample',
    featuresets.inner_1d_and_arb_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'train_start': OKCOIN_TRAIN_START,
        'train_end': OKCOIN_TRAIN_END,
        'test_start': OKCOIN_TEST_START,
        'test_end': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

# Gemini.
gemini_deciles_oversample = WorkUnit(
    'gemini_deciles_oversample',
    featuresets.inner_1d_and_arb_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)


# 12 WUs and 8 pipelines.
pipeline_work_units = {
    0: [bitstamp_deciles_oversample],
    1: [bitfinex_deciles_oversample],
    2: [coinbase_deciles_oversample],
    3: [itbit_deciles_oversample],
    4: [okcoin_deciles_oversample],
    5: [gemini_deciles_oversample],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
    12: [],
    13: [],
    14: [],
    15: [],
}

Spec = WorkSpec('june_27_2017b', pipeline_work_units)

