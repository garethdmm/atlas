"""
Let's put multiclassification aside and see if we can train good regression measures on
these multiclassification nets.
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
    return linr(td.shape[1])

def dnn1(td):
    return dnnr(td.shape[1], hidden_layers=[1])

def dnn2(td):
    return dnnr(td.shape[1], hidden_layers=[2])

def dnn3(td):
    return dnnr(td.shape[1], hidden_layers=[3])

def dnn6(td):
    return dnnr(td.shape[1], hidden_layers=[6])


# Model Specs.
model_specs = [
    ModelSpec('linear', linear, 100),
    ModelSpec('dnn1', dnn1, 20000),
    ModelSpec('dnn2', dnn2, 20000),
    ModelSpec('dnn3', dnn3, 20000),
    ModelSpec('dnn6', dnn6, 20000),
]


# Work Units.
bitstamp_reg = WorkUnit(
    'bitstamp_reg',
    featuresets.inner_1d_and_arb_bitstamp,
    featureset_params={
        'for_classification': False,
    },
    model_specs=model_specs,
)

# Bitfinex.
bitfinex_reg = WorkUnit(
    'bitfinex_reg',
    featuresets.inner_1d_and_arb_bitfinex,
    featureset_params={
        'for_classification': False,
    },
    model_specs=model_specs,
)

# Coinbase.
coinbase_reg = WorkUnit(
    'coinbase_reg',
    featuresets.inner_1d_and_arb_coinbase,
    featureset_params={
        'for_classification': False,
    },
    model_specs=model_specs,
)

# Itbit.
itbit_reg = WorkUnit(
    'itbit_reg',
    featuresets.inner_1d_and_arb_itbit,
    featureset_params={
        'for_classification': False,
    },
    model_specs=model_specs,
)

# OkCoin.
okcoin_reg = WorkUnit(
    'okcoin_reg',
    featuresets.inner_1d_and_arb_okcoin,
    featureset_params={
        'for_classification': False,
        'train_start': OKCOIN_TRAIN_START,
        'train_end': OKCOIN_TRAIN_END,
        'test_start': OKCOIN_TEST_START,
        'test_end': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

# Gemini.
gemini_reg = WorkUnit(
    'gemini_reg',
    featuresets.inner_1d_and_arb_gemini,
    featureset_params={
        'for_classification': False,
        'train_start': OKCOIN_TRAIN_START,
        'train_end': OKCOIN_TRAIN_END,
        'test_start': OKCOIN_TEST_START,
        'test_end': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)


# 5 WUs and 16 pipelines.
pipeline_work_units = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [bitstamp_reg],
    7: [bitfinex_reg],
    8: [coinbase_reg],
    9: [itbit_reg],
    10: [okcoin_reg],
    11: [gemini_reg],
    12: [],
    13: [],
    14: [],
    15: [],
}

Spec = WorkSpec('june_27_2017c', pipeline_work_units)

