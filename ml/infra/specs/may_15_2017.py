"""
Overnight training to be done for Wed 9 May 2017. Examining performance of smaller nets
on the inner strength USD series' with balancing by undersampling. Classification only.
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
    return linc(td.shape[1], n_classes=2)

def dnn1(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[1])

def dnn2(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[2])

def dnn3(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[3])

def dnn6(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[6])


# Model Specs.
model_specs = [
    ModelSpec('linear', linear, 100),
    ModelSpec('dnn1', dnn1, 10000),
    ModelSpec('dnn2', dnn2, 10000),
    ModelSpec('dnn3', dnn3, 10000),
    ModelSpec('dnn6', dnn6, 10000),
]


# Work Units.
bitstamp_price_diff_rebalanced = WorkUnit(
    'bitstamp_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs,
)

bitfinex_price_diff_rebalanced = WorkUnit(
    'bitfinex_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs,
)

coinbase_price_diff_rebalanced = WorkUnit(
    name='coinbase_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs,
)

itbit_price_diff_rebalanced = WorkUnit(
    'itbit_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=model_specs,
)

okcoin_price_diff_rebalanced = WorkUnit(
    'okcoin_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

gemini_price_diff_rebalanced = WorkUnit(
    'gemini_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)


pipeline_work_units = {
    0: [bitstamp_price_diff_rebalanced],
    1: [bitfinex_price_diff_rebalanced],
    2: [coinbase_price_diff_rebalanced],
    3: [itbit_price_diff_rebalanced],
    4: [okcoin_price_diff_rebalanced],
    5: [gemini_price_diff_rebalanced],
}

Spec = WorkSpec('may_15_2017', pipeline_work_units)

