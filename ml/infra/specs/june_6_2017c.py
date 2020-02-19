"""
Overnight training to be done 6 June 2017.

This is a re-run of our may 15th results, but using the 'drop' handle_boundary
behaviour, in lieu of the over-negative bug.

C version uses oversampling instead of undersampling.
"""

import ml.data.conversion as conversion
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
    ModelSpec('dnn1', dnn1, 5000),
    ModelSpec('dnn2', dnn2, 5000),
    ModelSpec('dnn3', dnn3, 5000),
    ModelSpec('dnn6', dnn6, 5000),
]


# Work Units.
bitstamp_price_diff_rebalanced = WorkUnit(
    'bitstamp_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'rebalance_method': 'oversample',
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

bitfinex_price_diff_rebalanced = WorkUnit(
    'bitfinex_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'rebalance_method': 'oversample',
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

coinbase_price_diff_rebalanced = WorkUnit(
    'coinbase_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'rebalance_method': 'oversample',
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

itbit_price_diff_rebalanced = WorkUnit(
    'itbit_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'rebalance_method': 'oversample',
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

okcoin_price_diff_rebalanced = WorkUnit(
    'okcoin_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'rebalance_method': 'oversample',
        'handle_boundary': conversion.BOUNDARY_DROP,
        'train_start': OKCOIN_TRAIN_START,
        'train_end': OKCOIN_TRAIN_END,
        'test_start': OKCOIN_TEST_START,
        'test_end': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

gemini_price_diff_rebalanced = WorkUnit(
    'gemini_price_diff_rebalanced',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'rebalance_method': 'oversample',
        'handle_boundary': conversion.BOUNDARY_DROP,
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)


# Pipelines.

pipeline_work_units = {
    0: [bitstamp_price_diff_rebalanced],
    1: [bitfinex_price_diff_rebalanced],
    2: [coinbase_price_diff_rebalanced],
    3: [itbit_price_diff_rebalanced],
    5: [],
    6: [],
    6: [okcoin_price_diff_rebalanced],
    7: [gemini_price_diff_rebalanced],
}

Spec = WorkSpec('june_6_2017c', pipeline_work_units)

