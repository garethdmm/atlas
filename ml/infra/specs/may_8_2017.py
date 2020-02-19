"""
Overnight training to be done for Thursday 8 May 2017. Classifier results for inner
strength series, both USD and BTC denominations with data rebalanced to have an equal
number of positive and negative examples by undersampling.

This targets only okcoin and gemini, using train/validate windows specific to those
exchanges instead of our default windows, since these two exchanges have much less data
than the others.
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


def dnn_small(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1]])


def dnn_large(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1]*4])


def dnn_dropout(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[td.shape[1]*4], dropout=0.5)


def dnn_layers(td):
    return dnnc(td.shape[1], n_classes=2,hidden_layers=[td.shape[1], td.shape[1]])


# Model Specs.
model_specs = [
    ModelSpec('linear', linear, 100),
    ModelSpec('dnn_small', dnn_small, 30000),
    ModelSpec('dnn_large', dnn_large, 30000),
    ModelSpec('dnn_dropout', dnn_dropout, 75000),
    ModelSpec('dnn_layers', dnn_layers, 75000),
]


# Work Units.
okcoin_price_diff_rebalanced_good_window = WorkUnit(
    'okcoin_price_diff_rebalanced_good_window',
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

gemini_price_diff_rebalanced_good_window = WorkUnit(
    'gemini_price_diff_rebalanced_good_window',
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
    0: [
        okcoin_price_diff_rebalanced_good_window,
        gemini_price_diff_rebalanced_good_window,
    ],
}

Spec = WorkSpec('may_8_2017', pipeline_work_units)

