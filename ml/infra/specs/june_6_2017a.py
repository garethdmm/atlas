"""
Overnight training to be done 6 June 2017, part A.

This is a rerun of our May 31st overnight results, due to our discovery of the over-
negative bug.

Initial investigations into training using arbitrage feature data.

- All exchanges.
- Regression and binary classification.
- Simple undersampling for classification.
- Small number of epochs to start.

50 Models.
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
    ModelSpec('dnn1', dnn1, 10000),
    ModelSpec('dnn2', dnn2, 10000),
    ModelSpec('dnn3', dnn3, 10000),
    ModelSpec('dnn6', dnn6, 10000),
]


# Work Units.

bitstamp_arb_rev_class = WorkUnit(
    'bitstamp_arb_rev_class',
    featuresets.bitstamp_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

bitfinex_arb_rev_class = WorkUnit(
    'bitfinex_arb_rev_class',
    featuresets.bitfinex_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

coinbase_arb_rev_class = WorkUnit(
    'coinbase_arb_rev_class',
    featuresets.coinbase_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

itbit_arb_rev_class = WorkUnit(
    'itbit_arb_rev_class',
    featuresets.itbit_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'handle_boundary': conversion.BOUNDARY_DROP,
    },
    model_specs=model_specs,
)

okcoin_arb_rev_class = WorkUnit(
    'okcoin_arb_rev_class',
    featuresets.okcoin_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'train_start': OKCOIN_TRAIN_START,
        'train_end': OKCOIN_TRAIN_END,
        'test_start': OKCOIN_TEST_START,
        'test_end': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

gemini_arb_rev_class = WorkUnit(
    'gemini_arb_rev_class',
    featuresets.gemini_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)


# Pipelines.

pipeline_work_units = {
    0: [bitstamp_arb_rev_class],
    1: [bitfinex_arb_rev_class],
    2: [okcoin_arb_rev_class],
    3: [coinbase_arb_rev_class],
    4: [itbit_arb_rev_class],
    5: [gemini_arb_rev_class],
}

Spec = WorkSpec('june_6a_2017', pipeline_work_units)

