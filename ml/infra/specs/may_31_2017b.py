"""
Initial investigations into training using arbitrage feature data.

- All exchanges.
- Regression and binary classification.
- Simple undersampling for classification.
- Small number of epochs to start.

- Updated, took out okcoin and gemini because they were messing up too much data.
"""

import ml.data.preconfigured_feature_label_sets as featuresets
from ml.defaults import *
from ml.models.tfl_linear_classifier import TFLLinearClassifier as linc
from ml.models.tfl_dnn_classifier import TFLDNNClassifier as dnnc
from ml.models.tfl_linear_regression import TFLLinearRegressor as linr
from ml.models.tfl_dnn_regressor import TFLDNNRegressor as dnnr
from ml.infra.work_spec import WorkSpec
from ml.infra.work_unit import WorkUnit
from ml.infra.model_spec import ModelSpec


# Classification models.
def linear_class(td):
    return linc(td.shape[1], n_classes=2)

def dnn1_class(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[1])

def dnn2_class(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[2])

def dnn3_class(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[3])

def dnn6_class(td):
    return dnnc(td.shape[1], n_classes=2, hidden_layers=[6])


classification_model_specs = [
    ModelSpec('linear', linear_class, 100),
    ModelSpec('dnn1', dnn1_class, 5000),
    ModelSpec('dnn2', dnn2_class, 5000),
    ModelSpec('dnn3', dnn3_class, 5000),
    ModelSpec('dnn6', dnn6_class, 5000),
]


# Regression models.

def linear_reg(td):
    return linr(td.shape[1])

def dnn1_reg(td):
    return dnnr(td.shape[1], hidden_layers=[1])

def dnn2_reg(td):
    return dnnr(td.shape[1], hidden_layers=[2])

def dnn3_reg(td):
    return dnnr(td.shape[1], hidden_layers=[3])

def dnn6_reg(td):
    return dnnr(td.shape[1], hidden_layers=[6])


regression_model_specs = [
    ModelSpec('linear', linear_reg, 100),
    ModelSpec('dnn1', dnn1_reg, 5000),
    ModelSpec('dnn2', dnn2_reg, 5000),
    ModelSpec('dnn3', dnn3_reg, 5000),
    ModelSpec('dnn6', dnn6_reg, 5000),
]


# Work Units.
bitstamp_arb_rev_class = WorkUnit(
    'bitstamp_arb_rev_class',
    featuresets.bitstamp_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=classification_model_specs,
)

bitfinex_arb_rev_class = WorkUnit(
    'bitfinex_arb_rev_class',
    featuresets.bitfinex_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=classification_model_specs,
)

coinbase_arb_rev_class = WorkUnit(
    'coinbase_arb_rev_class',
    featuresets.coinbase_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=classification_model_specs,
)

itbit_arb_rev_class = WorkUnit(
    'itbit_arb_rev_class',
    featuresets.itbit_arb_rev,
    featureset_params={
        'for_classification': True,
        'rebalance': True,
    },
    model_specs=classification_model_specs,
)

# Regression model specs
bitstamp_arb_rev_reg = WorkUnit(
    'bitstamp_arb_rev_reg',
    featuresets.bitstamp_arb_rev,
    featureset_params={},
    model_specs=regression_model_specs,
)

bitfinex_arb_rev_reg = WorkUnit(
    'bitfinex_arb_rev_reg',
    featuresets.bitfinex_arb_rev,
    featureset_params={},
    model_specs=regression_model_specs,
)

coinbase_arb_rev_reg = WorkUnit(
    'coinbase_arb_rev_reg',
    featuresets.coinbase_arb_rev,
    featureset_params={},
    model_specs=regression_model_specs,
)

itbit_arb_rev_reg = WorkUnit(
    'itbit_arb_rev_reg',
    featuresets.itbit_arb_rev,
    featureset_params={},
    model_specs=regression_model_specs,
)


pipeline_work_units = {
    0: [
        bitstamp_arb_rev_class,
        bitstamp_arb_rev_reg,
    ],
    1: [
        bitfinex_arb_rev_class,
        bitfinex_arb_rev_reg,
    ],
    2: [
        coinbase_arb_rev_class,
        coinbase_arb_rev_reg,
    ],
    3: [
        itbit_arb_rev_class,
        itbit_arb_rev_reg,
    ],
}

Spec = WorkSpec('may_31_2017b', pipeline_work_units)

