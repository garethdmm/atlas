"""
Training for June 21st. Examining different successes of over, SMOTE, and ADASYN
rebalancing methods on multiclassification of deciles.
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
    ModelSpec('dnn1', dnn1, 10000),
    ModelSpec('dnn2', dnn2, 10000),
    ModelSpec('dnn3', dnn3, 10000),
    ModelSpec('dnn6', dnn6, 10000),
]


# Work Units.
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

bitstamp_deciles_adasyn = WorkUnit(
    'bitstamp_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'adasyn'
    },
    model_specs=model_specs,
)

bitstamp_deciles_smote = WorkUnit(
    'bitstamp_deciles_smote',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'smote'
    },
    model_specs=model_specs,
)

# Bitfinex.
bitfinex_deciles_oversample = WorkUnit(
    'bitfinex_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

bitfinex_deciles_adasyn = WorkUnit(
    'bitfinex_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'adasyn'
    },
    model_specs=model_specs,
)

bitfinex_deciles_smote = WorkUnit(
    'bitfinex_deciles_smote',
    featuresets.ultra_strength_inner_1d_target_simple_returns_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'smote'
    },
    model_specs=model_specs,
)

# Coinbase.
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

coinbase_deciles_adasyn = WorkUnit(
    'coinbase_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'adasyn'
    },
    model_specs=model_specs,
)

coinbase_deciles_smote = WorkUnit(
    'coinbase_deciles_smote',
    featuresets.ultra_strength_inner_1d_target_simple_returns_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'smote'
    },
    model_specs=model_specs,
)

# Itbit.
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

itbit_deciles_adasyn = WorkUnit(
    'itbit_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'adasyn'
    },
    model_specs=model_specs,
)

itbit_deciles_smote = WorkUnit(
    'itbit_deciles_smote',
    featuresets.ultra_strength_inner_1d_target_simple_returns_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'smote'
    },
    model_specs=model_specs,
)

# OkCoin.
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

okcoin_deciles_adasyn = WorkUnit(
    'okcoin_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'adasyn',
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_deciles_smote = WorkUnit(
    'okcoin_deciles_smote',
    featuresets.ultra_strength_inner_1d_target_simple_returns_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'smote',
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
    featuresets.ultra_strength_inner_1d_target_simple_returns_gemini,
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

gemini_deciles_adasyn = WorkUnit(
    'gemini_deciles_adasyn',
    featuresets.ultra_strength_inner_1d_target_simple_returns_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'adasyn',
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_deciles_smote = WorkUnit(
    'gemini_deciles_smote',
    featuresets.ultra_strength_inner_1d_target_simple_returns_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'smote',
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)


# 18 WUs, 54 models
pipeline_work_units = {
    0: [
        bitstamp_deciles_oversample,
        bitstamp_deciles_adasyn,
    ],
    1: [
        bitstamp_deciles_smote,
        bitfinex_deciles_oversample,
    ],
    2: [
        bitfinex_deciles_adasyn,
        bitfinex_deciles_smote,
    ],
    3: [
        coinbase_deciles_oversample,
        coinbase_deciles_adasyn,
    ],
    4: [
        coinbase_deciles_smote,
        itbit_deciles_oversample,
    ],
    5: [
        itbit_deciles_adasyn,
        itbit_deciles_smote,
    ],
    6: [
        okcoin_deciles_oversample,
        okcoin_deciles_adasyn,
        okcoin_deciles_smote,
    ],
    7: [
        gemini_deciles_oversample,
        gemini_deciles_adasyn,
        gemini_deciles_smote,
    ],
}

Spec = WorkSpec('june_21_2017', pipeline_work_units)

