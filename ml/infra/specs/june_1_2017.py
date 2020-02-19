"""
Overnight training for June 1st. Expanding multiclassification investigation.

- All exchanges
- Split into thirds
- Under/oversampled
- Larger model sizes, two layer models, longer training times.
"""

import ml.data.preconfigured_feature_label_sets as featuresets
from ml.defaults import *
from ml.models.tfl_linear_classifier import TFLLinearClassifier as linc
from ml.models.tfl_dnn_classifier import TFLDNNClassifier as dnnc
from ml.infra.work_spec import WorkSpec
from ml.infra.work_unit import WorkUnit
from ml.infra.model_spec import ModelSpec


# Model fn's.
def dnn3(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[3])

def dnn6(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[6])

def dnn33(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[3, 3])

def dnn66(td):
    return dnnc(td.shape[1], n_classes=3, hidden_layers=[6, 6])


# Model Specs.
model_specs = [
    ModelSpec('dnn3', dnn3, 30000),
    ModelSpec('dnn6', dnn6, 30000),
    ModelSpec('dnn33', dnn33, 30000),
    ModelSpec('dnn66', dnn66, 30000),
]


# Work Units.
bitstamp_thirds_undersample = WorkUnit(
    'bitstamp_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

bitstamp_thirds_oversample = WorkUnit(
    'bitstamp_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Bitfinex.

bitfinex_thirds_undersample = WorkUnit(
    'bitfinex_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

bitfinex_thirds_oversample = WorkUnit(
    'bitfinex_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Coinbase.

coinbase_thirds_undersample = WorkUnit(
    'coinbase_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

coinbase_thirds_oversample = WorkUnit(
    'coinbase_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Itbit.

itbit_thirds_undersample = WorkUnit(
    'itbit_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

itbit_thirds_oversample = WorkUnit(
    'itbit_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# OkCoin.

okcoin_thirds_undersample = WorkUnit(
    'okcoin_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
        'train_start': OKCOIN_TRAIN_START,
        'train_end': OKCOIN_TRAIN_END,
        'test_start': OKCOIN_TEST_START,
        'test_end': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_thirds_oversample = WorkUnit(
    'okcoin_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
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

gemini_thirds_undersample = WorkUnit(
    'gemini_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_thirds_oversample = WorkUnit(
    'gemini_thirds_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'oversample',
        'train_start': GEMINI_TRAIN_START,
        'train_end': GEMINI_TRAIN_END,
        'test_start': GEMINI_TEST_START,
        'test_end': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)


# 12 Wu's, 4 model types, 48 models, 5.7m epochs.
# Gave bfx/stmp their own lanes because their datasets are the largest.
pipeline_work_units = {
    0: [
        bitstamp_thirds_undersample,
    ],
    1: [
        bitstamp_thirds_oversample,
    ],
    2: [
        coinbase_thirds_undersample,
        coinbase_thirds_oversample,
    ],
    3: [
        itbit_thirds_undersample,
        itbit_thirds_oversample,
    ],
    4: [
        okcoin_thirds_undersample,
        okcoin_thirds_oversample,
    ],
    5: [
        gemini_thirds_undersample,
        gemini_thirds_oversample,
    ],
    6: [
        bitfinex_thirds_undersample,
    ],
    7: [
        bitfinex_thirds_oversample, 
    ],
}

Spec = WorkSpec('june_1_2017', pipeline_work_units)

