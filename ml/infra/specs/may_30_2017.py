"""
Overnight training for may 27 (or maybe 28). Initial multiclassification investigations
into, including experiments with different resampling techniques.

- All exchanges.
- Small network sizes.
- 3 class classification
  - with borders at 0.33, 0.67
    - no resample
    - oversampled
    - undersampled
  - with borders at 0.10, 0.90
    - no resample
    - oversampled
    - undersampled
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
    ModelSpec('linear', linear, 100),
    ModelSpec('dnn1', dnn1, 10000),
    ModelSpec('dnn2', dnn2, 10000),
    ModelSpec('dnn3', dnn3, 10000),
    ModelSpec('dnn6', dnn6, 10000),
]


# Work Units.
bitstamp_thirds_nobalance = WorkUnit(
    'bitstamp_thirds_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': False,
    },
    model_specs=model_specs,
)

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

bitstamp_deciles_nobalance = WorkUnit(
    'bitstamp_deciles_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': False,
    },
    model_specs=model_specs,
)

bitstamp_deciles_undersample = WorkUnit(
    'bitstamp_deciles_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

bitstamp_deciles_oversample = WorkUnit(
    'bitstamp_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitstamp,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Bitfinex.

bitfinex_thirds_nobalance = WorkUnit(
    'bitfinex_thirds_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': False,
    },
    model_specs=model_specs,
)

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

bitfinex_deciles_nobalance = WorkUnit(
    'bitfinex_deciles_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': False,
    },
    model_specs=model_specs,
)

bitfinex_deciles_undersample = WorkUnit(
    'bitfinex_deciles_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

bitfinex_deciles_oversample = WorkUnit(
    'bitfinex_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_bitfinex,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Coinbase.

coinbase_thirds_nobalance = WorkUnit(
    'coinbase_thirds_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': False,
    },
    model_specs=model_specs,
)

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

coinbase_deciles_nobalance = WorkUnit(
    'coinbase_deciles_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': False,
    },
    model_specs=model_specs,
)

coinbase_deciles_undersample = WorkUnit(
    'coinbase_deciles_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

coinbase_deciles_oversample = WorkUnit(
    'coinbase_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_coinbase,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# Itbit.

itbit_thirds_nobalance = WorkUnit(
    'itbit_thirds_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': False,
    },
    model_specs=model_specs,
)

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

itbit_deciles_nobalance = WorkUnit(
    'itbit_deciles_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': False,
    },
    model_specs=model_specs,
)

itbit_deciles_undersample = WorkUnit(
    'itbit_deciles_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'undersample',
    },
    model_specs=model_specs,
)

itbit_deciles_oversample = WorkUnit(
    'itbit_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_itbit,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'oversample',
    },
    model_specs=model_specs,
)

# OkCoin.

okcoin_thirds_nobalance = WorkUnit(
    'okcoin_thirds_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': False,
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_thirds_undersample = WorkUnit(
    'okcoin_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
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
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_deciles_nobalance = WorkUnit(
    'okcoin_deciles_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': False,
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_deciles_undersample = WorkUnit(
    'okcoin_deciles_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'undersample',
        'TRAIN_START': OKCOIN_TRAIN_START,
        'TRAIN_END': OKCOIN_TRAIN_END,
        'TEST_START': OKCOIN_TEST_START,
        'TEST_END': OKCOIN_TEST_END,
    },
    model_specs=model_specs,
)

okcoin_deciles_oversample = WorkUnit(
    'okcoin_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_okcoin,
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

gemini_thirds_nobalance = WorkUnit(
    'gemini_thirds_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': False,
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_thirds_undersample = WorkUnit(
    'gemini_thirds_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.33, 0.67],
        'rebalance': True,
        'rebalance_method': 'undersample',
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
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
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_deciles_nobalance = WorkUnit(
    'gemini_deciles_nobalance',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': False,
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_deciles_undersample = WorkUnit(
    'gemini_deciles_undersample',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
    featureset_params={
        'for_classification': True,
        'percentiles': [0.10, 0.90],
        'rebalance': True,
        'rebalance_method': 'undersample',
        'TRAIN_START': GEMINI_TRAIN_START,
        'TRAIN_END': GEMINI_TRAIN_END,
        'TEST_START': GEMINI_TEST_START,
        'TEST_END': GEMINI_TEST_END,
    },
    model_specs=model_specs,
)

gemini_deciles_oversample = WorkUnit(
    'gemini_deciles_oversample',
    featuresets.ultra_strength_inner_1d_target_price_diff_gemini,
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


# 36 work units, 8 pipelines, 4 have 5 wus' and 4 have 4 wu's.
pipeline_work_units = {
    0: [
        bitstamp_thirds_nobalance,
        bitstamp_thirds_undersample,
        bitstamp_thirds_oversample,
        bitstamp_deciles_nobalance,
        bitstamp_deciles_undersample,
    ],
    1: [
        bitstamp_deciles_oversample,
        bitfinex_thirds_nobalance,
        bitfinex_thirds_undersample,
        bitfinex_thirds_oversample,
        bitfinex_deciles_nobalance,
    ],
    2: [
        bitfinex_deciles_undersample,
        bitfinex_deciles_oversample,
        coinbase_thirds_nobalance,
        coinbase_thirds_undersample,
        coinbase_thirds_oversample,
    ],
    3: [
        coinbase_deciles_nobalance,
        coinbase_deciles_undersample,
        coinbase_deciles_oversample,
        itbit_thirds_nobalance,
        itbit_thirds_undersample,
    ],
    4: [
        itbit_thirds_oversample,
        itbit_deciles_nobalance,
        itbit_deciles_undersample,
        itbit_deciles_oversample,
    ],
    5: [
        okcoin_thirds_nobalance,
        okcoin_thirds_undersample,
        okcoin_thirds_oversample,
        okcoin_deciles_nobalance,
    ],
    6: [
        okcoin_deciles_undersample,
        okcoin_deciles_oversample,
        gemini_thirds_nobalance,
        gemini_thirds_undersample,
    ],
    7: [
        gemini_thirds_oversample,
        gemini_deciles_nobalance,
        gemini_deciles_undersample,
        gemini_deciles_oversample,
    ],
}

Spec = WorkSpec('may_30_2017', pipeline_work_units)

