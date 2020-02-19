"""
This file demonstrates how to construct a work_spec to define a long training run.
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


# Model Specs.
model_specs = [
    ModelSpec('linear', linear, 100),
    ModelSpec('dnn_small', dnn_small, 1000),
]

# Work Units.
test_unit = WorkUnit(
    'test_unit',
    featuresets.simple_prices,
    featureset_params={},
    model_specs=model_specs,
)

# Work Spec.
pipeline_work_units = {
    0: [test_unit],
}

Spec = WorkSpec('test_spec', pipeline_work_units)

