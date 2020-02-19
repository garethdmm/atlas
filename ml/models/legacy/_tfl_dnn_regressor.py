import tensorflow as tf
import numpy as np
import pandas as pd

import ml.defaults as defaults
import ml.models.helpers as helpers
import ml.models.model

DEFAULT_HIDDEN_LAYERS = [4, 4]


def _dnn_old(training_data, training_labels, test_data, test_labels, hidden_layers=DEFAULT_HIDDEN_LAYERS, reg=0.5, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):

    n_input = training_data.shape[1]  # Input nodes

    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=n_input)]

    nn = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=hidden_layers,
    )

    nn.fit(x=training_data, y=training_labels, steps=training_epochs)

    summary_train = nn.evaluate(x=training_data, y=training_labels)
    summary_test = nn.evaluate(x=test_data, y=test_labels)

    # Double check which label is positive/negative
    predictions_train = nn.predict(training_data)
    predictions_train = pd.Series(predictions_train)
    reality_train = pd.Series(training_labels.T[0])
    accuracy_train = helpers.regression_acc(predictions_train, reality_train)

    predictions_test = nn.predict(test_data)
    predictions_test = pd.Series(predictions_test)
    reality_test = pd.Series(test_labels.T[0])
    accuracy_test = helpers.regression_acc(predictions_test, reality_test)

    results = {
        'train': {
            'cost': summary_train['loss'],
            'acc': accuracy_train,
            'predictions': predictions_train,
            'reality': training_labels,
        },
        'test': {
            'cost': summary_test['loss'],
            'acc': accuracy_test,
            'predictions': predictions_test,
            'reality': test_labels,
        },
        'model': {
            'weights': nn.weights_,
            'bias': nn.bias_,
            'object': nn,
        }
    }

    return results


