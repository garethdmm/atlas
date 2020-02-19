import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import ml.models.model
import ml.defaults as defaults
import ml.models.helpers as helpers

DEFAULT_HIDDEN_LAYERS = [4, 4]  # Default to two hidden layers.

def _dnn_old(training_data, training_labels, test_data, test_labels, hidden_layers=DEFAULT_HIDDEN_LAYERS, reg=0.5, learning_rate=defaults.LEARNING_RATE, training_epochs=defaults.EPOCHS, display_step=defaults.DISPLAY_STEP):
    """
    This model conforms to the contrib.learn interface that was present up until
    tensorflow 0.11. I personally prefer it and it has much fewer warnings in it.
    """

    # Reshape data. We get classes as 1-hot vectors but in this case TF wants a
    # integers.
    training_labels = np.asarray([np.argmax(t) for t in training_labels])
    test_labels = np.asarray([np.argmax(t) for t in test_labels])

    n_input = training_data.shape[1]  # Input nodes
    n_classes = len(set(training_labels))  # Number of classes

    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=n_input)]

    nn = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_layers,
        n_classes=n_classes,
    )

    nn.fit(x=training_data, y=training_labels, steps=training_epochs)

    summary_train = nn.evaluate(x=training_data, y=training_labels)
    summary_test = nn.evaluate(x=test_data, y=test_labels)

    # Double check which label is positive/negative
    predictions_train = nn.predict(training_data)
    tpr_train = tpr(predictions_train, training_labels)
    tnr_train = tnr(predictions_train, training_labels)

    predictions_test = nn.predict(test_data)
    tpr_test = tpr(predictions_test, test_labels)
    tnr_test = tnr(predictions_test, test_labels)

    results = {
        'train': {
            'cost': summary_train['loss'],
            'accuracy': summary_train['accuracy'],
            'tpr': tpr_train,
            'tnr': tnr_train,
            'predictions': predictions_train,
            'reality': training_labels,
        },
        'test': {
            'cost': summary_test['loss'],
            'accuracy': summary_test['accuracy'],
            'tpr': tpr_test,
            'tnr': tnr_test,
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

