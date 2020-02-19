"""
MLP using tensorflow's built-in models. This seems to work better and is much faster to
train than our custom version. That may be because it uses more complex optimization
algorithms, or it may be doing more things behind the scenes that I don't understand.

TODO:
- Double check that we're getting positive/negative labels right.
- Tensorflow is complaining about the ranks being different between the input and the
output tensors. It seems to be doing the right thing right now but I'm concerned about
the error message.
- There seem to be a ton of error messages in here, more than there needs to be. Maybe
we can cut down on them, either by conforming to the interface better or suppressing
them.
- We know for sure the method that we use to save the model is about to be deprecated.
Fix this sometime soon.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import ml.models.classifier_model
import ml.defaults as defaults


class TFLDNNClassifier(ml.models.classifier_model.TFLBinaryClassifierModel):
    def __init__(self, n_input, n_classes, hidden_layers, l1_reg=0.0, l2_reg=0.0, dropout=None, model_dir=None, summary_steps=None, checkpoint_secs=None):

        super(TFLDNNClassifier, self).__init__()

        self.n_input = n_input
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.summary_steps = summary_steps or self.DEFAULT_SUMMARY_STEPS
        self.checkpoint_secs = checkpoint_secs or self.DEFAULT_CHECKPOINT_SECS

        if model_dir is None:
            model_dir = self.create_working_directory()

        self.model_dir = model_dir

        self.estimator = self.build_model()

        self.set_as_current_model()

    def build_model(self):
        feature_columns = [
            tf.contrib.layers.real_valued_column('%s' % i)
            for i in range(0, self.n_input)
        ]

        # This is the default in tf.contrib.learn's dnn.py
        optimizer_learning_rate = 0.05

        estimator = tf.contrib.learn.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=self.hidden_layers,
            n_classes=self.n_classes,
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=optimizer_learning_rate,
                l1_regularization_strength=self.l1_reg,
                l2_regularization_strength=self.l2_reg,
            ),
            model_dir=self.model_dir,
            config=tf.contrib.learn.RunConfig(
                save_summary_steps=self.summary_steps,
                save_checkpoints_secs=self.checkpoint_secs,
            ),
            dropout=self.dropout,
        )

        return estimator

    def get_model_params(self):
        return {
            'n_input': self.n_input,
            'n_classes': self.n_classes,
            'hidden_layers': self.hidden_layers,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'dropout': self.dropout,
        }

    @classmethod
    def load_model_from_saved_params(cls, params, model_dir):
        return cls(
            params['n_input'],
            params['n_classes'],
            params['hidden_layers'],
            params['l1_reg'],
            params['l2_reg'],
            params['dropout'] if 'dropout' in params else None,
            model_dir=model_dir,
        )

    def number_of_parameters(self):
        return sum([x.size for x in self.estimator.weights_ + self.estimator.bias_])

