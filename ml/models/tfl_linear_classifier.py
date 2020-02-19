"""
Linear classifier using the built-in models in tf.contrib.learn.
"""

import math

import tensorflow as tf
import numpy as np

import ml.defaults as defaults
import ml.models.classifier_model


class TFLLinearClassifier(ml.models.classifier_model.TFLBinaryClassifierModel):
    def __init__(self, n_input, n_classes=2, l1_reg=0.0, l2_reg=0.0, model_dir=None, summary_steps=None, checkpoint_secs=None):
        super(TFLLinearClassifier, self).__init__()

        self.n_input = n_input
        self.n_classes = n_classes
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
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

        # This is the default in tf.contrib.learn's linear.py
        optimizer_learning_rate = min(0.2, 1 / math.sqrt(self.n_input))

        estimator = tf.contrib.learn.LinearClassifier(
            feature_columns=feature_columns,
            n_classes=self.n_classes,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=optimizer_learning_rate,
                l1_regularization_strength=self.l1_reg,
                l2_regularization_strength=self.l2_reg,
            ),
            model_dir=self.model_dir,
            config=tf.contrib.learn.RunConfig(
                save_summary_steps=self.summary_steps,
                save_checkpoints_secs=self.checkpoint_secs,
            )
        )

        return estimator

    def get_model_params(self):
        return {
            'n_input': self.n_input,
            'n_classes': self.n_classes,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
        }

    @classmethod
    def load_model_from_saved_params(cls, params, model_dir):
        return cls(
            params['n_input'],
            params['n_classes'],
            params['l1_reg'],
            params['l2_reg'],
            model_dir=model_dir,
        )

    def number_of_parameters(self):
        return len(self.estimator.weights_) + len(self.estimator.bias_)

