"""
Base class and derived classes of regression models.
"""
import shutil
import json

import numpy as np
import pandas as pd
import tensorflow as tf

import ml.models.model
import ml.models.tfl_model
import ml.data.conversion as conversion
import ml.stats.binary_metrics


class RegressorModel(ml.models.model.Model):
    @classmethod
    def summary_stats(cls, predictions, reality):
        # hack currently
        predictions = np.array(conversion.decimal_labels_to_binary_categorical(predictions).values)
        reality = np.array(conversion.decimal_labels_to_binary_categorical(reality).values)

        acc = ml.stats.binary_metrics.acc(predictions, reality)
        tpr = ml.stats.binary_metrics.tpr(predictions, reality)
        tnr = ml.stats.binary_metrics.tnr(predictions, reality)
        ppv = ml.stats.binary_metrics.ppv(predictions, reality)
        npv = ml.stats.binary_metrics.npv(predictions, reality)

        return acc, tpr, tnr, ppv, npv


class TFLRegressorModel(RegressorModel, ml.models.tfl_model.TFLModel):
    validation_metrics = {
        "loss_%s": tf.contrib.metrics.streaming_mean_squared_error,
        "mae_%s": tf.contrib.metrics.streaming_mean_absolute_error,
    }

    def eval_on_dataset(self, feature_data, label_data):
        feature_data, label_data = self.prepare_dataset(
            feature_data,
            label_data,
        )

        summary = self.get_summary(feature_data, label_data)

        predictions = self.get_predictions_on_dataset(feature_data)

        acc, tpr, tnr, ppv, npv = self.summary_stats(
            predictions,
            label_data,
        )

        reality = label_data.T

        results = {
            'summary': {
                'cost': summary['loss'],
                'acc': acc,
                'tpr': tpr,
                'tnr': tnr,
                'ppv': ppv,
                'npv': npv,
            },
            'series': {
                'predictions': predictions,
                'reality': reality,
            },
            'other': {
                'summary': summary,
            },
        }

        return results

    def input_fn(self, data, labels):
        """
        Convert our training data into the form that tf 0.12 wants.
        """
        feature_cols = self.input_fn_features_only(data)
        labels = tf.cast(tf.constant(labels), tf.float32)

        return feature_cols, labels

    def input_fn_features_only(self, data):
        """
        Input function that only gives back the features, used for prediction.
        """
        num_features = data.T.shape[0]
        feature_cols = {str(i): tf.cast(tf.constant(data.T[i]), tf.float32) for i in range(0, num_features)}

        return feature_cols

