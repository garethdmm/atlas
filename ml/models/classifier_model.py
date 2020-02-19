"""
Base class and derived classes of classifier models.
"""
import shutil
import json

import numpy as np
import tensorflow as tf

import ml.data.helpers as data_helpers
import ml.models.model
import ml.models.tfl_model
import ml.stats.binary_metrics


class BinaryClassifierModel(ml.models.model.Model):
    @classmethod
    def summary_stats(cls, predictions, reality):
        acc = ml.stats.binary_metrics.acc(predictions, reality)
        tpr = ml.stats.binary_metrics.tpr(predictions, reality)
        tnr = ml.stats.binary_metrics.tnr(predictions, reality)
        ppv = ml.stats.binary_metrics.ppv(predictions, reality)
        npv = ml.stats.binary_metrics.npv(predictions, reality)

        return acc, tpr, tnr, ppv, npv

    def get_probabilities_on_dataset(self, features):
        raise NotImplementedError


class TFLBinaryClassifierModel(BinaryClassifierModel, ml.models.tfl_model.TFLModel):
    validation_metrics = {
        "accuracy_%s": tf.contrib.learn.metric_spec.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.prediction_key.PredictionKey.CLASSES,
        ),
    }

    def get_probabilities_on_dataset(self, feature_data):
        probabilities = self.estimator.predict_proba(
            input_fn=lambda: self.input_fn_features_only(feature_data),
            as_iterable=False,
        )

        return probabilities

    def eval_on_dataset(self, feature_data, label_data):
        feature_data, label_data = self.prepare_dataset(
            feature_data,
            label_data,
        )

        summary = self.get_summary(feature_data, label_data)

        predictions = self.get_predictions_on_dataset(feature_data)

        probabilities = self.get_probabilities_on_dataset(feature_data)

        acc, tpr, tnr, ppv, npv = self.summary_stats(
            predictions,
            label_data,
        )

        return {
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
                'probabilities': probabilities,
                'reality': label_data,
            },
            'other': {
                'summary': summary,
            },
        }

        return feature_data, label_data

    def input_fn(self, data, labels):
        """
        Convert our training data into the form that tf 0.12 wants.
        """
        feature_cols = self.input_fn_features_only(data)
        labels = tf.constant(labels)

        return feature_cols, labels

    def input_fn_features_only(self, data):
        """
        Input function that only gives back the features, used for prediction.
        """
        num_features = data.T.shape[0]
        feature_cols = {
            str(i): tf.constant(data.T[i])
            for i in range(0, num_features)
        }

        return feature_cols

