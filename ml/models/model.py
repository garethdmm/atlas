"""
Base class to define the interface for machine learning models.
"""
import numpy as np


class Model(object):
    def __init__(self):
        pass

    def build_model():
        raise NotImplementedError

    def train():
        raise NotImplementedError

    def eval_on_dataset(self, features, labels):
        raise NotImplementedError

    def get_predictions_on_dataset(self, features):
        raise NotImplementedError

    def train_test_summary(self, features_train, labels_train, features_test, labels_test):
        return {
            'train': self.eval_on_dataset(features_train, labels_train),
            'test': self.eval_on_dataset(features_test, labels_test),
        }

    def save_model(self, model_name):
        raise NotImplementedError

    @classmethod
    def load_model(self, model_name):
        raise NotImplementedError

    @classmethod
    def train_many(cls, n_networks, training_epochs, training_data, training_labels, test_data, test_labels, *args, **kwargs):
        multi_results = []
        models = []

        for i in range(n_networks):
            model = cls(*args, **kwargs)

            model.train(training_data, training_labels, training_epochs=training_epochs)

            results = model.train_test_summary(
                training_data,
                training_labels,
                test_data,
                test_labels,
            )

            multi_results.append(results)
            models.append(model)

        return multi_results, models

    def prepare_dataset(self, feature_data, label_data):
        raise NotImplementedError

    def vectorize(self, data):
        return np.asarray(data)

    def number_of_parameters(self):
        raise NotImplementedError

