"""
Base class and derived classes of classifier models.
"""
from datetime import datetime
import shutil
import json
import pickle
import os

import importlib
import numpy as np
import tensorflow as tf

import ml.models.model

SAVED_MODEL_DIR = 'saved_models/'
TMP_MODEL_DIR = 'tmp_models/'
METADATA_FILENAME = 'metadata.json'
RESULTS_FILENAME = 'latest_results.pickle'

class TFLModel(ml.models.model.Model):
    DEFAULT_SUMMARY_STEPS = 100
    DEFAULT_CHECKPOINT_SECS = 10

    def __init__(self):
        self.remove_temp_files = False

    def __del__(self):
        if self.remove_temp_files:
            self.clean_working_dir()

    def clean_working_dir(self):
        shutil.rmtree(self.model_dir)

    def create_working_directory(self):
        new_working_dir = self.generate_new_working_directory_name()
        os.makedirs(new_working_dir)

        return new_working_dir

    def set_as_current_model(self):
        try:
            os.unlink('current_model')
        except OSError:  # This is thrown if the symlink doesn't already exist.
            pass

        os.symlink(self.estimator.model_dir, 'current_model')

        return

    @classmethod
    def generate_new_working_directory_name(cls):
        working_dir_name = '%s_%s_%s' % (
            cls.__name__,
            datetime.now().strftime('%s'),
            np.random.randint(10e6),
        )

        working_dir_path = TMP_MODEL_DIR + working_dir_name

        return working_dir_path

    def get_validation_metrics_for_dataset(self, dataset_name):
        metrics = {}

        for key, metric in self.validation_metrics.items():
            metrics[key % dataset_name] = metric

        return metrics

    def get_monitors(self, training_data, training_labels, test_data=None, test_labels=None):
        if len(self.validation_metrics) == 0:
            return []

        monitors = []

        train_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: self.input_fn(
                training_data,
                training_labels,
            ),
            eval_steps=1,
            metrics=self.get_validation_metrics_for_dataset('train'),
            every_n_steps=self.summary_steps,
        )

        monitors.append(train_monitor)

        if test_data is not None and test_labels is not None:
            test_data, test_labels = self.prepare_dataset(
                test_data,
                test_labels,
            )

            validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                input_fn=lambda: self.input_fn(
                    test_data,
                    test_labels,
                ),
                eval_steps=1,
                metrics=self.get_validation_metrics_for_dataset('test'),
                every_n_steps=self.summary_steps,
            )

            monitors.append(validation_monitor)

        return monitors

    def train(self, training_data, training_labels, training_epochs, test_data=None, test_labels=None):
        training_data, training_labels = self.prepare_dataset(
            training_data,
            training_labels,
        )

        monitors = self.get_monitors(
            training_data,
            training_labels,
            test_data,
            test_labels,
        )

        self.estimator.fit(
            input_fn=lambda: self.input_fn(training_data, training_labels),
            steps=training_epochs,
            monitors=monitors,
        )

        return self._summarize_train_run(
            training_data,
            training_labels,
            training_epochs,
            test_data,
            test_labels,
        )

    def get_predictions_on_dataset(self, feature_data):
        predictions = self.estimator.predict(
            input_fn=lambda: self.input_fn_features_only(feature_data),
            as_iterable=False,
        )

        return predictions

    def get_summary(self, feature_data, label_data):
        summary = self.estimator.evaluate(
            input_fn=lambda: self.input_fn(feature_data, label_data),
            steps=1,
        )

        return summary

    def save_metadata(self, model_params, rescale_params, model_dir):
        """
        Save the params of this Model class in a json file in the model's directory.
        """
        filename = model_dir + METADATA_FILENAME

        metadata = {
            'model_params': model_params,
            'rescale_params': rescale_params,
            'class_info': {
                'name': self.__class__.__name__,
                'module': self.__class__.__module__,
            }
        }

        with open(filename, 'wb') as f:
            f.write(json.dumps(metadata))

        return

    @classmethod
    def load_metadata(cls, model_dir):
        """
        Load the params of saved model in model_dir from the json file.
        """
        filename = model_dir + METADATA_FILENAME

        with open(filename) as f:
            raw = f.read()

        metadata = json.loads(raw)

        return metadata

    def save_model_with_data(self, name, rescale_params, td, tl, tdt, tlt):
        self.save_model(name, rescale_params)
        self.save_train_test_set(name, td, tl, tdt, tlt)

    def save_train_test_set(self, model_name, td, tl, tdt, tlt):
        self.save_dataset(model_name, 'td', td)
        self.save_dataset(model_name, 'tl', tl)
        self.save_dataset(model_name, 'tdt', tdt)
        self.save_dataset(model_name, 'tlt', tlt)

    def save_dataset(self, model_name, dataset_name, dataset):
        model_dir = self.model_dir_for_name(model_name)
        filename = model_dir + dataset_name + '.pickle'

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    def save_results(self, model_name, results):
        model_dir = self.model_dir_for_name(model_name)
        filename = model_dir + RESULTS_FILENAME

        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def save_model(self, name, rescale_params):
        """
        Freeze's the model's current state and saves it with the given name under the
        saved_models directory.
        """
        tmp_model_dir = self.estimator.model_dir
        model_dir = self.model_dir_for_name(name)

        shutil.copytree(tmp_model_dir, model_dir)

        model_params = self.get_model_params()

        self.save_metadata(model_params, rescale_params, model_dir)

    @classmethod
    def load_latest_results(cls, model_name):
        model_dir = cls.model_dir_for_name(model_name)
        filename = model_dir + RESULTS_FILENAME

        with open(filename, 'rb') as f:
            results = pickle.load(f)

        return results

    @classmethod
    def load_full_dataset(cls, model_name):
        td = cls.load_dataset_component(model_name, 'td')
        tl = cls.load_dataset_component(model_name, 'tl')
        tdt = cls.load_dataset_component(model_name, 'tdt')
        tlt = cls.load_dataset_component(model_name, 'tlt')

        return td, tl, tdt, tlt

    @classmethod
    def load_dataset_component(cls, model_name, component_name):
        model_dir = cls.model_dir_for_name(model_name)
        filename = model_dir + component_name + '.pickle'

        with open(filename, 'rb') as f:
            dataset = pickle.load(f)

        return dataset

    @classmethod
    def load_model_with_full_dataset(cls, model_name):
        model = cls.load_model(model_name)
        td, tl, tdt, tlt = cls.load_full_dataset(model_name)

        return model, td, tl, tdt, tlt

    @classmethod
    def load_model(cls, name):
        """
        Load a previously saved model given it's name.
        """
        saved_model_dir = cls.model_dir_for_name(name)

        metadata = cls.load_metadata(saved_model_dir)

        model_params = metadata['model_params']
        model_class = cls._get_model_class(metadata)

        # Get a new working directory (shutil wants the directory to not exist so we
        # only make the name here).
        new_working_dir = model_class.generate_new_working_directory_name()

        shutil.copytree(saved_model_dir, new_working_dir)

        return model_class.load_model_from_saved_params(model_params, new_working_dir)

    @classmethod
    def _get_model_class(cls, metadata):
        class_name = metadata['class_info']['name']
        class_module = metadata['class_info']['module']

        module = importlib.import_module(class_module)
        model_class = getattr(module, class_name)

        return model_class

    @classmethod
    def model_dir_for_name(cls, name):
        return SAVED_MODEL_DIR + name + '/'

    def prepare_dataset(self, feature_data, label_data):
        """
        TFL Models want vectorized data, and classifiers want numerical representation,
        so there's no need to do anything fancy here.
        """
        feature_data = self.vectorize(feature_data)
        label_data = self.vectorize(label_data)

        return feature_data, label_data

    def set_config(self, summary_steps, checkpoint_secs):
        self.summary_steps = summary_steps
        self.checkpoint_secs = checkpoint_secs
        self.reload_model()

    def reload_model(self):
        self.estimator = self.build_model()

    def _summarize_train_run(self, training_data, training_labels, training_epochs, test_data=None, test_labels=None):
        """
        Return the results for a train run when it is complete. If validation data is
        given return the results on that set too.
        """

        train_results = self.eval_on_dataset(training_data, training_labels)

        if test_data is not None and test_labels is not None:
            test_results = self.eval_on_dataset(test_data, test_labels)

            results = {'train': train_results, 'test': test_results}

            self.print_train_test_summary(results)

            return results
        else:
            self.print_run_summary(train_results, 'Train')

            return train_results

    def print_train_test_summary(self, results):
        """
        Output the results of a train/test run in a nice way.
        """
        self.print_run_summary(results['train'], 'Train')
        self.print_run_summary(results['test'], 'Test')

    def print_run_summary(self, run_results, dataset_name=None):
        """
        Output the results of a single dataset run in a nice way.
        """

        heading_line = ''
        stats_line = ''

        summary_items = sorted(run_results['summary'].items(), key=lambda k: k[0])

        for key, val in summary_items:
            new_heading = key.ljust(10)
            heading_line = heading_line + new_heading

            new_stat = '%.4f' % val
            new_stat = new_stat.ljust(10)
            stats_line = stats_line + new_stat

        if dataset_name is not None:
            print dataset_name

        print heading_line
        print stats_line

