"""
A simple class to describe a train/validate dataset and a list of models to train on
that dataset.
"""

import copy
import os
import pickle


DEFAULT_FEATURESET_PARAMS = {
    'interpolate': True,
    'max_interpolate': 2,
    'for_classification': False,
}


class WorkUnit(object):
    def __init__(self, name, featureset, featureset_params={}, model_specs=[]):
        self.name = name
        self.featureset = featureset
        self.model_specs = copy.deepcopy(model_specs)
        self.work_spec = None

        self.featureset_params = DEFAULT_FEATURESET_PARAMS.copy()
        self.featureset_params.update(featureset_params)

        for model_spec in self.model_specs:
            model_spec.work_unit = self

    def get_directory(self):
        return self.work_spec.get_directory() + self.name + '/'

    def get_all_model_dirs(self):
        directories = []

        for model_spec in self.model_specs:
            directories.append(model_spec.get_directory())

        return directories

    def get_all_results_obj_paths(self):
        file_paths = []

        for model_spec in self.model_specs:
            file_paths.append(model_spec.get_results_obj_path())

        return file_paths

    def get_all_results_objs(self):
        results_objs = []

        for model_spec in self.model_specs:
            results_objs.append(model_spec.get_results_obj())

        return results_objs

