"""
A very simple class to wrap a single run in a WorkUnit.
"""
import os
import pickle


class ModelSpec(object):
    RESULTS_RELATIVE_DIR = os.environ['LOCAL_AZ_DIR'] + 'saved_models/'

    def __init__(self, name, model_fn, epochs):
        self.name = name
        self.model_fn = model_fn
        self.epochs = epochs

    def get_directory(self):
        return self.RESULTS_RELATIVE_DIR\
            + self.work_unit.get_directory()\
            + self.name + '/'

    def get_results_obj_path(self):
        return self.get_directory() + 'latest_results.pickle'

    def get_results_obj(self):
        results_obj_path = self.get_results_obj_path()

        with open(results_obj_path) as f:
            results_obj = pickle.load(f)

        results_obj['spec_info'] = {
            'work_spec_name': self.work_unit.work_spec.name,
            'work_unit_name': self.work_unit.name,
            'model_spec_name': self.name,
        }

        return results_obj

    def create_model(self, *args):
        return self.model_fn(*args)

