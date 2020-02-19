"""
A work spec defines a set of training work to do on a variety of models/datasets, and
how to distribute that work among several different GPUs. This class also gives a nice
interface to the saved results/models/datasets from the run.
"""

import importlib
import os
import pickle

from gryphon.lib.logger import get_logger

logger = get_logger(__name__)


class WorkSpec(object):
    def __init__(self, name, pipeline_dict):
        self.name = name
        self.pipeline_dict = pipeline_dict
        self.work_units = [wu for wus in pipeline_dict.values() for wu in wus]

        for work_unit in self.work_units:
            work_unit.work_spec = self

    def get_directory(self):
        return self.name + '/'

    def get_all_model_dirs(self):
        file_paths = []

        for work_unit in self.work_units:
            file_paths = file_paths + work_unit.get_all_model_dirs()

        return file_paths

    def get_all_results_obj_paths(self):
        file_paths = []

        for work_unit in self.work_units:
            file_paths = file_paths + work_unit.get_all_results_obj_paths()

        return file_paths

    def get_all_results_objs(self, as_dict=False, filter_by=None):
        results_objs = []

        if as_dict is True:
            results_objs = {}

        for work_unit in self.work_units:
            try:
                if filter_by is not None and filter_by not in work_unit.name:
                    continue

                if as_dict is True:
                    results_objs[work_unit.name] = work_unit.get_all_results_objs()
                else:
                    results_objs = results_objs + work_unit.get_all_results_objs()
            except Exception as e:
                logger.info(e)
                logger.info(
                    'Couldn\'t get results object for work unit %s' % work_unit.name,
                )

        logger.info('Got %s of %s results objects for %s' % (
            len(results_objs),
            len(self.work_units),
            self.name,
        ))

        return results_objs

    def get_work_units_for_pipeline(self, pipe_num):
        return self.pipeline_dict[pipe_num]

    @classmethod
    def get_work_spec_object_by_name(cls, work_spec_name):
        try:
            module_name = 'ml.infra.specs.%s' % work_spec_name
            work_spec_mod = importlib.import_module(module_name)
            work_spec = work_spec_mod.Spec
        except ImportError:
            print 'Could not locate work spec "%s".' % work_spec_name
            return

        return work_spec

