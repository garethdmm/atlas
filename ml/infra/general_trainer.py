"""
This script allows us to specify lots of different types of training runs for our GPU
box to do, for example, overnight. Runs are defined in files like test_spec.py, and
documentation about how to define a run is in that file. This script handles turning
that spec into actual models, portioning work out to the available GPUs, getting the
training data, training the models, and saving the output of the runs in a meaningful
directory structure for later analysis. Pretty sweet.

The directory structure ends up looking like this:
  saved_models/
    run_name/
      work_unit_name/
        model_spec_name/
          results.pickle
          model.ckpt
          ...

e.g. saved_models/thursday_night/bitstamp/dnn_small/

TODO:
    - Make the interface to this nicer, probably access through ./atlas-zero trainer or
      something like that.
"""

from datetime import datetime
import importlib
import os
import sys

import ml.data.preconfigured_feature_label_sets as featuresets
import ml.infra.work_spec
from gryphon.lib import session
from gryphon.lib.logger import get_logger


logger = get_logger(__name__)

TEST_EPOCHS = 5
TEST_FEATURESET = featuresets.simple_prices


def train_and_save_model(td, tl, tdt, tlt, rescale, model, epochs, run_name, model_prefix, model_spec_name):
    model_name = '%s/%s/%s' % (
        run_name,
        model_prefix,
        model_spec_name,
    )

    logger.info('Training Model: %s' % model_name)

    model.set_config(300, 30)
    results = model.train(td, tl, epochs, tdt, tlt)

    model.save_model_with_data(model_name, rescale, td, tl, tdt, tlt)
    model.save_results(model_name, results)

    logger.info('Done.')


def run_pipeline(work_spec, pipeline_num, execute):
    pipe_start = datetime.now()

    pipe_units = work_spec.get_work_units_for_pipeline(pipeline_num)

    for work_unit in pipe_units:
        featureset = work_unit.featureset if execute is True else TEST_FEATURESET

        db = session.get_a_mysql_session(os.environ['ATLAS_ZERO_DB_CRED'])

        logger.info('Getting featureset')

        try:
            featureset_start = datetime.now()

            td, tl, tdt, tlt, rescale = featureset\
                .get_data_for_train_test_run(db, **(work_unit.featureset_params))

            featureset_end = datetime.now()
            print 'Featureset took: %s' % str(featureset_end - featureset_start)

        finally:
            db.remove()

        for model_spec in work_unit.model_specs:
            model = model_spec.create_model(td)
            model.remove_temp_files = True

            train_and_save_model(
                td,
                tl,
                tdt,
                tlt,
                rescale,
                model,
                model_spec.epochs if execute is True else TEST_EPOCHS,
                work_spec.name,
                work_unit.name,
                model_spec.name,
            )

    pipe_end = datetime.now()

    print 'Finished pipeline %s' % (pipeline_num)
    print 'Total time: %s' % str(pipe_end - pipe_start)


def set_up_tensorflow_and_configure_gpu(gpu_num):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % gpu_num

    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)


def do_confirmation_prompt(run_name, pipeline_num):
    print 'Running spec "%s" pipeline %s' % (run_name, pipeline_num)
    print 'Ready? (q to cancel)'

    response = raw_input()

    return response


def main():
    work_spec_name = sys.argv[1]
    pipeline_num = int(sys.argv[2])
    execute = sys.argv[3] == 'True'

    work_spec = ml.infra.work_spec.WorkSpec.get_work_spec_object_by_name(work_spec_name)

    response = do_confirmation_prompt(work_spec_name, pipeline_num)

    if response == 'q':
        return

    set_up_tensorflow_and_configure_gpu(pipeline_num)

    run_pipeline(work_spec, pipeline_num, execute)


if __name__ == '__main__':
    main()


