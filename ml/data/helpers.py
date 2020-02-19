import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy.orm
from sqlalchemy import func
import sklearn.metrics

import ml.data.processing as processing
from gryphon.lib.logger import get_logger
from gryphon.lib.models.atlaszero.metric import Metric
import gryphon.lib.models.atlaszero.metric_types as metric_types

logger = get_logger(__name__)


def baseline_loss_for_underlying_target(feature):
    """
    Feature is a complete index pandas series, as feature.get_data gives us.

    What would our loss be if we simply predicted that the feature stayed the same?
    """

    df = pd.concat([feature, feature.shift(-1)], axis=1).dropna()
    past = df.iloc[:, 0]
    future = df.iloc[:, 1]

    return sklearn.metrics.mean_squared_error(past, future)


def baseline_loss_for_diff_target(feature):
    """
    If the target is a future change and our baseline is predict it doesn't change, then
    our prediction is '0' every time, making the MSE math (0 - future) ** 2 / len(future)
    which is future ^ 2
    """

    future = feature.shift(-1).dropna()
    past = pd.Series(np.zeros(future.shape))

    return sklearn.metrics.mean_squared_error(past, future)


def baseline_loss_for_underlying_target_mad(feature):
    """
    Same as above but using MAE vs MSE.
    """

    df = pd.concat([feature, feature.shift(-1)], axis=1).dropna()
    past = df.iloc[:, 0]
    future = df.iloc[:, 1]

    return sklearn.metrics.mean_absolute_error(past, future)


def baseline_loss_for_diff_target_mad(feature):
    """
    Same as above but using MAE vs MSE.
    """

    future = feature.shift(-1).dropna()
    past = pd.Series(np.zeros(future.shape))

    return sklearn.metrics.mean_absolute_error(past, future)


def duplicates(db, metric_type):
    metrics_per_timestamp = db.query(Metric.timestamp, func.count(Metric))\
        .filter(Metric.metric_type == metric_type)\
        .group_by(Metric.timestamp)\
        .all()

    duplicated_timestamps = [m[0] for m in metrics_per_timestamp if m[1] != 1]

    return duplicated_timestamps


def delete_duplicates(db, series_id, execute=False):
    """
    This function cleans duplicates entries out of a series. CAREFUL! This is the
    only function in this file that makes modifications to the database so far.
    Don't use it haphazardly.
    """

    duplicated_timestamps = duplicates(db, series_id)
    logger.info('\nDeleting dupes for series %s' % series_id)

    num_metrics = db.query(Metric).filter(Metric.metric_type == series_id).count()

    for t in duplicated_timestamps:
        logger.info('For duplicated timestamp %s' % str(t))

        metrics = db.query(Metric)\
            .filter(Metric.metric_type == series_id)\
            .filter(Metric.timestamp == t)\
            .order_by(Metric.time_created.asc())\
            .all()

        assert len(metrics) >= 2
        assert len(set([m.value for m in metrics])) == 1

        old_metrics_value = metrics[1].value

        metric_to_save = metrics[0]
        metrics_to_delete = metrics[1:]

        logger.info('Saving metric:\t%s\t%s\t%s' % (
            metric_to_save.metric_id,
            metric_to_save.timestamp,
            metric_to_save.value,
        ))

        for m in metrics_to_delete:
            logger.info('Deleting metric:\t%s\t%s\t%s' % (
                m.metric_id,
                m.timestamp,
                m.value,
            ))

        if execute is True:
            for delete_m in metrics_to_delete:
                assert delete_m.value == metric_to_save.value
                assert delete_m.timestamp == metric_to_save.timestamp

                db.delete(delete_m)

            db.commit()

        new_metrics = db.query(Metric)\
            .filter(Metric.metric_type == series_id)\
            .filter(Metric.timestamp == t)\
            .order_by(Metric.time_created.asc())\
            .all()

        assert len(new_metrics) == 1
        assert new_metrics[0].value == old_metrics_value

    new_num_metrics = db.query(Metric).filter(Metric.metric_type == series_id).count()
    num_deleted_metrics = num_metrics - new_num_metrics

    logger.info('\nDeleted %s metrics' % num_deleted_metrics)
    logger.info('All looks good')

